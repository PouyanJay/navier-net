"""Stage-A trainer.

Objective: alpha supervision + VOF transport + continuity with an inferred
dilatation source + velocity boundary conditions.

Two details worth knowing:

**Resumable by default.** Every call continues from the run's checkpoint if one
exists, so a long run can be taken in chunks (``training.steps=500`` three
times is equivalent to ``training.steps=1500`` once) and interrupted work is
never lost.

**Gradient-norm loss rebalancing.** Hand-picked loss weights on a multi-term
PINN objective tend to let one term dominate. Periodically the per-term
gradient norms are measured and the weights nudged so each term contributes
comparably, relative to the data term. The measurement pass happens *before*
the optimising backward pass, and its gradients are discarded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from naviernet.config.schema import resolved_datasets, training_datasets
from naviernet.data.adaptive import rad_resample
from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BoundPINN, BubblePINN
from naviernet.physics import kinematics, registry, weighting
from naviernet.physics.groups import N_COND, compute_groups, conditioning_vector
from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths

log = get_logger(__name__)

# The residual-adaptive (RAR) candidate pool is this many times the collocation size:
# a broad pool to score by residual and draw the high-residual sample from. Only built
# every `resample_every` steps, so the extra forward pass is amortised.
CANDIDATE_POOL_MULT = 8

# Loss terms whose weights the rebalancer adjusts, derived from the registry for
# the Stage-A field set: `data` is the reference scale and `src` is a deliberate
# soft penalty, so both stay out. Kept as a module constant for callers that
# rebalance a Stage-A weight dict directly.
STAGE_A_FIELDS = ("phi", "u", "v", "s")
REBALANCED_TERMS = registry.rebalanced_terms(registry.enabled_equations(STAGE_A_FIELDS))


def _pulse_groups(groups: dict, cfg, domain) -> dict:
    """Augment ``groups`` with the nucleation pulse's fixed geometry, converted from the
    config's fractions to the model's absolute non-dimensional units via the dataset's
    domain, so the energy residual can read them. A no-op when the pulse is off."""
    if not getattr(cfg.model, "nucleation_pulse", False):
        return groups
    return {
        **groups,
        "x_pin_star": domain.x_pin,
        "pulse_t0": domain.t_min + cfg.model.pulse_t0 * (domain.t_max - domain.t_min),
        "pulse_sigma": cfg.model.pulse_width * (domain.x_max - domain.x_min),
    }


def _pin_anchor(cfg, data: BubbleDataset) -> tuple[float, float] | None:
    """The dataset's measured bubble-root anchor when the hard pin is on, else
    ``None``. One helper so every construction site derives the anchor the same
    way -- a site passing nothing would fail loudly in the model constructor."""
    if not getattr(cfg.model, "hard_pin", False):
        return None
    x0, y0 = data.pin_anchor
    log.info(
        "hard pin on: root anchor (x*=%.4f, y*=%.4f), d_ref=%.3f (stored x_pin=%.4f)",
        x0,
        y0,
        cfg.model.pin_d_ref,
        data.domain.x_pin,
    )
    return x0, y0


def _geometry_priors(cfg, data: BubbleDataset):
    """The dataset's measured geometry anchors when the front geometry is on,
    else ``None`` -- the construction-site helper, like :func:`_pin_anchor`."""
    if not getattr(cfg.model, "front_geometry", False):
        return None
    from naviernet.models.geometry import GeometryPriors

    x0, y0 = data.pin_anchor
    d = data.domain
    log.info(
        "front geometry on: root (x*=%.4f, y*=%.4f), nose starts at x*=%.4f",
        x0,
        y0,
        data.initial_front,
    )
    return GeometryPriors(x0, y0, data.initial_front, d.y_min, d.y_max, d.t_min, d.t_max)


def _pin_record(cfg) -> dict:
    """The architecture facts persisted in every checkpoint (hard pin and front
    geometry add no detectable state-dict signature for their *configuration*),
    so a later invocation can be checked against how the run was trained."""
    hard_pin = bool(getattr(cfg.model, "hard_pin", False))
    return {
        "hard_pin": hard_pin,
        "pin_d_ref": float(cfg.model.pin_d_ref) if hard_pin else None,
        "front_geometry": bool(getattr(cfg.model, "front_geometry", False)),
    }


def _check_pin_compat(cfg, ckpt: dict, path) -> None:
    """Refuse to consume a checkpoint whose hard-pin architecture disagrees with
    the composed config. The pin gate adds NO parameters, so ``load_state_dict``
    succeeds silently either way -- without this guard, evaluating or resuming a
    pinned run without ``model.hard_pin=true`` (or vice versa) would quietly load
    the weights into a differently-shaped solution space. Checkpoints written
    before the record existed pass unchecked (nothing to compare)."""
    saved = ckpt.get("hard_pin")
    if saved is None:
        return
    current = bool(getattr(cfg.model, "hard_pin", False))
    if bool(saved) != current:
        raise ValueError(
            f"{path} was trained with model.hard_pin={bool(saved)} but this invocation "
            f"composes model.hard_pin={current}. The pin is architectural: pass the "
            f"same model.hard_pin override the run was trained with."
        )
    saved_d_ref = ckpt.get("pin_d_ref")
    if current and saved_d_ref is not None and float(saved_d_ref) != float(cfg.model.pin_d_ref):
        raise ValueError(
            f"{path} was trained with model.pin_d_ref={saved_d_ref} but this invocation "
            f"composes model.pin_d_ref={float(cfg.model.pin_d_ref)}; pass the value the "
            f"run was trained with."
        )
    saved_geo = ckpt.get("front_geometry")
    current_geo = bool(getattr(cfg.model, "front_geometry", False))
    if saved_geo is not None and bool(saved_geo) != current_geo:
        raise ValueError(
            f"{path} was trained with model.front_geometry={bool(saved_geo)} but this "
            f"invocation composes model.front_geometry={current_geo}. The geometry is "
            f"architectural: pass the same override the run was trained with."
        )


def _validate_training_config(tcfg, fields) -> None:
    """Every up-front training-config validation, shared by both loops."""
    _validate_causal(tcfg)
    _validate_weighting(tcfg)
    _validate_kinematics(tcfg, fields)


def _validate_kinematics(tcfg, fields) -> None:
    """Reject a degenerate kinematics config up front, loudly."""
    if not tcfg.kinematics:
        return
    if tcfg.kin_grid < 2:
        raise ValueError(f"training.kin_grid={tcfg.kin_grid} must be >= 2 (a quadrature grid)")
    if tcfg.kin_times < 1:
        raise ValueError(f"training.kin_times={tcfg.kin_times} must be >= 1")
    if not 0.0 < tcfg.kin_time_frac <= 1.0:
        raise ValueError(f"training.kin_time_frac={tcfg.kin_time_frac} must be in (0, 1]")
    for key in (
        "kin_weight_mono",
        "kin_weight_balance",
        "kin_weight_evap",
        "kin_margin_frac",
        "kin_evap_floor",
        "kin_ramp",
    ):
        if getattr(tcfg, key) < 0:
            raise ValueError(f"training.{key}={getattr(tcfg, key)} must be >= 0")
    if tcfg.kin_weight_evap > 0 and "T" not in fields:
        raise ValueError(
            "training.kin_weight_evap > 0 needs the 'T' field (the evaporation drive "
            "reads the temperature). Use the Stage-B model, or set "
            "training.kin_weight_evap=0 to run mono+balance only."
        )


def _kin_ramp(step: int, start_step: int, ramp_steps: int) -> float:
    """Linear ramp from ``start_step`` over ``ramp_steps`` steps (0 disables the
    ramp; before ``start_step`` the factor is 0)."""
    if step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, (step - start_step + 1) / ramp_steps)


def _kin_schedule(step: int, first_step: int, tcfg) -> kinematics.KinematicSchedule:
    """The per-step ramp factors for the source-reading kinematic terms.

    Balance ramps from this call's first step (the source field is noise
    early). The evap floor additionally waits out the Stage-B warm-up: until it
    ends the temperature net has received zero gradient, and flooring the
    source against an untrained temperature pattern would bias exactly the
    late-window behaviour the term exists to fix."""
    balance = _kin_ramp(step, first_step, tcfg.kin_ramp)
    evap_start = max(first_step, int(tcfg.stage_b_warmup_steps) + 1)
    return kinematics.KinematicSchedule(
        balance=balance, evap=_kin_ramp(step, evap_start, tcfg.kin_ramp)
    )


def _kinematic_plan(cfg, data: BubbleDataset, device) -> kinematics.KinematicPlan | None:
    """The run's fixed kinematic quadrature when the constraints are on, else
    ``None``. Built once (deterministic, no RNG) so resume reproduces it."""
    tcfg = cfg.training
    if not tcfg.kinematics:
        return None
    rate = data.supervised_growth_rate
    plan = kinematics.build_plan(data.domain, rate, tcfg, device)
    log.info(
        "kinematics on: %d x %d^2 quadrature over t* >= %.3f, r_ref=%.4f",
        plan.n_times,
        int(tcfg.kin_grid),
        float(plan.points[:, 2].min()),
        rate,
    )
    return plan


def _initial_state(cfg, equations) -> dict:
    weights = cfg.training.weights
    w = {"data": float(weights.data)}
    for eq in equations:
        w[eq.weight_key] = float(getattr(weights, eq.weight_key))
    return {"done": 0, "hist": [], "w": w}


def _gradient_norms(model, losses: dict[str, torch.Tensor], opt) -> dict[str, float]:
    """Per-term gradient norms, measured with throwaway backward passes."""
    norms = {}
    for name, loss in losses.items():
        opt.zero_grad()
        loss.backward(retain_graph=True)
        total = sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)
        norms[name] = float(torch.sqrt(total)) + 1e-12
    opt.zero_grad()
    return norms


def _curriculum(step: int, curriculum_steps: int) -> tuple[float, float]:
    """Soft -> hard evaporation schedule: ``(src_factor, evap_factor)``.

    Over ``curriculum_steps`` the off-interface source penalty decays 1 -> 0 while
    the evaporation mass-closure ramps 0 -> 1, so the converged state is the hard
    closure without the early gradient shock. ``0`` disables the schedule.
    """
    if curriculum_steps <= 0:
        return 1.0, 1.0
    frac = min(1.0, step / curriculum_steps)
    return 1.0 - frac, frac


def _loss_schedule(step: int, tcfg, stage_b_keys: tuple[str, ...]) -> dict[str, float]:
    """Per-term loss multipliers at ``step``: the evaporation curriculum plus the
    Stage-B warm-up gate.

    While ``step <= stage_b_warmup_steps`` the Stage-B terms (momentum, energy,
    evaporation) are held at zero so the model fits the interface on the Stage-A
    objective first; adding those terms to a random field collapses it. After the
    gate they act at full weight (evaporation still following the curriculum). A
    term absent from the returned dict keeps its static weight (multiplier 1).

    NB the curriculum's ``frac`` is measured from the absolute step, so combining a
    non-zero ``curriculum_steps`` with the warm-up gate does not ramp evaporation
    across the *post*-gate steps -- if ``curriculum_steps <= stage_b_warmup_steps``
    it is already saturated when the gate lifts. The shipped path leaves
    ``curriculum_steps=0`` (the warm-up is the schedule); set both by hand only
    knowingly."""
    src_factor, evap_factor = _curriculum(step, tcfg.curriculum_steps)
    schedule = {"src": src_factor, "evap": evap_factor}
    if step <= tcfg.stage_b_warmup_steps:
        for key in stage_b_keys:
            schedule[key] = 0.0
    return schedule


def _causal_weights(chunk_losses: torch.Tensor, eps: float) -> torch.Tensor:
    """Temporal causal weights (Wang et al., arXiv:2203.07404).

    Given the PDE residual aggregated into time-ordered chunks
    ``chunk_losses[i] = L_r(t_i)`` (with ``t_0 < t_1 < ...``), a chunk is weighted
    down until all *earlier* chunks are satisfied, so the network learns forward in
    time instead of fitting late times on an unconverged early solution (the bias
    that hurts extrapolation to held-out late frames):

        w_i = exp(-eps * sum_{k<i} L_r(t_k))

    The weights are **detached** -- they steer which residuals matter now but carry
    no gradient. ``eps=0`` returns all ones (uniform-in-time, i.e. today's loss)."""
    with torch.no_grad():
        exclusive_prior = torch.cumsum(chunk_losses, dim=0) - chunk_losses
        return torch.exp(-eps * exclusive_prior)


def _causal_eps(step: int, first_step: int, last_step: int, schedule: list[float]) -> float:
    """The causal weighting ``eps`` for ``step``, annealed across this call's step
    window ``[first_step, last_step]``.

    ``eps`` climbs through ``schedule`` as training progresses -- a small ``eps``
    early (near uniform-in-time) steepening to a large one late -- so the network
    first fits the whole trajectory loosely, then enforces causality hard once it
    has a coarse solution to march forward from (Wang et al., arXiv:2203.07404). An
    empty schedule disables the weighting (``eps=0`` -> uniform, i.e. today's loss).

    Progress is measured over the current ``train``/``_train_joint`` invocation, not
    a stored run total (the trainer keeps none): a single-call run -- the recipe the
    benchmark and every UI launch use -- anneals cleanly across the whole schedule,
    and a chunked/resumed run anneals within each chunk rather than jumping to the
    final ``eps`` (which dividing by a per-call step count would do).
    """
    if not schedule:
        return 0.0
    frac = min(1.0, max(0.0, (step - first_step) / max(1, last_step - first_step)))
    return float(schedule[min(len(schedule) - 1, int(frac * len(schedule)))])


def _time_chunks(x_coll: torch.Tensor, n_chunks: int) -> list[torch.Tensor]:
    """``x_coll`` split into ``n_chunks`` time-ordered batches (ascending in the time
    column, index 2).

    Each batch is a fresh leaf with ``requires_grad`` set, so the PDE residuals
    differentiate through it exactly as they do the full collocation batch. The
    chunks are contiguous and near-equal in size, so no bin is empty and -- at
    ``eps=0`` -- the causal loss reduces to the ordinary collocation mean.
    """
    time = x_coll.detach()[:, 2]
    ordered = x_coll.detach()[torch.argsort(time)]
    return [
        chunk.clone().requires_grad_(True)
        for chunk in torch.chunk(ordered, max(1, n_chunks), dim=0)
        if chunk.shape[0] > 0
    ]


def _causal_collocation_loss(
    model,
    x_coll: torch.Tensor,
    groups: dict[str, float],
    c: torch.Tensor | None,
    coll_equations,
    weights: dict[str, float],
    schedule: dict[str, float],
    n_chunks: int,
    eps: float,
) -> torch.Tensor:
    """Time-causal aggregation of the collocation residual (Wang et al.).

    ``x_coll`` is binned into ``n_chunks`` time-ordered groups; within a bin the
    active PDE terms are summed with their per-term weight and warm-up schedule
    (exactly the multipliers the uniform path applies) into ``L_r(t_i)``. Each bin
    is then scaled by the detached causal weight ``w_i = exp(-eps*sum_{k<i} L_r(t_k))``
    so a late-time bin only matters once the earlier bins are satisfied. Returns
    ``mean_i(w_i * L_r(t_i))``.

    ``coll_equations`` is always non-empty on any real run (the Stage-A terms
    vof/div/src are ``core`` and enabled whenever a field set is present), so the
    ``torch.stack`` below always has bins to stack.

    Cost note: the bins together hold ``x_coll``'s points, so this pass is ~1x the
    uniform path on its own. But it is a *second* collocation pass -- the caller
    keeps the uniform per-term means (for logging and the rebalancer) -- so enabling
    causal weighting roughly doubles the per-step collocation compute. That is the
    deliberate Phase-1/Phase-2 isolation (the rebalancer still sees plain means);
    Phase 2 collapses the two passes when it replaces the gradient-norm rebalancer.
    """
    chunk_losses = torch.stack(
        [
            sum(
                weights[eq.weight_key]
                * schedule.get(eq.weight_key, 1.0)
                * eq.term(registry.LossContext(model, chunk, groups=groups, c=c))
                for eq in coll_equations
            )
            for chunk in _time_chunks(x_coll, n_chunks)
        ]
    )
    return (_causal_weights(chunk_losses, eps) * chunk_losses).mean()


def _march_active_bins(progress: float, n_chunks: int, full_frac: float) -> int:
    """How many of the earliest time bins the marching horizon covers at ``progress``
    (a fraction in ``[0, 1]`` of the way through the run).

    One bin at the start, expanding to all ``n_chunks`` by the time ``progress``
    reaches ``full_frac`` (the ``1 + int(...)`` discretisation reaches full width a
    little before that point) and staying there -- a monotonic forward-in-time
    curriculum.
    """
    expanded = min(1.0, max(0.0, progress) / max(1e-9, full_frac))
    return min(n_chunks, 1 + int(expanded * n_chunks))


def _time_march_collocation_loss(
    model,
    x_coll: torch.Tensor,
    groups: dict[str, float],
    c: torch.Tensor | None,
    coll_equations,
    weights: dict[str, float],
    schedule: dict[str, float],
    n_chunks: int,
    progress: float,
    full_frac: float,
) -> torch.Tensor:
    """Time-marching curriculum (Wang et al., arXiv:2203.07404).

    Enforce the collocation residual only up to a time horizon that expands from the
    earliest bin to the whole domain across the run (:func:`_march_active_bins`).
    Within the active horizon every term is at full weight -- unlike causal
    *weighting*, which keeps late-time residuals suppressed throughout -- so once the
    front reaches the late frames they are trained at full strength. Returns the plain
    per-term weighted, scheduled collocation loss over the active points.
    """
    bins = _time_chunks(x_coll, n_chunks)
    active = _march_active_bins(progress, len(bins), full_frac)
    x_active = torch.cat([b.detach() for b in bins[:active]]).requires_grad_(True)
    ctx = registry.LossContext(model, x_active, groups=groups, c=c)
    return sum(
        weights[eq.weight_key] * schedule.get(eq.weight_key, 1.0) * eq.term(ctx)
        for eq in coll_equations
    )


def _weighted_sum(
    losses: dict[str, torch.Tensor],
    weights: dict[str, float],
    schedule: dict[str, float],
    skip: frozenset[str] = frozenset(),
) -> torch.Tensor:
    """The per-term-weighted, scheduled sum of ``losses``, omitting ``skip`` keys.

    ``skip`` carries the collocation terms when causal weighting supplies them
    separately, leaving the supervised (``data``) and boundary (``bc``) terms.
    """
    return sum(
        weights[name] * schedule.get(name, 1.0) * loss
        for name, loss in losses.items()
        if name not in skip
    )


@dataclass(frozen=True)
class CollocationBatch:
    """One dataset's collocation batch for the causal pass: the model to evaluate
    it with (that dataset's pin-bound view on a joint hard-pin run, the plain model
    otherwise), the points, and the dataset's groups/conditioning row."""

    model: BubblePINN | BoundPINN
    x: torch.Tensor
    groups: dict
    c: torch.Tensor | None


@dataclass(frozen=True)
class _LossPlan:
    """Per-run-constant inputs to :func:`_total_loss`.

    ``coll_equations`` are the equations evaluated on the collocation points and
    ``coll_keys`` their weight keys; ``first_step``/``last_step`` bound this call's
    step window for the causal ``eps`` schedule; ``tcfg`` is the training config.
    Built once before the training loop so the per-step call stays small.
    """

    coll_equations: list
    coll_keys: frozenset[str]
    first_step: int
    last_step: int
    tcfg: object


CAUSAL_MODES = ("weight", "march")


def _validate_causal(tcfg) -> None:
    """Reject an unusable causal config up front, so a typo or a degenerate value fails
    loudly rather than silently falling through to one variant or defeating the
    curriculum. Only checked when causal weighting is actually engaged."""
    if not tcfg.causal_weighting:
        return
    if tcfg.causal_mode not in CAUSAL_MODES:
        raise ValueError(
            f"training.causal_mode={tcfg.causal_mode!r} is not one of {CAUSAL_MODES}"
        )
    # 0 or negative collapses the marching horizon to the full domain on the first
    # step (via the div-by-zero guard in _march_active_bins), silently disabling the
    # curriculum; >1 ("never fully expand within the run") is a valid deliberate choice.
    if tcfg.causal_mode == "march" and tcfg.causal_march_full_frac <= 0:
        raise ValueError(
            f"training.causal_march_full_frac={tcfg.causal_march_full_frac} must be > 0"
        )


WEIGHTING_MODES = ("gradnorm", "rba")


def _validate_weighting(tcfg) -> None:
    """Reject an unusable weighting config up front: an unknown scheme, or the not-yet
    supported RBA x causal-weighting combination (their composition is Phase 4)."""
    if tcfg.weighting not in WEIGHTING_MODES:
        raise ValueError(
            f"training.weighting={tcfg.weighting!r} is not one of {WEIGHTING_MODES}"
        )
    if tcfg.weighting == "rba" and tcfg.causal_weighting:
        raise ValueError(
            "training.weighting='rba' with training.causal_weighting=true is not "
            "supported yet (RBA x causal composition is Phase 4); disable one."
        )
    if tcfg.adaptive_collocation:
        if tcfg.weighting != "rba":
            raise ValueError(
                "training.adaptive_collocation=true refreshes the RBA fixed pool, so it "
                "requires training.weighting='rba'."
            )
        if tcfg.resample_every <= 0:
            raise ValueError(
                f"training.resample_every={tcfg.resample_every} must be > 0 "
                f"(it is the step period between residual-adaptive pool refreshes)."
            )
        if not 0.0 <= tcfg.resample_fraction <= 1.0:
            raise ValueError(
                f"training.resample_fraction={tcfg.resample_fraction} must be in [0, 1]."
            )


def _init_attention(state: dict, coll_keys, n_coll: int, device) -> dict[str, torch.Tensor]:
    """The persistent per-point RBA attention for each collocation term: carried from
    the checkpoint when resuming an RBA run, otherwise zeros (a fresh run == the plain
    mean-squared residual). Shape ``(n_coll, 1)`` per term."""
    saved = state.get("attention") or {}
    for key, tensor in saved.items():
        if tensor.shape[0] != n_coll:
            raise ValueError(
                f"checkpoint RBA attention for {key!r} has {tensor.shape[0]} points but "
                f"training.n_coll={n_coll}; resume with the n_coll the run was built with, "
                f"or delete the checkpoint to start fresh."
            )
    return {
        key: (saved[key].to(device) if key in saved else torch.zeros(n_coll, 1, device=device))
        for key in coll_keys
    }


def _residual_magnitude(ctx: registry.LossContext, coll_equations) -> torch.Tensor:
    """Per-point residual magnitude ``sqrt(sum_k r_k^2)`` over the collocation terms,
    detached -- the score residual-adaptive resampling draws high-residual points by."""
    total_sq = sum(eq.pointwise(ctx) for eq in coll_equations)
    return total_sq.detach().sqrt().squeeze(-1)


def _resample_pool(model, data, groups, coll_equations, tcfg, rng) -> torch.Tensor:
    """A residual-adaptively refreshed collocation pool (RAD, :mod:`naviernet.data.adaptive`).

    Scores a broad candidate pool by residual magnitude under the current model and
    draws ``resample_fraction`` of the points from the high-residual regions plus a
    uniform base, so the pool tracks the moving interface and stiff late times while
    keeping global coverage. Returned detached (its own fixed points until the next
    resample).
    """
    candidates = data.sample_collocation(tcfg.n_coll * CANDIDATE_POOL_MULT, rng)
    magnitude = _residual_magnitude(
        registry.LossContext(model, candidates, groups=groups), coll_equations
    )
    base = data.sample_collocation(tcfg.n_coll, rng).detach()
    return rad_resample(
        candidates.detach(), magnitude, base, tcfg.n_coll, tcfg.resample_fraction, rng
    ).detach()


def _rba_collocation_loss(
    ctx: registry.LossContext,
    weights: dict[str, float],
    schedule: dict[str, float],
    attention: dict[str, torch.Tensor],
    plan: _LossPlan,
) -> torch.Tensor:
    """The collocation loss under Residual-Based Attention for one context.

    For each collocation term the per-point attention is updated from that term's
    per-point squared residual (only while the term is active, i.e. not gated off by
    the warm-up schedule) and the term contributes
    ``w_k * sched_k * mean((1+lambda_k) * r_k^2)``. ``attention`` is mutated in place so
    it persists across steps; the weights are the fixed config values (RBA replaces the
    gradient-norm rebalancer, so nothing ratchets them). At zero attention every term
    reduces to ``w_k * sched_k * mean(r_k^2)`` -- the pre-RBA objective.
    """
    gamma, eta = plan.tcfg.rba_gamma, plan.tcfg.rba_eta
    total = 0.0
    for eq in plan.coll_equations:
        pointwise = eq.pointwise(ctx)
        sched = schedule.get(eq.weight_key, 1.0)
        # Use the current attention, THEN update it for the next step -- so a fresh run
        # (attention 0) applies the plain mean on its first step, exactly the pre-RBA
        # objective, and a warm-up-gated term (sched 0) never updates.
        total = total + weights[eq.weight_key] * sched * weighting.rba_weighted_mean(
            pointwise, attention[eq.weight_key]
        )
        if sched > 0:
            attention[eq.weight_key] = weighting.rba_update(
                attention[eq.weight_key], pointwise, gamma, eta
            )
    return total


def _causal_collocation(
    model,
    x_coll: torch.Tensor,
    groups: dict[str, float],
    c: torch.Tensor | None,
    weights: dict[str, float],
    schedule: dict[str, float],
    step: int,
    plan: _LossPlan,
) -> torch.Tensor:
    """One dataset's causal collocation loss, dispatched on ``causal_mode``: the soft
    per-time ``weight`` variant or the ``march`` time-curriculum."""
    tcfg = plan.tcfg
    if tcfg.causal_mode == "march":
        progress = (step - plan.first_step) / max(1, plan.last_step - plan.first_step)
        return _time_march_collocation_loss(
            model,
            x_coll,
            groups,
            c,
            plan.coll_equations,
            weights,
            schedule,
            tcfg.causal_time_chunks,
            progress,
            tcfg.causal_march_full_frac,
        )
    eps = _causal_eps(step, plan.first_step, plan.last_step, tcfg.causal_eps_schedule)
    return _causal_collocation_loss(
        model,
        x_coll,
        groups,
        c,
        plan.coll_equations,
        weights,
        schedule,
        tcfg.causal_time_chunks,
        eps,
    )


def _total_loss(
    losses: dict[str, torch.Tensor],
    coll_batches: list[CollocationBatch],
    weights: dict[str, float],
    schedule: dict[str, float],
    step: int,
    plan: _LossPlan,
) -> torch.Tensor:
    """The optimised total loss for one step, shared by ``train`` and ``_train_joint``.

    Off (default): the per-term weighted, scheduled sum of ``losses``. On: the
    collocation terms are replaced by their causal aggregation (soft time weighting or
    a time-marching curriculum, per ``causal_mode``), averaged over the datasets'
    collocation batches (a single batch for a single-dataset run), while the supervised
    (``data``) and boundary (``bc``) terms keep their uniform contribution. ``losses``
    -- the plain per-term means -- is never touched here, so logging and the
    gradient-norm rebalancer see the same quantities either way.
    """
    if not plan.tcfg.causal_weighting:
        return _weighted_sum(losses, weights, schedule)
    coll = torch.stack(
        [
            _causal_collocation(b.model, b.x, b.groups, b.c, weights, schedule, step, plan)
            for b in coll_batches
        ]
    ).mean()
    return coll + _weighted_sum(losses, weights, schedule, skip=plan.coll_keys)


def _stage_b_engages_at(step: int, tcfg, stage_b_keys: tuple[str, ...]) -> bool:
    """True only at the one step the warm-up hands off to Stage-B physics -- where
    the optimiser is restarted. Never true when the warm-up is disabled
    (``stage_b_warmup_steps=0``) or the model has no Stage-B physics to engage."""
    return (
        bool(stage_b_keys)
        and tcfg.stage_b_warmup_steps > 0
        and step == tcfg.stage_b_warmup_steps + 1
    )


def _restart_for_stage_b(
    model, lr: float, reset_weights: dict[str, float]
) -> tuple[torch.optim.Optimizer, dict[str, float]]:
    """A fresh optimiser and a clean copy of the config loss weights, for the moment
    Stage-B physics engages after the warm-up.

    Carrying Adam's Stage-A state and the warm-up-rebalanced weights across the
    switch destabilises the interface; a fresh-optimiser warm start from a Stage-A
    checkpoint does not, and this reproduces that clean transition in one run."""
    return torch.optim.Adam(model.parameters(), lr=lr), dict(reset_weights)


def _rebalance(
    weights: dict[str, float],
    norms: dict[str, float],
    terms: tuple[str, ...] = REBALANCED_TERMS,
) -> None:
    """Nudge weights so each term's gradient matches the data term's. In place."""
    reference = norms["data"] * weights["data"]
    for name in terms:
        target = reference / norms[name]
        # Half-step towards the target: full steps oscillate.
        weights[name] = float(np.clip(0.5 * weights[name] + 0.5 * target, 1e-2, 1e3))


def train(
    cfg,
    paths: RunPaths,
    steps: int | None = None,
    on_log: Callable[[dict], None] | None = None,
) -> tuple[BubblePINN, BubbleDataset, dict]:
    """Train (or continue training) and write the checkpoint. Returns the model.

    ``on_log``, when given, receives a copy of each history record as it is
    logged, so a caller can observe progress while the run is still going.
    """
    # Validate the held-out split up front, so a misconfigured `heldout_datasets`
    # fails loudly on every path -- including a single-dataset CLI run, which never
    # reaches `_train_joint` and its own guard.
    training_datasets(cfg)
    if len(resolved_datasets(cfg)) > 1:
        return _train_joint(cfg, paths, steps=steps, on_log=on_log)

    tcfg = cfg.training
    _validate_training_config(tcfg, cfg.model.fields)
    rba = tcfg.weighting == "rba"
    steps = int(steps if steps is not None else tcfg.steps)
    device = torch.device(tcfg.device)

    paths.ensure()
    torch.manual_seed(tcfg.seed)

    groups = compute_groups(cfg)
    u_inlet = groups["u_inlet_star"]
    data = BubbleDataset(cfg, paths, device=str(device))
    groups = _pulse_groups(groups, cfg, data.domain)
    model = BubblePINN(
        cfg, pin=_pin_anchor(cfg, data), geometry=_geometry_priors(cfg, data)
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    kin_plan = _kinematic_plan(cfg, data, device)

    equations = registry.enabled_equations(cfg.model.fields)
    rebalanced = registry.rebalanced_terms(equations)
    stage_b_keys = registry.stage_b_terms(equations)
    coll_equations = registry.collocation_equations(equations)
    coll_keys = frozenset(e.weight_key for e in coll_equations)
    state = _initial_state(cfg, equations)
    if paths.checkpoint.exists():
        ckpt = torch.load(paths.checkpoint, map_location=device, weights_only=False)
        _check_pin_compat(cfg, ckpt, paths.checkpoint)
        incompatible = model.load_state_dict(ckpt["model"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"checkpoint {paths.checkpoint} has parameters not in this model: "
                f"{incompatible.unexpected_keys}"
            )
        if incompatible.missing_keys:
            # Warm start: e.g. a Stage-A checkpoint feeding a Stage-B run. Keep the
            # loaded fields, initialise the new ones fresh, start the optimiser
            # over, and carry the step count and Stage-A weights forward.
            log.info(
                "warm start from %s: %d fresh parameters for new fields",
                paths.checkpoint,
                len(incompatible.missing_keys),
            )
            state["w"].update(ckpt["state"]["w"])
            state["done"] = ckpt["state"]["done"]
            state["hist"] = ckpt["state"]["hist"]
        else:
            opt.load_state_dict(ckpt["opt"])
            state = ckpt["state"]
            log.info("resuming from %s at step %d", paths.checkpoint, state["done"])

    # Offset the seed by completed steps so a resumed run does not replay the
    # same sample sequence it already saw.
    rng = np.random.default_rng(tcfg.seed + state["done"])
    weights = state["w"]
    reset_weights = _initial_state(cfg, equations)["w"]  # config defaults, for the handoff

    # RBA trains on a FIXED collocation pool so the per-point attention persists across
    # steps. The pool is drawn from a fixed seed (not offset by `done`) so a resumed run
    # regenerates the identical pool the saved attention is aligned to.
    x_pool = attention = None
    if rba:
        # Restore the exact pool the saved attention aligns to; a fresh run draws it from
        # a fixed seed. (With RAR the pool evolves, so it must be restored, not just
        # reseeded -- for a non-adaptive run the two are identical.)
        saved_pool = state.get("rba_pool")
        x_pool = (
            saved_pool.to(device)
            if saved_pool is not None
            else data.sample_collocation(tcfg.n_coll, np.random.default_rng(tcfg.seed)).detach()
        )
        attention = _init_attention(state, coll_keys, x_pool.shape[0], device)
        state.setdefault("rba_resamples", 0)

    first_step = state["done"] + 1
    last_step = state["done"] + steps
    plan = _LossPlan(coll_equations, coll_keys, first_step, last_step, tcfg)
    log.info("training steps %d-%d on %s", first_step, last_step, device)

    for step in range(first_step, last_step + 1):
        lr = tcfg.lr * (0.5 ** (step // tcfg.lr_halflife))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad()

        # Residual-adaptive resampling (RAR): periodically move the RBA pool toward the
        # high-residual regions and reset the per-point attention for the new points.
        if rba and tcfg.adaptive_collocation and step % tcfg.resample_every == 0:
            x_pool = _resample_pool(model, data, groups, coll_equations, tcfg, rng)
            attention = _init_attention({}, coll_keys, x_pool.shape[0], device)
            state["rba_resamples"] += 1
            log.info("step %5d | RAR resampled the collocation pool", step)

        x_data, alpha_target = data.sample_supervised(tcfg.n_data, rng)
        # RBA reuses the fixed pool (a fresh leaf each step so the PDE derivatives
        # differentiate through it, with no cross-step grad accumulation).
        x_coll = (
            x_pool.clone().requires_grad_(True)
            if rba
            else data.sample_collocation(tcfg.n_coll, rng)
        )
        inlet, walls = data.sample_boundary(tcfg.n_bc, rng)

        ctx = registry.LossContext(model, x_coll, inlet, walls, u_inlet, groups)
        losses = {"data": ((model.alpha(x_data) - alpha_target) ** 2).mean()}
        for eq in equations:
            losses[eq.weight_key] = eq.term(ctx)
        coll_batches = [CollocationBatch(model, x_coll, groups, None)]  # single dataset

        # Gradient-norm rebalancing is the legacy scheme only; RBA replaces it with the
        # bounded per-point attention below. Skip the handoff step: weights reset there.
        if (
            not rba
            and step % tcfg.rebalance_every == 0
            and not _stage_b_engages_at(step, tcfg, stage_b_keys)
        ):
            _rebalance(weights, _gradient_norms(model, losses, opt), rebalanced)
            log.info("step %5d | rebalanced weights: %s", step, _fmt(weights))

        if _stage_b_engages_at(step, tcfg, stage_b_keys):
            # Engage Stage-B physics on the interface the warm-up converged, from a
            # clean optimiser (see _restart_for_stage_b).
            opt, weights = _restart_for_stage_b(model, lr, reset_weights)
            log.info("step %5d | Stage-B physics engaged (%s)", step, ", ".join(stage_b_keys))
        schedule = _loss_schedule(step, tcfg, stage_b_keys)
        if rba:
            total = _rba_collocation_loss(
                ctx, weights, schedule, attention, plan
            ) + _weighted_sum(losses, weights, schedule, skip=coll_keys)
        else:
            total = _total_loss(losses, coll_batches, weights, schedule, step, plan)
        kin_record: dict[str, float] = {}
        if kin_plan is not None:
            # Outside the causal gate, RBA, and the rebalancer by construction:
            # added directly to the total with fixed config weights.
            kin_total, kin_record = kinematics.kinematic_losses(
                kinematics.KinematicContext(model, tcfg, groups),
                kin_plan,
                _kin_schedule(step, first_step, tcfg),
            )
            total = total + kin_total
        total.backward()
        opt.step()

        if step % tcfg.log_every == 0 or step == first_step:
            record = {name: float(loss.detach()) for name, loss in losses.items()}
            record.update(kin_record)
            record["step"] = step
            record["lr"] = lr
            state["hist"].append(record)
            if on_log is not None:
                on_log(dict(record))
            log.info(
                "step %5d | lr=%.2e | %s",
                step,
                lr,
                " ".join(f"{k}={v:.2e}" for k, v in record.items() if k not in ("step", "lr")),
            )

    state["done"] += steps
    state["w"] = weights
    if rba:
        state["attention"] = attention  # persist per-point attention for resume
        state["rba_pool"] = x_pool  # persist the pool the attention aligns to (RAR evolves it)
        # Fingerprint of the FINAL pool (RAR mutates it in-loop), so a resume can be
        # checked to have restored the identical pool the saved attention aligns to.
        state["rba_pool_sig"] = float(x_pool.sum().item())
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "state": state,
            **_pin_record(cfg),
        },
        paths.checkpoint,
    )
    log.info("checkpoint written to %s (%d steps total)", paths.checkpoint, state["done"])
    return model, data, state


@dataclass
class _JointDataset:
    """One dataset participating in a joint run: its tensors, its dimensionless
    groups (the physics coefficients), the conditioning row the model reads, and
    its inlet velocity."""

    name: str
    data: BubbleDataset
    groups: dict
    c: torch.Tensor  # (1, N_COND) conditioning row, broadcast per point batch
    u_inlet: float
    # The dataset's measured root anchor when the hard pin is on, else None.
    # Joint models bind it per call: model.bound(c, pin=pin).
    pin: tuple[float, float] | None = None
    # The dataset's fixed kinematic quadrature when the constraints are on.
    kin: kinematics.KinematicPlan | None = None


def _load_joint_datasets(
    cfg, paths: RunPaths, device, names: list[str] | None = None
) -> list[_JointDataset]:
    """Load the named datasets of a joint run, each with its conditioning row.

    ``names`` defaults to every dataset the run spans; the trainer passes only the
    training datasets (holding conditions out), and evaluation loads the held-out
    ones separately. Each dataset's regime comes from the groups recorded in its
    tensors, so no per-dataset Hydra recomposition is needed at train time.
    """
    if names is None:
        names = resolved_datasets(cfg)
    contexts: list[_JointDataset] = []
    for name in names:
        data = BubbleDataset(cfg, paths.for_dataset(name), device=str(device))
        groups = data.groups
        if groups is None:
            raise ValueError(
                f"dataset {name!r} has no groups in its tensors -- re-run preprocess "
                f"before joining it to a multi-dataset run"
            )
        c = torch.tensor([conditioning_vector(groups)], dtype=torch.float32, device=device)
        # The conditioning row is built from the physics groups before the pulse geometry
        # is mixed in, so `conditioning_vector` sees the same keys as a single-dataset run.
        pulse_groups = _pulse_groups(groups, cfg, data.domain)
        contexts.append(
            _JointDataset(
                name,
                data,
                pulse_groups,
                c,
                float(groups["u_inlet_star"]),
                pin=_pin_anchor(cfg, data),
                kin=_kinematic_plan(cfg, data, device),
            )
        )
    return contexts


def _joint_losses(
    model, contexts, equations, tcfg, rng
) -> tuple[dict[str, torch.Tensor], list[CollocationBatch]]:
    """Each loss term averaged over the datasets, every dataset's residuals
    evaluated with its own conditioning row so one model fits them all.

    Also returns each dataset's :class:`CollocationBatch` -- carrying that
    dataset's pin-bound view on a hard-pin run -- so the caller can apply causal
    temporal weighting to the same collocation points without resampling and
    without losing the per-dataset anchor.
    """
    aggregate: dict[str, torch.Tensor] = {}
    coll_batches: list[CollocationBatch] = []
    for cx in contexts:
        # Every call goes through the dataset's bound view: on a hard-pin run it
        # carries the root anchor; otherwise binding is a behavioral no-op (the
        # view forwards the same c the raw-model calls passed before).
        view = model.bound(cx.c, pin=cx.pin)
        x_data, target = cx.data.sample_supervised(tcfg.n_data, rng)
        x_coll = cx.data.sample_collocation(tcfg.n_coll, rng)
        inlet, walls = cx.data.sample_boundary(tcfg.n_bc, rng)

        ctx = registry.LossContext(view, x_coll, inlet, walls, cx.u_inlet, cx.groups, c=cx.c)
        c_data = cx.c.expand(x_data.shape[0], -1)
        per_term = {"data": ((view.alpha(x_data, c_data) - target) ** 2).mean()}
        for eq in equations:
            per_term[eq.weight_key] = eq.term(ctx)

        for name, loss in per_term.items():
            aggregate[name] = loss if name not in aggregate else aggregate[name] + loss
        coll_batches.append(CollocationBatch(view, x_coll, cx.groups, cx.c))

    n = len(contexts)
    return {name: loss / n for name, loss in aggregate.items()}, coll_batches


def _joint_kinematic_losses(
    model, contexts, tcfg, schedule: kinematics.KinematicSchedule
) -> tuple[torch.Tensor, dict[str, float]]:
    """The kinematic total averaged over the datasets, each evaluated through its
    pin-bound view on its own quadrature/references; the logged values are the
    across-dataset means of the raw term magnitudes."""
    totals = []
    merged: dict[str, float] = {}
    for cx in contexts:
        view = model.bound(cx.c, pin=cx.pin)
        kin_total, kin_record = kinematics.kinematic_losses(
            kinematics.KinematicContext(view, tcfg, cx.groups), cx.kin, schedule
        )
        totals.append(kin_total)
        for name, value in kin_record.items():
            merged[name] = merged.get(name, 0.0) + value / len(contexts)
    return torch.stack(totals).mean(), merged


def _train_joint(
    cfg,
    paths: RunPaths,
    steps: int | None = None,
    on_log: Callable[[dict], None] | None = None,
) -> tuple[BubblePINN, BubbleDataset, dict]:
    """Joint (transfer-learning) training over several datasets at once.

    One model, conditioned on each dataset's dimensionless groups, is fit to all
    the datasets together: every step sums each dataset's supervised + physics
    losses (each evaluated with that dataset's conditioning row), so the model
    learns the shared physics parameterised by the operating condition. Writes a
    single checkpoint. The unconditioned single-dataset path (:func:`train`) is
    untouched; this runs only when ``cfg.datasets`` names more than one series.
    """
    tcfg = cfg.training
    _validate_training_config(tcfg, cfg.model.fields)
    if tcfg.weighting == "rba":
        # RBA needs a per-dataset fixed pool + attention; wired for single-dataset runs
        # first (T2.3), joint runs in T2.5. Raise rather than silently use gradnorm.
        raise NotImplementedError(
            "training.weighting='rba' is not yet supported for joint (multi-dataset) "
            "runs; use weighting='gradnorm' for joint runs for now."
        )
    steps = int(steps if steps is not None else tcfg.steps)
    device = torch.device(tcfg.device)

    # The datasets actually supervised: held-out conditions (axis B) are loaded
    # only at evaluation, never here, so they contribute nothing to the loss. This
    # also validates the split (rejects an all-held-out run) before any I/O.
    spanned = resolved_datasets(cfg)
    train_names = training_datasets(cfg)
    heldout = [name for name in spanned if name not in set(train_names)]

    paths.ensure()
    torch.manual_seed(tcfg.seed)

    contexts = _load_joint_datasets(cfg, paths, device, train_names)
    names = [cx.name for cx in contexts]
    model = BubblePINN(cfg, n_cond=N_COND).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)

    equations = registry.enabled_equations(cfg.model.fields)
    rebalanced = registry.rebalanced_terms(equations)
    stage_b_keys = registry.stage_b_terms(equations)
    coll_equations = registry.collocation_equations(equations)
    coll_keys = frozenset(e.weight_key for e in coll_equations)
    state = _initial_state(cfg, equations)

    rng = np.random.default_rng(tcfg.seed + state["done"])
    weights = state["w"]
    reset_weights = _initial_state(cfg, equations)["w"]  # config defaults, for the handoff
    first_step = state["done"] + 1
    last_step = state["done"] + steps
    plan = _LossPlan(coll_equations, coll_keys, first_step, last_step, tcfg)
    log.info(
        "joint training steps %d-%d on %s over %s%s",
        first_step,
        last_step,
        device,
        names,
        f" (holding out {heldout})" if heldout else "",
    )

    for step in range(first_step, last_step + 1):
        lr = tcfg.lr * (0.5 ** (step // tcfg.lr_halflife))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad()

        losses, coll_batches = _joint_losses(model, contexts, equations, tcfg, rng)

        # Skip rebalancing on the handoff step: its weights are about to be reset.
        if step % tcfg.rebalance_every == 0 and not _stage_b_engages_at(
            step, tcfg, stage_b_keys
        ):
            _rebalance(weights, _gradient_norms(model, losses, opt), rebalanced)
            log.info("step %5d | rebalanced weights: %s", step, _fmt(weights))

        if _stage_b_engages_at(step, tcfg, stage_b_keys):
            # Engage Stage-B physics on the interface the warm-up converged, from a
            # clean optimiser (see _restart_for_stage_b).
            opt, weights = _restart_for_stage_b(model, lr, reset_weights)
            log.info("step %5d | Stage-B physics engaged (%s)", step, ", ".join(stage_b_keys))
        schedule = _loss_schedule(step, tcfg, stage_b_keys)
        total = _total_loss(losses, coll_batches, weights, schedule, step, plan)
        kin_record: dict[str, float] = {}
        if tcfg.kinematics:
            kin_total, kin_record = _joint_kinematic_losses(
                model, contexts, tcfg, _kin_schedule(step, first_step, tcfg)
            )
            total = total + kin_total
        total.backward()
        opt.step()

        if step % tcfg.log_every == 0 or step == first_step:
            record = {name: float(loss.detach()) for name, loss in losses.items()}
            record.update(kin_record)
            record["step"] = step
            record["lr"] = lr
            state["hist"].append(record)
            if on_log is not None:
                on_log(dict(record))
            log.info(
                "step %5d | lr=%.2e | %s",
                step,
                lr,
                " ".join(f"{k}={v:.2e}" for k, v in record.items() if k not in ("step", "lr")),
            )

    state["done"] += steps
    state["w"] = weights
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "state": state,
            "datasets": spanned,  # every series the run spans
            "training_datasets": names,  # the ones actually supervised
            "heldout_datasets": heldout,  # kept out for the transfer test (axis B)
            "n_cond": N_COND,  # so evaluation rebuilds the conditioned architecture
            **_pin_record(cfg),  # so evaluation rebuilds the pinned architecture
        },
        paths.checkpoint,
    )
    log.info(
        "joint checkpoint written to %s (%d steps, %d datasets)",
        paths.checkpoint,
        state["done"],
        len(names),
    )
    return model, contexts[0].data, state


def load_model(cfg, paths: RunPaths) -> tuple[BubblePINN, BubbleDataset, dict]:
    """Load a trained model and its dataset for evaluation or rendering."""
    if not paths.checkpoint.exists():
        raise FileNotFoundError(
            f"{paths.checkpoint} not found -- run the train stage first:\n"
            f"  naviernet stage=train run_name={cfg.run_name}"
        )
    device = torch.device(cfg.training.device)
    ckpt = torch.load(paths.checkpoint, map_location=device, weights_only=False)
    _check_pin_compat(cfg, ckpt, paths.checkpoint)
    data = BubbleDataset(cfg, paths, device=str(device))
    # `n_cond` (0 for a single-dataset checkpoint) rebuilds the right architecture,
    # so a conditioned joint checkpoint loads without a shape mismatch.
    model = BubblePINN(
        cfg,
        n_cond=int(ckpt.get("n_cond", 0)),
        pin=_pin_anchor(cfg, data),
        geometry=_geometry_priors(cfg, data),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, data, ckpt["state"]


def load_joint(cfg, paths: RunPaths) -> tuple[BubblePINN, list[_JointDataset], list[str]]:
    """The conditioned model, its per-dataset contexts, and the held-out split, for
    evaluating a joint run. Mirrors :func:`load_model` but returns every dataset the
    run spans (each with its conditioning row) plus the ``heldout_datasets`` the
    model was *trained* with.

    The dataset list and the split come from the checkpoint, not a freshly composed
    ``cfg``: a standalone ``stage=evaluate`` must score each dataset by how it was
    actually trained, regardless of the overrides the re-run happens to compose.
    Pre-split checkpoints fall back to ``cfg`` (``datasets``/``heldout_datasets``)."""
    if not paths.checkpoint.exists():
        raise FileNotFoundError(
            f"{paths.checkpoint} not found -- run the train stage first:\n"
            f"  naviernet stage=train run_name={cfg.run_name}"
        )
    device = torch.device(cfg.training.device)
    ckpt = torch.load(paths.checkpoint, map_location=device, weights_only=False)
    _check_pin_compat(cfg, ckpt, paths.checkpoint)
    names = list(ckpt.get("datasets") or resolved_datasets(cfg))
    heldout = list(ckpt.get("heldout_datasets", cfg.heldout_datasets))
    contexts = _load_joint_datasets(cfg, paths, device, names)
    model = BubblePINN(cfg, n_cond=int(ckpt.get("n_cond", N_COND))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, contexts, heldout


def _fmt(weights: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.3g}" for k, v in weights.items())
