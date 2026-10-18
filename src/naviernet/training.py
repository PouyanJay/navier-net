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

import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
import torch

from naviernet import warmup_cache
from naviernet.config.schema import resolved_datasets, training_datasets
from naviernet.data.adaptive import rad_resample
from naviernet.data.dataset import BubbleDataset
from naviernet.models.geometry import GeometryPriors
from naviernet.models.pinn import BoundPINN, BubblePINN
from naviernet.physics import (
    cap_supervision,
    front_velocity,
    growth_closure,
    heating,
    kinematics,
    mass_balance,
    nose_supervision,
    registry,
    root_supervision,
    weighting,
)
from naviernet.physics.groups import N_COND, compute_groups, conditioning_vector
from naviernet.physics.shape_solve import bind_solved_shape
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


def _geometry_priors(cfg, data: BubbleDataset) -> GeometryPriors | None:
    """The dataset's measured geometry anchors when the front geometry is on,
    else ``None`` -- the construction-site helper, like :func:`_pin_anchor`."""
    if not getattr(cfg.model, "front_geometry", False):
        return None
    x0, y0 = data.pin_anchor
    d = data.domain
    if bool(getattr(cfg.model, "mirror_symmetry", False)):
        # The root sits on the axis the fields are reflected about, not at the
        # measured anchor (3 pixels off it on Series-1): the construction is
        # symmetric by design, and so must be the point it hangs from.
        y0 = 0.5 * (data.y_lo + data.y_hi)
        log.info("mirror symmetry on: root height set to the channel centreline y*=%.4f", y0)
    log.info(
        "front geometry on: root (x*=%.4f, y*=%.4f), nose from x*=%.4f at rate %.3f, w0=%.4f",
        x0,
        y0,
        data.initial_front,
        data.nose_rate,
        data.initial_half_width,
    )
    return GeometryPriors(
        x_root=x0,
        y_root=y0,
        s0=data.initial_front,
        w0=data.initial_half_width,
        rate0=data.nose_rate,
        # The run's wall span (crop edges or the physical walls, per
        # model.walls_at): the capsule's width bound and centreline swing.
        y_min=data.y_lo,
        y_max=data.y_hi,
        t_min=d.t_min,
        t_max=d.t_max,
    )


def bind_cover_times(cfg, model, data: BubbleDataset) -> None:
    """Give a microlayer-wetted model the instant the images first cover each
    spot of the floor; a no-op under the uniform wetting."""
    if getattr(model, "footprint_wetting", "uniform") != "microlayer":
        return
    age = compute_groups(cfg)["footprint_age_star"]
    model.bind_cover_times(data.x, data.y, heating.cover_times(data, age))


def _growth_closure_plan(cfg, data: BubbleDataset, device):
    """The growth closure's quadrature when its weight is on, else None."""
    if float(getattr(cfg.training, "growth_closure_weight", 0.0)) <= 0.0:
        return None
    return growth_closure.build_plan(data, int(cfg.training.growth_closure_stride), device)


def _validate_growth_closure(tcfg, model_cfg) -> None:
    """The closure trains the footprint's wetting against the measured growth,
    so there must be a thermal source with a footprint to train."""
    weight = float(getattr(tcfg, "growth_closure_weight", 0.0))
    if weight < 0.0:
        raise ValueError(f"training.growth_closure_weight must be >= 0, got {weight}")
    if weight == 0.0:
        return
    if str(getattr(model_cfg, "source_mode", "free")) != "thermal" or not bool(
        getattr(model_cfg, "footprint_evaporation", False)
    ):
        raise ValueError(
            "training.growth_closure_weight scores the vapour the heat makes, so it requires "
            "model.source_mode=thermal and model.footprint_evaporation=true"
        )


def _wetting_record(model) -> dict[str, float]:
    """The fitted wetting of the footprint, for the log."""
    if getattr(model, "footprint_wetting", "uniform") == "microlayer":
        return {"film_um": float(torch.as_tensor(model.microlayer_um).detach())}
    if getattr(model, "wet_fraction_mode", "fixed") == "fit":
        return {"wet_fraction": float(model.wet_fraction.detach())}
    return {}


def _stage_b_keys(cfg) -> tuple[str, ...]:
    """The weight keys of the terms held at zero through the warm-up."""
    equations = registry.enabled_equations(cfg.model.fields, **registry.flags_from(cfg.model))
    return tuple(registry.stage_b_terms(equations))


def _sharp(cfg) -> bool:
    """Whether this run imposes the interface conditions on the explicit front."""
    return bool(getattr(cfg.model, "sharp_interface", False))


def _film(cfg) -> bool:
    """Whether this run carries the liquid film delta(u, t) on the front."""
    return bool(getattr(cfg.model, "liquid_film", False))


def _closure(cfg) -> str:
    """The sharp-mode momentum law this run solves: ``darcy`` or ``brinkman``."""
    return str(getattr(cfg.model, "momentum_closure", "darcy"))


def _two_phase(cfg) -> bool:
    """Whether this run represents the pressure per phase (p_v inside, p_l outside)."""
    return str(getattr(cfg.model, "pressure_representation", "single")) == "two_phase"


def _sharp_thermal(cfg) -> bool:
    """Whether this run solves the sharp thermal model (interface at saturation)."""
    return str(getattr(cfg.model, "thermal_model", "legacy")) == "sharp"


class _BoundaryPlan(NamedTuple):
    """The boundary choices the samplers and the ``bc`` term read every step."""

    wall_condition: str
    outlet: bool


def _boundary_plan(cfg) -> _BoundaryPlan:
    model_cfg = cfg.model
    return _BoundaryPlan(
        str(getattr(model_cfg, "wall_condition", "no_slip")),
        str(getattr(model_cfg, "outlet_condition", "none")) != "none",
    )


def _mass_balance_plan(cfg, data: BubbleDataset, groups: dict, device, rng=None):
    """The cross-section mass-balance quadrature when the term is on, else None.
    With ``rng`` (``training.quadrature_jitter``) its instants are a fresh
    stratified draw, so the balance is scored across the window rather than at
    the same eight instants every step."""
    if not bool(getattr(cfg.training, "mass_balance", False)):
        return None
    at_times = None
    if rng is not None:
        at_times = torch.tensor(
            _stratified_times(data.domain, int(cfg.training.mass_balance_times), rng)
        )
    return mass_balance.build_plan(
        data.domain, (data.y_lo, data.y_hi), groups, cfg.training, device, at_times=at_times
    )


def _stratified_times(domain, n: int, rng) -> np.ndarray:
    """One instant drawn inside each of ``n`` equal strata of the window, in
    order: the window stays covered and no instant is ever the same twice."""
    edges = np.linspace(domain.t_min, domain.t_max, n + 1)
    return rng.uniform(edges[:-1], edges[1:]).astype(np.float32)


def _front_times(cfg, data: BubbleDataset, device, rng=None) -> torch.Tensor | None:
    """The times the interface conditions are sampled at, or ``None`` when the run
    is not sharp-interface.

    Without ``rng``, a fixed uniform sweep of the window. With it
    (``training.quadrature_jitter``), a fresh stratified draw each step: the
    first full-chain seed satisfied the Young-Laplace condition at the sixteen
    fixed instants and violated it between them three to one.
    """
    if not _sharp(cfg):
        return None
    d = data.domain
    n = int(cfg.model.front_times)
    if rng is not None:
        return torch.tensor(_stratified_times(d, n, rng), device=device).reshape(-1, 1)
    return torch.linspace(d.t_min, d.t_max, n, device=device).reshape(-1, 1)


def _jitter_rng(tcfg, rng):
    """The rng the quadrature draws from under ``training.quadrature_jitter``,
    else None (fixed grids). The step rng, so a resumed run continues the draw."""
    return rng if bool(getattr(tcfg, "quadrature_jitter", False)) else None


def _architecture_record(cfg) -> dict:
    """The architecture facts persisted in every checkpoint (hard pin and front
    geometry add no detectable state-dict signature for their *configuration*),
    so a later invocation can be checked against how the run was trained."""
    hard_pin = bool(getattr(cfg.model, "hard_pin", False))
    cap_freedom = bool(getattr(cfg.model, "cap_freedom", False))
    nucleation_root = bool(getattr(cfg.model, "nucleation_root", False))
    nose_shape = bool(getattr(cfg.model, "nose_shape", False))
    return {
        "hard_pin": hard_pin,
        "pin_d_ref": float(cfg.model.pin_d_ref) if hard_pin else None,
        "cap_freedom": cap_freedom,
        # Like pin_d_ref: a VALUE the weights were trained against, so it is only
        # meaningful (and only checked) when its flag is on.
        "cap_delta": float(cfg.model.cap_delta) if cap_freedom else None,
        "nucleation_root": nucleation_root,
        "root_delta": float(cfg.model.root_delta) if nucleation_root else None,
        "nose_shape": nose_shape,
        "nose_delta": float(cfg.model.nose_delta) if nose_shape else None,
        "root_attachment": bool(getattr(cfg.model, "root_attachment", False)),
        "front_geometry": bool(getattr(cfg.model, "front_geometry", False)),
        "sharp_interface": bool(getattr(cfg.model, "sharp_interface", False)),
        "allow_pinch": bool(getattr(cfg.model, "allow_pinch", False)),
        "evolving_width": bool(getattr(cfg.model, "evolving_width", False)),
        "film_pressure": bool(getattr(cfg.model, "film_pressure", False)),
        "liquid_film": bool(getattr(cfg.model, "liquid_film", False)),
        "depletable_superheat": bool(getattr(cfg.model, "depletable_superheat", False)),
    }


def _require_matching_flag(cfg, ckpt: dict, key: str, path, why: str) -> None:
    """Refuse a checkpoint whose architectural boolean ``key`` disagrees with the
    composed config.

    Each of these flags changes what the weights MEAN without necessarily
    changing their shape, so ``load_state_dict`` succeeds either way and a
    mismatched invocation would quietly consume them in a differently-shaped
    solution space. Flags absent from the record predate it and pass unchecked.
    """
    saved = ckpt.get(key)
    if saved is None:
        return
    current = bool(getattr(cfg.model, key, False))
    if bool(saved) != current:
        raise ValueError(
            f"{path} was trained with model.{key}={bool(saved)} but this invocation "
            f"composes model.{key}={current}. {why} Pass the same override the run "
            f"was trained with."
        )


def _require_matching_value(
    cfg, ckpt: dict, flag_key: str, value_key: str, path, why: str
) -> None:
    """Refuse a checkpoint whose recorded VALUE ``value_key`` disagrees with the
    composed config while its owning flag is on.

    ``_require_matching_flag``'s sibling for the scale bounds (pin_d_ref,
    cap_delta, root_delta): each scales what its net's weights mean, so the
    same weights read as a different shape through a different value. Values
    absent from the record predate it and pass unchecked.
    """
    saved = ckpt.get(value_key)
    if saved is None or not bool(getattr(cfg.model, flag_key, False)):
        return
    current = float(getattr(cfg.model, value_key))
    if float(saved) != current:
        raise ValueError(
            f"{path} was trained with model.{value_key}={float(saved)} but this "
            f"invocation composes model.{value_key}={current}. {why} Pass the same "
            f"override the run was trained with."
        )


def _check_architecture_compat(cfg, ckpt: dict, path) -> None:
    """Refuse to consume a checkpoint whose recorded architecture disagrees with
    the composed config."""
    if ckpt.get("hard_pin") is None:
        return  # written before the record existed; nothing to compare
    _require_matching_flag(cfg, ckpt, "hard_pin", path, "The pin is architectural.")
    _require_matching_flag(cfg, ckpt, "front_geometry", path, "The geometry is architectural.")
    _require_matching_flag(
        cfg,
        ckpt,
        "sharp_interface",
        path,
        "The interface treatment changes the objective AND adds the p_v(t) parameters.",
    )
    _require_matching_flag(
        cfg,
        ckpt,
        "depletable_superheat",
        path,
        "The temperature's floor changes what its weights mean.",
    )
    _require_matching_flag(
        cfg,
        ckpt,
        "film_pressure",
        path,
        "The film offset adds a parameter and changes the jump condition.",
    )
    _require_matching_flag(
        cfg,
        ckpt,
        "liquid_film",
        path,
        "The liquid film adds the delta(u, t) net and its loss term.",
    )
    _require_matching_flag(
        cfg,
        ckpt,
        "evolving_width",
        path,
        "The width reparameterisation replaces one net with two.",
    )
    _require_matching_flag(
        cfg,
        ckpt,
        "allow_pinch",
        path,
        "The signed radius loads into the same tensors but means a different shape.",
    )
    _require_matching_flag(
        cfg,
        ckpt,
        "cap_freedom",
        path,
        "Free caps add the cap net and change what the cap radius means.",
    )
    _require_matching_flag(
        cfg,
        ckpt,
        "nucleation_root",
        path,
        "The Hugelschaffer root adds the w(t) net and reshapes the root cap.",
    )
    _require_matching_flag(
        cfg,
        ckpt,
        "root_attachment",
        path,
        "The attachment adds theta_app and the patch unknowns and changes the jump.",
    )
    _require_matching_flag(
        cfg,
        ckpt,
        "nose_shape",
        path,
        "The Hugelschaffer nose adds the w_nose(t) net and reshapes the nose cap.",
    )

    _require_matching_value(
        cfg,
        ckpt,
        "nucleation_root",
        "root_delta",
        path,
        "The bound scales the bluntness net's output, so the same weights describe "
        "a different root.",
    )
    _require_matching_value(
        cfg,
        ckpt,
        "nose_shape",
        "nose_delta",
        path,
        "The bound scales the nose bluntness net's output, so the same weights "
        "describe a different head.",
    )

    _require_matching_value(
        cfg,
        ckpt,
        "cap_freedom",
        "cap_delta",
        path,
        "The bound scales the cap net's output, so the same weights describe a different cap.",
    )
    _require_matching_value(
        cfg,
        ckpt,
        "hard_pin",
        "pin_d_ref",
        path,
        "The gate scale sets where the pin releases the field.",
    )


def _validate_training_config(cfg) -> None:
    """Every up-front training-config validation, shared by both loops.

    Takes the whole config rather than the four pieces it needs: every caller
    had one to hand, and the two that were optional were never once omitted --
    an optionality that only made it possible to skip a check by accident.
    """
    tcfg = cfg.training
    _validate_causal(tcfg)
    _validate_weighting(tcfg)
    _validate_kinematics(tcfg, cfg.model.fields)
    _validate_front_velocity(tcfg, cfg.model)
    _validate_root_supervision(tcfg, cfg.model)
    _validate_nose_supervision(tcfg, cfg.model)
    _validate_sharpening(tcfg, cfg.model.alpha_eps)
    _validate_boundaries(cfg.model)
    _validate_thermal_training(tcfg, cfg.model)
    _validate_rebalance_cap(tcfg)
    _validate_outline_guard(tcfg)
    _validate_freeze_geometry(tcfg, cfg.model)
    _validate_temperature_stage(tcfg, cfg.model)
    _validate_growth_closure(tcfg, cfg.model)


def _validate_rebalance_cap(tcfg) -> None:
    """A negative cap is a typo, not a choice."""
    if float(getattr(tcfg, "rebalance_max_ratio", 0.0)) < 0.0:
        raise ValueError(
            f"training.rebalance_max_ratio={tcfg.rebalance_max_ratio} must be >= 0 "
            "(0 keeps the fixed ceiling alone)."
        )


def _validate_freeze_geometry(tcfg, model_cfg) -> None:
    """The geometry is frozen at the warm-up's fit, so there must be a warm-up
    to fit it, and an explicit geometry to freeze."""
    weighting = str(getattr(tcfg, "frozen_weighting", "fixed"))
    if weighting not in ("fixed", "equalised"):
        raise ValueError(
            f"training.frozen_weighting must be 'fixed' or 'equalised', got {weighting!r}"
        )
    if not bool(getattr(tcfg, "freeze_geometry", False)):
        return
    if int(tcfg.stage_b_warmup_steps) <= 0:
        raise ValueError(
            "training.freeze_geometry freezes the bubble at the fit its warm-up reached, so it "
            "needs training.stage_b_warmup_steps > 0; set a warm-up, or turn it off."
        )
    if not bool(getattr(model_cfg, "front_geometry", False)):
        raise ValueError(
            "training.freeze_geometry freezes the explicit front, so it requires "
            "model.front_geometry=true; a free level set has no geometry to freeze apart "
            "from the fields."
        )


def _validate_temperature_stage(tcfg, model_cfg) -> None:
    """The temperature-only stage trains the temperature on a locked outline and
    flow, so the outline must already be frozen, the flow must have had its own
    stage after the warm-up, and there must be a temperature to train."""
    start = int(getattr(tcfg, "temperature_stage_from", 0))
    if start == 0:
        return
    if start < 0:
        raise ValueError(f"training.temperature_stage_from must be >= 0, got {start}")
    if not bool(getattr(tcfg, "freeze_geometry", False)):
        raise ValueError(
            "training.temperature_stage_from locks the flow on a frozen outline, so it "
            "requires training.freeze_geometry=true"
        )
    if start <= int(tcfg.stage_b_warmup_steps):
        raise ValueError(
            f"training.temperature_stage_from ({start}) must come after the warm-up "
            f"(training.stage_b_warmup_steps={tcfg.stage_b_warmup_steps}): the flow is "
            "solved between the two"
        )
    if "T" not in list(model_cfg.fields):
        raise ValueError(
            "training.temperature_stage_from trains the temperature, so it requires the "
            f"'T' field in model.fields; this model has {list(model_cfg.fields)}"
        )
    if tcfg.weighting == "rba":
        raise ValueError(
            "training.temperature_stage_from runs on the gradient-norm weighting; the RBA "
            "attention pool is not rebuilt for the temperature terms"
        )


def _validate_outline_guard(tcfg) -> None:
    """The guard's floor is the warm-up's fit, so it needs a warm-up to record one;
    a drop must be a non-negative tolerance."""
    every = int(getattr(tcfg, "outline_guard_every", 0))
    if every < 0:
        raise ValueError(f"training.outline_guard_every={every} must be >= 0 (0 = off)")
    if float(getattr(tcfg, "outline_guard_drop", 0.0)) < 0.0:
        raise ValueError(
            f"training.outline_guard_drop={tcfg.outline_guard_drop} must be >= 0 (an IoU tolerance)"
        )
    if every > 0 and int(tcfg.stage_b_warmup_steps) <= 0:
        raise ValueError(
            "training.outline_guard_every holds the run to the IoU its warm-up reached, so it "
            "needs training.stage_b_warmup_steps > 0 to record that fit; set a warm-up, or "
            "turn the guard off."
        )


def _validate_thermal_training(tcfg, model_cfg) -> None:
    """A thermal source rebuilds the front once per distinct collocation instant,
    so the batch's times must be binned; an unbinned batch has one instant per
    point and would cost thousands of fronts a step."""
    thermal_source = str(getattr(model_cfg, "source_mode", "free")) == "thermal"
    if thermal_source and int(tcfg.collocation_time_bins) <= 0:
        raise ValueError(
            "model.source_mode=thermal builds the front at every distinct collocation "
            "instant, so it needs training.collocation_time_bins > 0 (the trusted recipe "
            "uses 8)."
        )


_BOUNDARY_CHOICES = {
    "walls_at": ("crop", "physical"),
    "wall_condition": ("no_slip", "no_penetration"),
    "momentum_closure": ("darcy", "brinkman"),
    "outlet_condition": ("none", "zero_gradient"),
}


def _validate_boundaries(model_cfg) -> None:
    """The domain and boundary choices: an unknown value fails at launch, and the
    over-determined Darcy + no-slip pair (today's default) is allowed for existing
    configs but warned about, since it has no exact solution."""
    chosen = {
        name: str(getattr(model_cfg, name, allowed[0]))
        for name, allowed in _BOUNDARY_CHOICES.items()
    }
    for name, value in chosen.items():
        if value not in _BOUNDARY_CHOICES[name]:
            raise ValueError(
                f"model.{name} must be one of {_BOUNDARY_CHOICES[name]}, got {value!r}"
            )
    darcy_active = (
        chosen["momentum_closure"] == "darcy"
        and "p" in model_cfg.fields
        and bool(getattr(model_cfg, "sharp_interface", False))
    )
    if (
        bool(getattr(model_cfg, "momentum_inertia", False))
        and chosen["momentum_closure"] != "brinkman"
    ):
        raise ValueError(
            "model.momentum_inertia adds the depth-averaged inertia to the Brinkman law, so it "
            f"requires model.momentum_closure=brinkman; this model uses {chosen['momentum_closure']!r}"
        )
    if darcy_active and chosen["wall_condition"] == "no_slip":
        log.warning(
            "model.momentum_closure=darcy with model.wall_condition=no_slip is "
            "over-determined: a first-order law admits only no-penetration at a wall. "
            "Use momentum_closure=brinkman, or wall_condition=no_penetration."
        )


def _validate_root_supervision(tcfg, model_cfg) -> None:
    """Reject a root-supervision config that cannot drive anything, loudly and
    before any measurement is taken."""
    if not getattr(tcfg, "root_supervision", False):
        return
    if not bool(getattr(model_cfg, "nucleation_root", False)):
        raise ValueError(
            "training.root_supervision drives the Hugelschaffer bluntness w(t), "
            "which only exists under model.nucleation_root=true. Enable it, or "
            "turn training.root_supervision off."
        )
    for key in ("root_sdf_weight", "root_bluntness_weight"):
        if getattr(tcfg, key) < 0:
            raise ValueError(f"training.{key}={getattr(tcfg, key)} must be >= 0")
    if tcfg.root_sdf_weight == 0 and tcfg.root_bluntness_weight == 0:
        raise ValueError(
            "training.root_supervision=true with both training.root_sdf_weight and "
            "training.root_bluntness_weight at 0 supervises nothing -- an undriven "
            "knob wearing a supervision flag. Give one of them a weight, or turn "
            "training.root_supervision off."
        )
    if tcfg.root_cap_samples < 2:
        raise ValueError(
            f"training.root_cap_samples={tcfg.root_cap_samples} must be >= 2 "
            f"(a cap is an arc, not a point)."
        )


def _validate_nose_supervision(tcfg, model_cfg) -> None:
    """Reject a nose-supervision config that cannot drive anything, loudly and
    before any measurement is taken -- ``_validate_root_supervision``'s mirror."""
    if not getattr(tcfg, "nose_supervision", False):
        return
    if not bool(getattr(model_cfg, "nose_shape", False)):
        raise ValueError(
            "training.nose_supervision drives the Hugelschaffer nose bluntness "
            "w_nose(t), which only exists under model.nose_shape=true. Enable it, "
            "or turn training.nose_supervision off."
        )
    for key in ("nose_sdf_weight", "nose_bluntness_weight"):
        if getattr(tcfg, key) < 0:
            raise ValueError(f"training.{key}={getattr(tcfg, key)} must be >= 0")
    if tcfg.nose_sdf_weight == 0 and tcfg.nose_bluntness_weight == 0:
        raise ValueError(
            "training.nose_supervision=true with both training.nose_sdf_weight and "
            "training.nose_bluntness_weight at 0 supervises nothing -- an undriven "
            "knob wearing a supervision flag. Give one of them a weight, or turn "
            "training.nose_supervision off."
        )
    if tcfg.nose_cap_samples < 2:
        raise ValueError(
            f"training.nose_cap_samples={tcfg.nose_cap_samples} must be >= 2 "
            f"(a cap is an arc, not a point)."
        )


def _validate_front_velocity(tcfg, model_cfg) -> None:
    """Reject a front-velocity config that cannot supervise anything, loudly and
    before any measurement is taken."""
    if not tcfg.front_velocity:
        return
    if not bool(getattr(model_cfg, "front_geometry", False)):
        raise ValueError(
            "training.front_velocity supervises the explicit front's own normal "
            "speed, which only the front geometry has -- it requires "
            "model.front_geometry=true. Enable the front geometry, or turn "
            "training.front_velocity off."
        )
    if tcfg.fv_samples_per_bin < 1:
        raise ValueError(
            f"training.fv_samples_per_bin={tcfg.fv_samples_per_bin} must be >= 1: a bin "
            f"averaging no samples is not a measurement."
        )
    if tcfg.fv_smooth_px < 0:
        raise ValueError(
            f"training.fv_smooth_px={tcfg.fv_smooth_px} must be >= 0 (a blur radius)"
        )
    for key in ("fv_weight", "fv_apex_weight"):
        if getattr(tcfg, key) < 0:
            raise ValueError(f"training.{key}={getattr(tcfg, key)} must be >= 0")
    if tcfg.fv_weight == 0 and tcfg.fv_apex_weight == 0:
        raise ValueError(
            "training.front_velocity=true with both training.fv_weight and "
            "training.fv_apex_weight at 0 measures the front and then ignores it. "
            "Give one of them a weight, or turn training.front_velocity off."
        )


def _validate_sharpening(tcfg, alpha_eps: float) -> None:
    """Reject an interface-sharpening schedule that cannot sharpen."""
    if tcfg.alpha_eps_anneal_steps <= 0:
        return
    final = float(tcfg.alpha_eps_final)
    if final <= 0.0:
        raise ValueError(
            f"training.alpha_eps_final={final} must be > 0: alpha_eps divides phi, "
            f"so a zero interface thickness is a division by zero, not a sharp "
            f"interface."
        )
    if final >= float(alpha_eps):
        raise ValueError(
            f"training.alpha_eps_final={final} must be below model.alpha_eps="
            f"{alpha_eps}: the schedule sharpens the interface, it does not blur it."
        )


def annealed_alpha_eps(step: int, cfg) -> float:
    """The interface half-thickness at ``step``.

    Geometric, not linear: ``alpha_eps`` is a scale, so equal FRACTIONAL steps
    are what keep the sharpening gradual through the end of the schedule, where
    each further halving costs the most. Held at the final value afterwards, and
    a function of the absolute step so a resumed run picks the schedule up
    exactly where it left off.
    """
    tcfg = cfg.training
    start = float(cfg.model.alpha_eps)
    span = int(tcfg.alpha_eps_anneal_steps)
    if span <= 0:
        return start
    progress = min(max(step / span, 0.0), 1.0)
    return start * (float(tcfg.alpha_eps_final) / start) ** progress


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


def _front_velocity_plan(
    cfg, data: BubbleDataset, device
) -> front_velocity.FrontVelocityPlan | None:
    """The run's measured front kinematics when the term is on, else ``None``.
    Measured once (deterministic, no RNG) so resume supervises against exactly
    the same numbers."""
    tcfg = cfg.training
    if not tcfg.front_velocity:
        return None
    plan = front_velocity.build_plan(data, cfg, device)
    log.info(
        "front velocity on: %d frame pairs %s, v_ref=%.4f, %d body / %d cap bins "
        "at %d samples each, blur %.1f px",
        len(plan.pairs),
        [(data.frame_numbers[a], data.frame_numbers[b]) for a, b in plan.pairs],
        plan.v_ref,
        plan.bins_body,
        plan.bins_cap,
        int(tcfg.fv_samples_per_bin),
        float(tcfg.fv_smooth_px),
    )
    skipped = sorted(set(range(data.n_event - 1)) - {row for row, _ in plan.pairs})
    if skipped:
        log.info(
            "front velocity: no target from frames %s (held out of supervision, or "
            "across an excluded frame)",
            [data.frame_numbers[row] for row in skipped],
        )
    return plan


def _root_plan(cfg, data: BubbleDataset, device) -> root_supervision.RootPlan | None:
    """The run's measured root targets when the supervision is on, else ``None``.
    Measured once (deterministic, no RNG), the front-velocity pattern."""
    if not getattr(cfg.training, "root_supervision", False):
        return None
    plan = root_supervision.build_plan(data, cfg, device)
    log.info(
        "root supervision on: %d frames %s, bluntness targets %.3f -> %.3f, %d cap samples",
        len(plan.rows),
        [data.frame_numbers[r] for r in plan.rows],
        float(plan.bluntness[0]),
        float(plan.bluntness[-1]),
        int(cfg.training.root_cap_samples),
    )
    return plan


def _nose_plan(cfg, data: BubbleDataset, device) -> cap_supervision.WindowPlan | None:
    """The run's measured nose targets when the supervision is on, else ``None``.
    Measured once (deterministic, no RNG), the root/front-velocity pattern."""
    if not getattr(cfg.training, "nose_supervision", False):
        return None
    plan = nose_supervision.build_plan(data, cfg, device)
    log.info(
        "nose supervision on: %d frames %s, bluntness targets %.3f -> %.3f "
        "(above 1 = the overhang), %d cap samples",
        len(plan.rows),
        [data.frame_numbers[r] for r in plan.rows],
        float(plan.bluntness[0]),
        float(plan.bluntness[-1]),
        int(cfg.training.nose_cap_samples),
    )
    return plan


def _initial_state(cfg, equations) -> dict:
    weights = cfg.training.weights
    w = {"data": float(weights.data)}
    for eq in equations:
        w[eq.weight_key] = float(getattr(weights, eq.weight_key))
    # The warm-up length travels with the run: a later `stage=evaluate` composes
    # its own config, which need not carry the override the run was launched
    # with, and reading the wrong one silently averages a term's convergence
    # across the window where it was held at zero weight.
    return {
        "done": 0,
        "hist": [],
        "w": w,
        "stage_b_warmup_steps": int(cfg.training.stage_b_warmup_steps),
    }


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


@dataclass
class CausalSchedule:
    """The causal ``eps`` for each step.

    Without a gate this is the fixed per-step annealing of :func:`_causal_eps`.
    With one, ``eps`` holds its current schedule value until every time chunk's
    causal weight clears the gate, then advances (Wang et al.'s stopping rule).
    The weight observed each step is kept so the run's record can show how much
    of the time window the residual terms actually trained on.
    """

    values: list[float]
    first_step: int
    last_step: int
    gate: float | None
    index: int = 0
    min_weight: float = 1.0

    def eps(self, step: int) -> float:
        if self.gate is None:
            return _causal_eps(step, self.first_step, self.last_step, self.values)
        if not self.values:
            return 0.0
        return float(self.values[self.index])

    def observe(self, weights: torch.Tensor, armed: bool = True) -> None:
        """Record this step's causal weights and, while ``armed``, advance ``eps``
        once all clear the gate.

        Unarmed while a collocation term is still held at zero (the Stage-B
        warm-up, the evaporation curriculum): the weights then describe a loss
        the physics is absent from, and the Stage-A terms alone pass any gate.
        The first WP1 benchmark advanced eps to 1.0 inside the warm-up, and every
        late chunk's weight collapsed to ~0.01 the moment Stage B engaged.
        """
        self.min_weight = float(weights.min())
        if not armed or self.gate is None or self.index >= len(self.values) - 1:
            return
        if self.min_weight > self.gate:
            self.index += 1
            log.info(
                "causal eps advanced to %.3g (every chunk weight > %.2f)",
                self.eps(0),
                self.gate,
            )


def _causal_schedule(
    tcfg, first_step: int, last_step: int, state: dict
) -> CausalSchedule | None:
    """The run's causal schedule, resumed at the gate index the checkpoint saved;
    ``None`` when the soft causal weighting is off (uniform, or time-marching)."""
    if not tcfg.causal_weighting or tcfg.causal_mode != "weight":
        return None
    return CausalSchedule(
        values=list(tcfg.causal_eps_schedule),
        first_step=first_step,
        last_step=last_step,
        gate=tcfg.causal_gate_min_weight,
        index=int(state.get("causal_index", 0)),
    )


def _causal_record(plan: _LossPlan, step: int) -> dict[str, float]:
    """The causal state worth a line in the history: nothing when the weighting is off."""
    if plan.causal is None:
        return {}
    return {"causal_eps": plan.causal.eps(step), "causal_min_weight": plan.causal.min_weight}


def _chunk_diagnostics(model, x_coll, groups, coll_equations, n_chunks: int) -> dict:
    """Each collocation term's plain mean in each time chunk, keyed by weight key,
    with the chunks' mid-times under ``"t"``: the per-slice table that makes a
    residual left unenforced at late times visible in the run's own record.
    Evaluated on the step's own batch in one pass, so it costs one extra
    collocation evaluation at the logging cadence only."""
    ordered, bounds = _time_ordered(x_coll, n_chunks)
    ctx = registry.LossContext(model, ordered, groups=groups)
    times = ordered.detach()[:, 2]
    report: dict[str, list[float]] = {"t": [float(times[lo:hi].mean()) for lo, hi in bounds]}
    for eq in coll_equations:
        per_point = eq.pointwise(ctx).detach()
        report[eq.weight_key] = [float(per_point[lo:hi].mean()) for lo, hi in bounds]
    return report


def _time_ordered(
    x_coll: torch.Tensor, n_chunks: int
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """``x_coll`` sorted by time (column 2) as one fresh leaf, and the ``[lo, hi)``
    bounds of its ``n_chunks`` contiguous, near-equal, time-ordered chunks --
    the same chunks :func:`_time_chunks` returns as separate batches."""
    ordered = x_coll.detach()[torch.argsort(x_coll.detach()[:, 2])].clone().requires_grad_(True)
    return ordered, _chunk_bounds(ordered.shape[0], n_chunks)


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
    ctx = registry.LossContext(model, x_coll, groups=groups, c=c)
    chunk_losses = _chunk_losses(ctx, coll_equations, weights, schedule, n_chunks)
    return (_causal_weights(chunk_losses, eps) * chunk_losses).mean()


def _chunk_losses(
    ctx: registry.LossContext,
    coll_equations,
    weights: dict[str, float],
    schedule: dict[str, float],
    n_chunks: int,
) -> torch.Tensor:
    """``L_r(t_i)``: the per-term weighted, scheduled collocation loss in each
    time-ordered chunk, shape ``(n_chunks,)``, with gradient.

    No evaluation of its own: every term's per-point residual is read off the
    step's context (whose residual bundles are already computed and cached for
    the plain means) and reduced chunk by chunk in time order. The networks and
    the residuals are pointwise, so each chunk's mean is exactly what evaluating
    that chunk alone gives (held by a test). The 2026-10 profile put over half
    of a step in the ``n_chunks`` separate passes this replaces, each over a
    batch too small to fill a core, with the thermal source's front rebuilt in
    every one.
    """
    order = torch.argsort(ctx.x_coll.detach()[:, 2])
    per_point = sum(
        weights[eq.weight_key] * schedule.get(eq.weight_key, 1.0) * eq.pointwise(ctx)
        for eq in coll_equations
    )[order]
    bounds = _chunk_bounds(len(order), n_chunks)
    return torch.stack([per_point[lo:hi].mean() for lo, hi in bounds])


def _chunk_bounds(n_points: int, n_chunks: int) -> list[tuple[int, int]]:
    """The ``[lo, hi)`` bounds of ``n_chunks`` contiguous, near-equal chunks of
    ``n_points`` ordered points: the chunks :func:`_time_chunks` returns."""
    sizes = [c.shape[0] for c in torch.chunk(torch.empty(n_points), max(1, n_chunks))]
    edges = [0, *itertools.accumulate(sizes)]
    return [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:], strict=True) if hi > lo]


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
    otherwise), the points, the dataset's groups/conditioning row, and the step's
    loss context over those points, whose cached residuals the pass reuses."""

    model: BubblePINN | BoundPINN
    x: torch.Tensor
    groups: dict
    c: torch.Tensor | None
    ctx: registry.LossContext | None = None

    def context(self) -> registry.LossContext:
        """The step's context, or a fresh one over the batch when none was kept."""
        if self.ctx is None:
            self.ctx = registry.LossContext(self.model, self.x, groups=self.groups, c=self.c)
        return self.ctx


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
    # The soft causal weighting's eps policy (gated or fixed); None when it is off.
    # Mutable on purpose: the gate index is run state, persisted with the checkpoint.
    causal: CausalSchedule | None = None


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
    candidates = data.sample_collocation(
        tcfg.n_coll * CANDIDATE_POOL_MULT, rng, tcfg.collocation_time_bins
    )
    magnitude = _residual_magnitude(
        registry.LossContext(model, candidates, groups=groups), coll_equations
    )
    base = data.sample_collocation(tcfg.n_coll, rng, tcfg.collocation_time_bins).detach()
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
    batch: CollocationBatch,
    weights: dict[str, float],
    schedule: dict[str, float],
    step: int,
    plan: _LossPlan,
) -> torch.Tensor:
    """One dataset's causal collocation loss, dispatched on ``causal_mode``: the soft
    per-time ``weight`` variant (read off the batch's own context, so the step
    evaluates the collocation residuals once) or the ``march`` time-curriculum."""
    tcfg = plan.tcfg
    if tcfg.causal_mode == "march":
        progress = (step - plan.first_step) / max(1, plan.last_step - plan.first_step)
        return _time_march_collocation_loss(
            batch.model,
            batch.x,
            batch.groups,
            batch.c,
            plan.coll_equations,
            weights,
            schedule,
            tcfg.causal_time_chunks,
            progress,
            tcfg.causal_march_full_frac,
        )
    chunk_losses = _chunk_losses(
        batch.context(), plan.coll_equations, weights, schedule, tcfg.causal_time_chunks
    )
    if plan.causal is None:
        eps = _causal_eps(step, plan.first_step, plan.last_step, tcfg.causal_eps_schedule)
        return (_causal_weights(chunk_losses, eps) * chunk_losses).mean()
    causal_weights = _causal_weights(chunk_losses, plan.causal.eps(step))
    plan.causal.observe(causal_weights, armed=_gate_armed(schedule, plan.coll_keys))
    return (causal_weights * chunk_losses).mean()


def _gate_armed(schedule: dict[str, float], coll_keys: frozenset[str]) -> bool:
    """Whether every collocation term is in the loss the causal gate judges: a
    term at multiplier zero (warm-up gated, or at the foot of its curriculum)
    is absent from the chunk losses, so their weights say nothing about it."""
    return all(schedule.get(key, 1.0) > 0.0 for key in coll_keys)


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
    gradient-norm rebalancer see the same quantities either way; the soft weighting
    reads the same residuals through each batch's context, so it adds no pass.
    """
    if not plan.tcfg.causal_weighting:
        return _weighted_sum(losses, weights, schedule)
    coll = torch.stack(
        [_causal_collocation(b, weights, schedule, step, plan) for b in coll_batches]
    ).mean()
    return coll + _weighted_sum(losses, weights, schedule, skip=plan.coll_keys)


def _freeze_nose_through_warmup(model, tcfg, stage_b_keys, first_step: int) -> None:
    """Hold the nose DOF still through the Stage-B warm-up (the nose-release
    plan §1): the warm-up transient ALONE drives w_nose to the tanh rail --
    measured at -0.59 by step 1500 with no supervision and no jump physics --
    and saturation then traps it (~250x gradient attenuation) exactly where
    every trained-state preference (data, measured shape, the jump's own
    demanded profile) points the other way. Released at Stage-B engagement,
    where the geometry already sits near the mask. Frozen means frozen AT ITS
    INIT (the net's small random output), not clamped to zero -- the losses
    keep reading it, so their logged magnitudes stay comparable across the
    boundary.

    State-based, so resume reproduces it: a run resumed past the boundary
    starts released, one resumed into the warm-up starts frozen. A run with
    no warm-up (or no Stage-B physics) is never frozen -- today's behavior,
    which is what the fast Stage-A mechanism tests train under.
    """
    geometry = model.nets["phi"]
    if not getattr(geometry, "nose_shape", False):
        return
    frozen = bool(stage_b_keys) and 0 < first_step <= tcfg.stage_b_warmup_steps
    for p in geometry.nose_net.parameters():
        p.requires_grad_(not frozen)
    if frozen:
        log.info(
            "nose DOF frozen through the warm-up (releases at step %d)",
            tcfg.stage_b_warmup_steps + 1,
        )


def _release_nose_dof(model) -> None:
    """Let the nose DOF move -- the warm-up freeze's other half, called at the
    same step Stage-B engages (and BEFORE the optimiser restart, so the fresh
    Adam sees the trainable parameters). A no-op for every run that was never
    frozen."""
    geometry = model.nets["phi"]
    if not getattr(geometry, "nose_shape", False):
        return
    if all(p.requires_grad for p in geometry.nose_net.parameters()):
        return
    for p in geometry.nose_net.parameters():
        p.requires_grad_(True)
    log.info("nose DOF released: w_nose(t) may move")


def _clip_gradients(model, tcfg) -> torch.Tensor:
    """Bound the step a gradient spike can take (``training.grad_clip`` > 0) and
    return the total gradient norm before any clipping, for the run's record.
    A tensor, not a float: reading it every step would sync a GPU run every
    step, so the log branch converts it on its own cadence."""
    limit = float(getattr(tcfg, "grad_clip", 0.0))
    bound = limit if limit > 0.0 else float("inf")
    return torch.nn.utils.clip_grad_norm_(model.parameters(), bound)


class OutlineDrift(RuntimeError):
    """The outline guard stopped the run: its fit of the masks fell further below
    the warm-up's than ``training.outline_guard_drop`` allows."""


def _outline_fit(cfg, model, data) -> dict[str, float]:
    """The per-frame IoU against the masks, keyed by camera frame (as strings, so
    the record survives a round trip through JSON)."""
    from naviernet.evaluation import iou_report

    report = iou_report(cfg, model, data)
    return {str(frame): float(iou) for frame, iou in report["iou_per_frame"].items()}


def _record_warmup_fit(cfg, model, data, state: dict) -> None:
    """At the step the physics engages: the warm-up's fit of every frame, the
    floor the outline guard and the audit's outline verdict hold the run to."""
    state["warmup_iou"] = _outline_fit(cfg, model, data)
    log.info(
        "warm-up fit recorded: IoU mean %.3f, min %.3f",
        float(np.mean(list(state["warmup_iou"].values()))),
        min(state["warmup_iou"].values()),
    )


def _require_guard_arms(tcfg, stage_b_keys, state: dict, first_step: int) -> None:
    """A guard that could never fire is a switch that does nothing: it needs
    Stage-B physics to engage after the warm-up (that is the step the floor is
    recorded at), and a run resumed past that step needs the record."""
    if int(getattr(tcfg, "outline_guard_every", 0)) <= 0:
        return
    if not stage_b_keys:
        raise ValueError(
            "training.outline_guard_every holds the run to the fit its warm-up reached "
            "before the Stage-B physics engaged, and this model has no Stage-B physics to "
            "engage. Turn the guard off, or train the physics."
        )
    if first_step > int(tcfg.stage_b_warmup_steps) and not state.get("warmup_iou"):
        log.warning(
            "outline guard: this run was resumed past its warm-up without a record of the "
            "warm-up's fit, so the guard cannot fire; the audit's outline_drop verdict is "
            "absent for the same reason."
        )


def _outline_drifted(cfg, model, data, state: dict, step: int, stage_b_keys) -> bool:
    """The outline guard (``training.outline_guard_every``): every so many steps
    after the warm-up, measure the fit and append it to the run's record; True
    when its mean has fallen more than ``outline_guard_drop`` below the warm-up's."""
    tcfg = cfg.training
    every = int(getattr(tcfg, "outline_guard_every", 0))
    if every <= 0 or step % every or not stage_b_keys or step <= tcfg.stage_b_warmup_steps:
        return False
    reference = state.get("warmup_iou")
    if not reference:
        return False
    fit = _outline_fit(cfg, model, data)
    mean = float(np.mean(list(fit.values())))
    floor = float(np.mean(list(reference.values()))) - float(tcfg.outline_guard_drop)
    record = {"step": step, "iou_mean": mean, "iou_min": min(fit.values()), "floor": floor}
    state.setdefault("outline_hist", []).append(record)
    log.info("step %5d | outline guard: IoU mean %.3f (floor %.3f)", step, mean, floor)
    return mean < floor


def _drift_message(state: dict) -> str:
    last = state["outline_hist"][-1]
    return (
        f"outline guard stopped the run at step {last['step']}: IoU mean {last['iou_mean']:.3f} "
        f"fell below the floor {last['floor']:.3f} set by the warm-up's fit minus "
        "training.outline_guard_drop; the checkpoint at this step is kept. The physics "
        "overrode the images: look at the rebalanced weights and the residuals in the log."
    )


def _engage_thermal_source(model, step: int, tcfg, stage_b_keys: tuple[str, ...]) -> None:
    """Hold the thermal source back through the Stage-A warm-up.

    While the Stage-B terms are gated to zero the energy equation and the
    temperature conditions are not in the loss, so a source read from the
    temperature's gradient would let continuity alone shape the temperature:
    the first full-chain benchmark's temperature reached the thousands during
    its warm-up and the physics never recovered. The free ``s`` carries the
    warm-up, as in the legacy Stage A, and the thermal source takes over at the
    step the physics engages. A model without the attribute (a joint view) is
    left alone; the joint trainer refuses the thermal model anyway.
    """
    if not hasattr(model, "thermal_source_engaged"):
        return
    model.thermal_source_engaged = not (
        bool(stage_b_keys) and step <= tcfg.stage_b_warmup_steps
    )


def _stage_b_engages_at(step: int, tcfg, stage_b_keys: tuple[str, ...]) -> bool:
    """True only at the one step the warm-up hands off to Stage-B physics -- where
    the optimiser is restarted. Never true when the warm-up is disabled
    (``stage_b_warmup_steps=0``) or the model has no Stage-B physics to engage."""
    return (
        bool(stage_b_keys)
        and tcfg.stage_b_warmup_steps > 0
        and step == tcfg.stage_b_warmup_steps + 1
    )


def _geometry_frozen(step: int, tcfg, stage_b_keys: tuple[str, ...]) -> bool:
    """Whether ``step`` falls in the physics stage of a two-stage run
    (``training.freeze_geometry``): after a warm-up, with Stage-B physics to
    engage. State-based, so a resumed run lands on the same side."""
    return (
        bool(getattr(tcfg, "freeze_geometry", False))
        and bool(stage_b_keys)
        and int(tcfg.stage_b_warmup_steps) > 0
        and step > int(tcfg.stage_b_warmup_steps)
    )


def _freeze_geometry(model, frozen: bool) -> None:
    """Hold every parameter of the explicit geometry (the capsule, its nose,
    root and width nets) where it is, or release it. The fields' own nets and
    the scalar unknowns stay trainable."""
    for p in model.nets["phi"].parameters():
        p.requires_grad_(not frozen)
    if frozen:
        log.info("geometry frozen at the warm-up's fit: the physics stage trains the fields")


# The terms the temperature-only stage trains on: the liquid energy equation and
# the measured start and inlet temperatures (the interface is at saturation by
# construction of the gated temperature).
TEMPERATURE_TERMS = frozenset({"energy_liquid", "theta_bc"})


def _temperature_stage(step: int, tcfg) -> bool:
    """Whether ``step`` falls in the temperature-only stage
    (``training.temperature_stage_from``). State-based, so a resumed run lands on
    the same side."""
    start = int(getattr(tcfg, "temperature_stage_from", 0))
    return start > 0 and step > start


def _temperature_stage_engages_at(step: int, tcfg) -> bool:
    """Whether ``step`` is the first of the temperature-only stage, where the
    flow is locked and the optimiser restarts."""
    start = int(getattr(tcfg, "temperature_stage_from", 0))
    return start > 0 and step == start + 1


def _lr_origin(step: int, tcfg) -> int:
    """The step the learning-rate halving counts from: the temperature-only
    stage restarts the clock, so the temperature trains from the full rate."""
    return int(tcfg.temperature_stage_from) if _temperature_stage(step, tcfg) else 0


def _lock_flow(model) -> None:
    """Hold every parameter but the temperature net's where it is: the outline,
    the velocity, the pressure, the vapour scalars and the wet share."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.nets["T"].parameters():
        p.requires_grad_(True)
    log.info("flow locked: the temperature-only stage trains the temperature net alone")


def _temperature_equations(equations) -> list:
    """The enabled equations the temperature-only stage evaluates."""
    return [eq for eq in equations if eq.weight_key in TEMPERATURE_TERMS]


def _temperature_plan(plan: _LossPlan) -> _LossPlan:
    """``plan`` narrowed to the temperature terms, keeping its causal state."""
    coll = _temperature_equations(plan.coll_equations)
    return replace(plan, coll_equations=coll, coll_keys=frozenset(e.weight_key for e in coll))


def _physics_reference(
    weights: dict[str, float], norms: dict[str, float], terms: tuple[str, ...]
) -> float:
    """The rebalancer's reference once the geometry is frozen: the median of
    the rebalanced terms' weighted gradient norms, so the physics terms are
    equalised among themselves (the data term no longer has a gradient)."""
    return float(np.median([norms[name] * weights[name] for name in terms]))


def _restart_for_stage_b(
    model, lr: float, reset_weights: dict[str, float]
) -> tuple[torch.optim.Optimizer, dict[str, float]]:
    """A fresh optimiser and a clean copy of the config loss weights, for the moment
    Stage-B physics engages after the warm-up.

    Carrying Adam's Stage-A state and the warm-up-rebalanced weights across the
    switch destabilises the interface; a fresh-optimiser warm start from a Stage-A
    checkpoint does not, and this reproduces that clean transition in one run."""
    return torch.optim.Adam(model.parameters(), lr=lr), dict(reset_weights)


# The rebalanced weights' fixed bounds: a floor so a term never vanishes from
# the objective, and a ceiling a runaway ratchet stops at.
REBALANCE_FLOOR = 1e-2
REBALANCE_CEILING = 1e3


def _rebalance(
    weights: dict[str, float],
    norms: dict[str, float],
    terms: tuple[str, ...] = REBALANCED_TERMS,
    max_ratio: float = 0.0,
    reference: float | None = None,
) -> None:
    """Nudge weights so each term's gradient matches the data term's. In place.

    ``max_ratio`` (``training.rebalance_max_ratio``) caps every rebalanced weight
    at that multiple of the data weight, under the fixed ceiling: the images are
    the measurement, and no physics term may outweigh them by more than it. Zero
    keeps the fixed ceiling alone. ``reference`` replaces the data term's
    weighted norm as the target (the frozen-geometry stage, where the data term
    has no gradient; see :func:`_physics_reference`).
    """
    if reference is None:
        reference = norms["data"] * weights["data"]
    ceiling = REBALANCE_CEILING
    if max_ratio > 0.0:
        ceiling = min(ceiling, max_ratio * weights["data"])
    for name in terms:
        target = reference / norms[name]
        # Half-step towards the target: full steps oscillate.
        weights[name] = float(
            np.clip(0.5 * weights[name] + 0.5 * target, REBALANCE_FLOOR, ceiling)
        )


def train(
    cfg,
    paths: RunPaths,
    steps: int | None = None,
    on_log: Callable[[dict], None] | None = None,
    use_warmup_cache: bool = True,
) -> tuple[BubblePINN, BubbleDataset, dict]:
    """Train (or continue training) and write the checkpoint. Returns the model.

    ``on_log``, when given, receives a copy of each history record as it is
    logged, so a caller can observe progress while the run is still going.
    Under ``training.warmup_cache`` a fresh run starts from the shared
    warm-up's checkpoint, training and caching it first when it is missing
    (see :mod:`naviernet.warmup_cache`); ``use_warmup_cache=False`` is the call
    that trains it.
    """
    # Validate the held-out split up front, so a misconfigured `heldout_datasets`
    # fails loudly on every path -- including a single-dataset CLI run, which never
    # reaches `_train_joint` and its own guard.
    training_datasets(cfg)
    if len(resolved_datasets(cfg)) > 1:
        return _train_joint(cfg, paths, steps=steps, on_log=on_log)

    tcfg = cfg.training
    _validate_training_config(cfg)
    if use_warmup_cache and warmup_cache.enabled(cfg):
        steps = warmup_cache.reuse_or_train(
            cfg,
            paths,
            int(steps if steps is not None else tcfg.steps),
            _stage_b_keys(cfg),
            lambda n: train(cfg, paths, steps=n, on_log=on_log, use_warmup_cache=False),
        )
    rba = tcfg.weighting == "rba"
    pressure_shape = bool(getattr(cfg.model, "pressure_shape", False))
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
    bind_cover_times(cfg, model, data)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    kin_plan = _kinematic_plan(cfg, data, device)
    fv_plan = _front_velocity_plan(cfg, data, device)
    root_plan = _root_plan(cfg, data, device)
    nose_plan = _nose_plan(cfg, data, device)
    front_times = _front_times(cfg, data, device)
    boundaries = _boundary_plan(cfg)
    mass_plan = _mass_balance_plan(cfg, data, groups, device)
    closure_plan = _growth_closure_plan(cfg, data, device)
    thermal = _sharp_thermal(cfg)
    if thermal:
        log.info(
            "energy budget of the measured growth: %s",
            _fmt(heating.budget_summary(heating.growth_budget(data, cfg, groups))),
        )

    equations = registry.enabled_equations(cfg.model.fields, **registry.flags_from(cfg.model))
    rebalanced = registry.rebalanced_terms(equations)
    stage_b_keys = registry.stage_b_terms(equations)
    coll_equations = registry.collocation_equations(equations)
    coll_keys = frozenset(e.weight_key for e in coll_equations)
    state = _initial_state(cfg, equations)
    if paths.checkpoint.exists():
        ckpt = torch.load(paths.checkpoint, map_location=device, weights_only=False)
        _check_architecture_compat(cfg, ckpt, paths.checkpoint)
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
            if "warmup_iou" in ckpt["state"]:
                # The outline guard and the audit hold the run to this record.
                state["warmup_iou"] = ckpt["state"]["warmup_iou"]
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
            else data.sample_collocation(
                tcfg.n_coll, np.random.default_rng(tcfg.seed), tcfg.collocation_time_bins
            ).detach()
        )
        attention = _init_attention(state, coll_keys, x_pool.shape[0], device)
        state.setdefault("rba_resamples", 0)

    first_step = state["done"] + 1
    last_step = state["done"] + steps
    plan = _LossPlan(
        coll_equations,
        coll_keys,
        first_step,
        last_step,
        tcfg,
        _causal_schedule(tcfg, first_step, last_step, state),
    )
    _freeze_nose_through_warmup(model, tcfg, stage_b_keys, first_step)
    _require_guard_arms(tcfg, stage_b_keys, state, first_step)
    if _geometry_frozen(first_step, tcfg, stage_b_keys):
        _freeze_geometry(model, True)  # a run resumed inside the physics stage
    if _temperature_stage(first_step, tcfg):
        _lock_flow(model)  # a run resumed inside the temperature-only stage
        equations = _temperature_equations(equations)
        plan = _temperature_plan(plan)
    log.info("training steps %d-%d on %s", first_step, last_step, device)

    for step in range(first_step, last_step + 1):
        model.eps = annealed_alpha_eps(step, cfg)
        if _temperature_stage_engages_at(step, tcfg):
            _lock_flow(model)
            equations = _temperature_equations(equations)
            plan = _temperature_plan(plan)
            opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
            log.info("step %5d | temperature-only stage engaged", step)
        temperature_only = _temperature_stage(step, tcfg)
        lr = tcfg.lr * (0.5 ** ((step - _lr_origin(step, tcfg)) // tcfg.lr_halflife))
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
            else data.sample_collocation(tcfg.n_coll, rng, tcfg.collocation_time_bins)
        )
        inlet, walls = data.sample_boundary(tcfg.n_bc, rng)
        outlet = data.sample_outlet(tcfg.n_bc, rng) if boundaries.outlet else None
        initial = data.sample_initial(tcfg.n_bc, rng) if thermal else None

        # Pressure-driven shape: solve the width modes at exactly the instants this
        # step will evaluate, so every point gets its own instant's solved bubble
        # rather than a neighbour's. Gated on the Stage-B warm-up for the same
        # reason every Stage-B term is: during the warm-up `p` is untrained, and a
        # geometry solved from a random pressure field would destroy the interface
        # the warm-up exists to converge.
        if pressure_shape and step > tcfg.stage_b_warmup_steps:
            bind_solved_shape(
                model,
                torch.unique(torch.cat([x_data[:, 2], x_coll[:, 2].detach()])).reshape(-1, 1),
                groups,
                cfg,
            )

        _engage_thermal_source(model, step, tcfg, stage_b_keys)
        if _jitter_rng(tcfg, rng) is not None:
            front_times = _front_times(cfg, data, device, rng)
            mass_plan = _mass_balance_plan(cfg, data, groups, device, rng)
        ctx = registry.LossContext(
            model,
            x_coll,
            inlet,
            walls,
            u_inlet,
            groups,
            front_times=front_times,
            outlet=outlet,
            wall_condition=boundaries.wall_condition,
            initial=initial,
        )
        # The temperature-only stage has no use for the outline's data term: the
        # outline is locked, and the term would only cost a forward pass.
        losses = (
            {}
            if temperature_only
            else {"data": ((model.alpha(x_data) - alpha_target) ** 2).mean()}
        )
        for eq in equations:
            losses[eq.weight_key] = eq.term(ctx)
        coll_batches = [CollocationBatch(model, x_coll, groups, None, ctx)]  # single dataset

        # Gradient-norm rebalancing is the legacy scheme only; RBA replaces it with the
        # bounded per-point attention below. Skip the handoff step: weights reset there.
        frozen = _geometry_frozen(step, tcfg, stage_b_keys)
        # Once the geometry is frozen the physics terms keep their configured
        # weights unless the run asks for them to be equalised.
        held = frozen and str(getattr(tcfg, "frozen_weighting", "fixed")) == "fixed"
        if (
            not rba
            and not held
            and not temperature_only
            and step % tcfg.rebalance_every == 0
            and not _stage_b_engages_at(step, tcfg, stage_b_keys)
        ):
            # Once the geometry is frozen the data term has no gradient: the
            # physics terms are measured and equalised among themselves.
            measured = {k: v for k, v in losses.items() if k != "data"} if frozen else losses
            norms = _gradient_norms(model, measured, opt)
            _rebalance(
                weights,
                norms,
                rebalanced,
                float(tcfg.rebalance_max_ratio),
                _physics_reference(weights, norms, rebalanced) if frozen else None,
            )
            log.info("step %5d | rebalanced weights: %s", step, _fmt(weights))

        if _stage_b_engages_at(step, tcfg, stage_b_keys):
            # Engage Stage-B physics on the interface the warm-up converged, from a
            # clean optimiser (see _restart_for_stage_b). The warm-up's fit is
            # recorded first (the floor the outline guard and the audit hold the
            # run to), then the nose DOF releases, so the fresh optimiser sees it
            # trainable.
            _record_warmup_fit(cfg, model, data, state)
            _release_nose_dof(model)
            if frozen:
                _freeze_geometry(model, True)
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
        if kin_plan is not None and not temperature_only:
            # Outside the causal gate, RBA, and the rebalancer by construction:
            # added directly to the total with fixed config weights.
            kin_total, kin_record = kinematics.kinematic_losses(
                kinematics.KinematicContext(model, tcfg, groups),
                kin_plan,
                _kin_schedule(step, first_step, tcfg),
            )
            total = total + kin_total
        fv_record: dict[str, float] = {}
        if fv_plan is not None and not frozen:
            # Outside the causal gate, RBA and the rebalancer, like the kinematic
            # terms: supervised functionals on the front, not pointwise residuals.
            # Not evaluated once the geometry is frozen: they act on it alone.
            fv_total, fv_record = front_velocity.front_velocity_losses(
                front_velocity.FrontVelocityContext(model, tcfg), fv_plan
            )
            total = total + fv_total
        root_record: dict[str, float] = {}
        if root_plan is not None and not frozen:
            # Outside the causal gate, RBA and the rebalancer, same reason.
            root_total, root_record = root_supervision.root_losses(
                cap_supervision.SupervisionContext(model, tcfg), root_plan
            )
            total = total + root_total
        nose_record: dict[str, float] = {}
        if nose_plan is not None and not frozen:
            # Outside the causal gate, RBA and the rebalancer, same reason.
            nose_total, nose_record = nose_supervision.nose_losses(
                cap_supervision.SupervisionContext(model, tcfg), nose_plan
            )
            total = total + nose_total
        mass_record: dict[str, float] = {}
        if mass_plan is not None and not temperature_only:
            # A functional on a fixed quadrature, outside the causal gate, RBA and
            # the rebalancer, like the terms above. A balance of the flow, so not
            # evaluated once the flow is locked.
            mass_total, mass_record = mass_balance.mass_balance_loss(model, mass_plan)
            total = total + float(tcfg.mass_balance_weight) * mass_total
        closure_record: dict[str, float] = {}
        if (
            closure_plan is not None
            and not temperature_only
            and getattr(model, "thermal_source_engaged", False)
        ):
            # The integral of continuity over the bubble, like the mass balance a
            # functional outside the gate, RBA and the rebalancer.
            closure_total, closure_record = growth_closure.growth_closure_loss(
                model, closure_plan
            )
            total = total + float(tcfg.growth_closure_weight) * closure_total
        total.backward()
        grad_norm = _clip_gradients(model, tcfg)
        opt.step()
        if tcfg.checkpoint_every and step % tcfg.checkpoint_every == 0:
            _checkpoint_mid_run(model, opt, state, weights, plan, cfg, paths, step)
        if _outline_drifted(cfg, model, data, state, step, stage_b_keys):
            # Keep the checkpoint at the step the drift was seen, then stop: a
            # run that has lost the images is not worth the hours left in it.
            _checkpoint_mid_run(model, opt, state, weights, plan, cfg, paths, step)
            raise OutlineDrift(_drift_message(state))

        if step % tcfg.log_every == 0 or step == first_step:
            record = {name: float(loss.detach()) for name, loss in losses.items()}
            record.update(kin_record)
            record.update(fv_record)
            record.update(root_record)
            record.update(nose_record)
            record.update(mass_record)
            record.update(closure_record)
            record.update(_wetting_record(model))
            record["step"] = step
            record["lr"] = lr
            record["grad_norm"] = float(grad_norm)
            record.update(_causal_record(plan, step))
            if tcfg.log_chunk_residuals:
                state.setdefault("chunk_hist", []).append(
                    {
                        "step": step,
                        **_chunk_diagnostics(
                            model, x_coll, groups, coll_equations, tcfg.causal_time_chunks
                        ),
                    }
                )
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
    state["causal_index"] = plan.causal.index if plan.causal is not None else 0
    if rba:
        state["attention"] = attention  # persist per-point attention for resume
        state["rba_pool"] = x_pool  # persist the pool the attention aligns to (RAR evolves it)
        # Fingerprint of the FINAL pool (RAR mutates it in-loop), so a resume can be
        # checked to have restored the identical pool the saved attention aligns to.
        state["rba_pool_sig"] = float(x_pool.sum().item())
    _save_checkpoint(model, opt, state, cfg, paths)
    return model, data, state


def _save_checkpoint(model, opt, state: dict, cfg, paths: RunPaths) -> None:
    """Write the checkpoint atomically (a crash mid-write leaves the previous one)."""
    tmp = paths.checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "state": state,
            "alpha_eps": float(model.eps),
            **_architecture_record(cfg),
        },
        tmp,
    )
    tmp.replace(paths.checkpoint)
    log.info("checkpoint written to %s (%d steps total)", paths.checkpoint, state["done"])


def _checkpoint_mid_run(model, opt, state: dict, weights, plan: _LossPlan, cfg, paths, step):
    """The periodic save (``training.checkpoint_every``): the state as the final
    save would record it at this step, so a resume continues from here. The RBA
    attention is not carried mid-run; a resume re-initialises it."""
    snapshot = {
        **state,
        "done": step,
        "w": weights,
        "causal_index": plan.causal.index if plan.causal is not None else 0,
    }
    _save_checkpoint(model, opt, snapshot, cfg, paths)


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
    # The dataset's measured geometry anchors when the front geometry is on.
    # Bound the same way, so one shared construction lands on each condition's
    # own root, front, channel and time window.
    geometry: GeometryPriors | None = None
    # The times this dataset's interface conditions are sampled at (sharp-interface
    # runs only). Per dataset, because their time windows differ.
    front_times: torch.Tensor | None = None
    # The dataset's fixed kinematic quadrature when the constraints are on.
    kin: kinematics.KinematicPlan | None = None
    # The dataset's measured front kinematics when that supervision is on. Per
    # dataset, because each is measured from its own masks over its own frames.
    fv: front_velocity.FrontVelocityPlan | None = None


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
                geometry=_geometry_priors(cfg, data),
                front_times=_front_times(cfg, data, device),
                kin=_kinematic_plan(cfg, data, device),
                fv=_front_velocity_plan(cfg, data, device),
            )
        )
    return contexts


def _joint_losses(
    model,
    contexts,
    equations,
    tcfg,
    rng,
    boundaries: _BoundaryPlan = _BoundaryPlan("no_slip", False),
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
        view = model.bound(cx.c, pin=cx.pin, geometry=cx.geometry)
        x_data, target = cx.data.sample_supervised(tcfg.n_data, rng)
        x_coll = cx.data.sample_collocation(tcfg.n_coll, rng, tcfg.collocation_time_bins)
        inlet, walls = cx.data.sample_boundary(tcfg.n_bc, rng)
        outlet = cx.data.sample_outlet(tcfg.n_bc, rng) if boundaries.outlet else None

        ctx = registry.LossContext(
            view,
            x_coll,
            inlet,
            walls,
            cx.u_inlet,
            cx.groups,
            c=cx.c,
            front_times=cx.front_times,
            outlet=outlet,
            wall_condition=boundaries.wall_condition,
        )
        c_data = cx.c.expand(x_data.shape[0], -1)
        per_term = {"data": ((view.alpha(x_data, c_data) - target) ** 2).mean()}
        for eq in equations:
            per_term[eq.weight_key] = eq.term(ctx)

        for name, loss in per_term.items():
            aggregate[name] = loss if name not in aggregate else aggregate[name] + loss
        coll_batches.append(CollocationBatch(view, x_coll, cx.groups, cx.c, ctx))

    n = len(contexts)
    return {name: loss / n for name, loss in aggregate.items()}, coll_batches


def _joint_functional(
    model, contexts, term: Callable[[object, _JointDataset], tuple[torch.Tensor, dict]]
) -> tuple[torch.Tensor, dict[str, float]]:
    """A per-dataset functional averaged over a joint run's datasets.

    Every dataset is evaluated through its OWN bound view, so it contributes its
    conditioning row and its measured anchors -- going around that would score
    every condition against the first one's interface. The logged values are the
    across-dataset means of the raw term magnitudes.

    Shared by the kinematic constraints and the measured front velocity: both are
    scalar functionals with per-dataset plans, and only the term itself differs.
    """
    totals, merged = [], {}
    for cx in contexts:
        total, record = term(model.bound(cx.c, pin=cx.pin, geometry=cx.geometry), cx)
        totals.append(total)
        for name, value in record.items():
            merged[name] = merged.get(name, 0.0) + value / len(contexts)
    return torch.stack(totals).mean(), merged


def _joint_kinematic_losses(
    model, contexts, tcfg, schedule: kinematics.KinematicSchedule
) -> tuple[torch.Tensor, dict[str, float]]:
    """The kinematic total over a joint run, on each dataset's own quadrature."""
    return _joint_functional(
        model,
        contexts,
        lambda view, cx: kinematics.kinematic_losses(
            kinematics.KinematicContext(view, tcfg, cx.groups), cx.kin, schedule
        ),
    )


def _joint_front_velocity_losses(
    model, contexts, tcfg
) -> tuple[torch.Tensor, dict[str, float]]:
    """The front-velocity total over a joint run, on each dataset's own measured
    frames -- so one condition's front is never scored against another's
    measurement."""
    return _joint_functional(
        model,
        contexts,
        lambda view, cx: front_velocity.front_velocity_losses(
            front_velocity.FrontVelocityContext(view, tcfg), cx.fv
        ),
    )


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
    _validate_training_config(cfg)
    if warmup_cache.enabled(cfg):
        raise ValueError(
            "training.warmup_cache is single-dataset for now: the key reads one series' "
            "processed archive"
        )
    if float(getattr(tcfg, "growth_closure_weight", 0.0)) > 0.0:
        raise ValueError(
            "training.growth_closure_weight is single-dataset for now: the measured growth "
            "is read from one series' images"
        )
    if str(getattr(cfg.experiment, "footprint_wetting", "uniform")) == "microlayer":
        raise ValueError(
            "experiment.footprint_wetting=microlayer is single-dataset for now: the cover "
            "times are read from one series' images"
        )
    if bool(getattr(tcfg, "freeze_geometry", False)):
        raise NotImplementedError(
            "training.freeze_geometry is single-dataset for now: the two-stage schedule is "
            "not threaded through the joint trainer. Run it on one series, or turn it off."
        )
    if int(getattr(tcfg, "outline_guard_every", 0)) > 0:
        # The guard measures one dataset's fit against its own warm-up record;
        # a joint run has one record per dataset and none is threaded yet.
        raise NotImplementedError(
            "training.outline_guard_every is single-dataset for now: a joint run has no "
            "per-dataset warm-up record to hold each series to. Run the guard on one "
            "series, or turn it off for the joint run."
        )
    if tcfg.weighting == "rba":
        # RBA needs a per-dataset fixed pool + attention; wired for single-dataset runs
        # first (T2.3), joint runs in T2.5. Raise rather than silently use gradnorm.
        raise NotImplementedError(
            "training.weighting='rba' is not yet supported for joint (multi-dataset) "
            "runs; use weighting='gradnorm' for joint runs for now."
        )
    if bool(getattr(cfg.model, "nucleation_root", False)):
        # The root supervision plan is per dataset (like the front-velocity one)
        # and is not yet threaded through the joint contexts; an unsupervised
        # joint w(t) would be exactly the undriven knob this feature exists to
        # avoid. Raise rather than silently train it blind.
        raise NotImplementedError(
            "model.nucleation_root is not yet supported for joint (multi-dataset) "
            "runs: its supervision channel is single-dataset. Run it on one "
            "dataset, or turn it off for the joint run."
        )
    if bool(getattr(cfg.model, "nose_shape", False)):
        # The same reason as the root: an unsupervised joint w_nose(t) would be
        # exactly the undriven knob this feature exists to avoid.
        raise NotImplementedError(
            "model.nose_shape is not yet supported for joint (multi-dataset) "
            "runs: its supervision channel is single-dataset. Run it on one "
            "dataset, or turn it off for the joint run."
        )
    if bool(getattr(tcfg, "mass_balance", False)):
        # Its quadrature is per dataset and not yet threaded through the joint
        # contexts; raise rather than silently train without it.
        raise NotImplementedError(
            "training.mass_balance is not yet supported for joint (multi-dataset) "
            "runs; run it on one dataset, or turn it off for the joint run."
        )
    if _two_phase(cfg):
        # The blend reads the indicator inside the model, where a joint run's
        # per-dataset anchors are not bound; raise rather than blend the wrong front.
        raise NotImplementedError(
            "model.pressure_representation=two_phase is not yet supported for joint "
            "(multi-dataset) runs; use the single representation for the joint run."
        )
    if _sharp_thermal(cfg):
        # Its gate, heater centre and initial batch are per dataset and not yet
        # threaded through the joint contexts.
        raise NotImplementedError(
            "model.thermal_model=sharp is not yet supported for joint (multi-dataset) "
            "runs; use the legacy thermal model for the joint run."
        )
    if bool(getattr(cfg.model, "mirror_symmetry", False)):
        # The mirror axis is one dataset's wall span; a joint run has one per dataset.
        raise NotImplementedError(
            "model.mirror_symmetry is not yet supported for joint (multi-dataset) runs."
        )
    steps = int(steps if steps is not None else tcfg.steps)
    device = torch.device(tcfg.device)
    boundaries = _boundary_plan(cfg)

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
    # The first training dataset's anchors set the geometry nets' initial scales;
    # every dataset's own are bound per call (see `_JointDataset.geometry`), so
    # this choice fixes only where the shared construction starts, not where it
    # lands.
    model = BubblePINN(cfg, n_cond=N_COND, geometry=contexts[0].geometry).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)

    equations = registry.enabled_equations(cfg.model.fields, **registry.flags_from(cfg.model))
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
    plan = _LossPlan(
        coll_equations,
        coll_keys,
        first_step,
        last_step,
        tcfg,
        _causal_schedule(tcfg, first_step, last_step, state),
    )
    log.info(
        "joint training steps %d-%d on %s over %s%s",
        first_step,
        last_step,
        device,
        names,
        f" (holding out {heldout})" if heldout else "",
    )

    for step in range(first_step, last_step + 1):
        model.eps = annealed_alpha_eps(step, cfg)
        lr = tcfg.lr * (0.5 ** (step // tcfg.lr_halflife))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad()

        losses, coll_batches = _joint_losses(model, contexts, equations, tcfg, rng, boundaries)

        # Skip rebalancing on the handoff step: its weights are about to be reset.
        if step % tcfg.rebalance_every == 0 and not _stage_b_engages_at(
            step, tcfg, stage_b_keys
        ):
            _rebalance(
                weights,
                _gradient_norms(model, losses, opt),
                rebalanced,
                float(tcfg.rebalance_max_ratio),
            )
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
        fv_record: dict[str, float] = {}
        if tcfg.front_velocity:
            fv_total, fv_record = _joint_front_velocity_losses(model, contexts, tcfg)
            total = total + fv_total
        total.backward()
        _clip_gradients(model, tcfg)
        opt.step()

        if step % tcfg.log_every == 0 or step == first_step:
            record = {name: float(loss.detach()) for name, loss in losses.items()}
            record.update(kin_record)
            record.update(fv_record)
            record["step"] = step
            record["lr"] = lr
            record.update(_causal_record(plan, step))
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
    state["causal_index"] = plan.causal.index if plan.causal is not None else 0
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "state": state,
            "datasets": spanned,  # every series the run spans
            "training_datasets": names,  # the ones actually supervised
            "heldout_datasets": heldout,  # kept out for the transfer test (axis B)
            "n_cond": N_COND,  # so evaluation rebuilds the conditioned architecture
            "alpha_eps": float(model.eps),  # the thickness the run ENDED at
            **_architecture_record(cfg),  # so evaluation rebuilds the pinned architecture
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
    _check_architecture_compat(cfg, ckpt, paths.checkpoint)
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
    bind_cover_times(cfg, model, data)
    # The interface thickness the run ENDED at, not the one it started from:
    # scoring a sharpened model through its initial blur would measure a
    # different solution than the one that was trained.
    model.eps = float(ckpt.get("alpha_eps", model.eps))
    # Re-solve the pressure-driven shape at the dataset's own frame instants, so
    # everything downstream -- IoU, figures, video, the interface diagnostics --
    # measures the bubble the run TRAINED rather than the learned envelope
    # underneath it. Derived from the pressure in this checkpoint, which is why it
    # is re-solved here instead of being carried in the checkpoint and going
    # stale. A no-op when the feature is off.
    bind_solved_shape(
        model,
        torch.tensor(np.asarray(data.t), dtype=torch.float32, device=device).reshape(-1, 1),
        compute_groups(cfg),
        cfg,
    )
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
    _check_architecture_compat(cfg, ckpt, paths.checkpoint)
    names = list(ckpt.get("datasets") or resolved_datasets(cfg))
    heldout = list(ckpt.get("heldout_datasets", cfg.heldout_datasets))
    contexts = _load_joint_datasets(cfg, paths, device, names)
    model = BubblePINN(
        cfg,
        n_cond=int(ckpt.get("n_cond", N_COND)),
        geometry=contexts[0].geometry,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    # The thickness the run ENDED at, exactly as `load_model` does: a joint run
    # scored through its initial blur is a different solution than the one that
    # was trained.
    model.eps = float(ckpt.get("alpha_eps", model.eps))
    model.eval()
    return model, contexts, heldout


def _fmt(weights: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.3g}" for k, v in weights.items())
