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
from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BubblePINN
from naviernet.physics import registry
from naviernet.physics.groups import N_COND, compute_groups, conditioning_vector
from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths

log = get_logger(__name__)

# Loss terms whose weights the rebalancer adjusts, derived from the registry for
# the Stage-A field set: `data` is the reference scale and `src` is a deliberate
# soft penalty, so both stay out. Kept as a module constant for callers that
# rebalance a Stage-A weight dict directly.
STAGE_A_FIELDS = ("phi", "u", "v", "s")
REBALANCED_TERMS = registry.rebalanced_terms(registry.enabled_equations(STAGE_A_FIELDS))


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
    if len(resolved_datasets(cfg)) > 1:
        return _train_joint(cfg, paths, steps=steps, on_log=on_log)

    tcfg = cfg.training
    steps = int(steps if steps is not None else tcfg.steps)
    device = torch.device(tcfg.device)

    paths.ensure()
    torch.manual_seed(tcfg.seed)

    groups = compute_groups(cfg)
    u_inlet = groups["u_inlet_star"]
    data = BubbleDataset(cfg, paths, device=str(device))
    model = BubblePINN(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)

    equations = registry.enabled_equations(cfg.model.fields)
    rebalanced = registry.rebalanced_terms(equations)
    state = _initial_state(cfg, equations)
    if paths.checkpoint.exists():
        ckpt = torch.load(paths.checkpoint, map_location=device, weights_only=False)
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

    first_step = state["done"] + 1
    last_step = state["done"] + steps
    log.info("training steps %d-%d on %s", first_step, last_step, device)

    for step in range(first_step, last_step + 1):
        lr = tcfg.lr * (0.5 ** (step // tcfg.lr_halflife))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad()

        x_data, alpha_target = data.sample_supervised(tcfg.n_data, rng)
        x_coll = data.sample_collocation(tcfg.n_coll, rng)
        inlet, walls = data.sample_boundary(tcfg.n_bc, rng)

        ctx = registry.LossContext(model, x_coll, inlet, walls, u_inlet, groups)
        losses = {"data": ((model.alpha(x_data) - alpha_target) ** 2).mean()}
        for eq in equations:
            losses[eq.weight_key] = eq.term(ctx)

        if step % tcfg.rebalance_every == 0:
            _rebalance(weights, _gradient_norms(model, losses, opt), rebalanced)
            log.info("step %5d | rebalanced weights: %s", step, _fmt(weights))

        src_factor, evap_factor = _curriculum(step, tcfg.curriculum_steps)
        schedule = {"src": src_factor, "evap": evap_factor}
        total = sum(
            weights[name] * schedule.get(name, 1.0) * loss for name, loss in losses.items()
        )
        total.backward()
        opt.step()

        if step % tcfg.log_every == 0 or step == first_step:
            record = {name: float(loss.detach()) for name, loss in losses.items()}
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
        {"model": model.state_dict(), "opt": opt.state_dict(), "state": state},
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


def _load_joint_datasets(cfg, paths: RunPaths, device, names=None) -> list[_JointDataset]:
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
        contexts.append(_JointDataset(name, data, groups, c, float(groups["u_inlet_star"])))
    return contexts


def _joint_losses(model, contexts, equations, tcfg, rng) -> dict[str, torch.Tensor]:
    """Each loss term averaged over the datasets, every dataset's residuals
    evaluated with its own conditioning row so one model fits them all."""
    aggregate: dict[str, torch.Tensor] = {}
    for cx in contexts:
        x_data, target = cx.data.sample_supervised(tcfg.n_data, rng)
        x_coll = cx.data.sample_collocation(tcfg.n_coll, rng)
        inlet, walls = cx.data.sample_boundary(tcfg.n_bc, rng)

        ctx = registry.LossContext(model, x_coll, inlet, walls, cx.u_inlet, cx.groups, c=cx.c)
        c_data = cx.c.expand(x_data.shape[0], -1)
        per_term = {"data": ((model.alpha(x_data, c_data) - target) ** 2).mean()}
        for eq in equations:
            per_term[eq.weight_key] = eq.term(ctx)

        for name, loss in per_term.items():
            aggregate[name] = loss if name not in aggregate else aggregate[name] + loss

    n = len(contexts)
    return {name: loss / n for name, loss in aggregate.items()}


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
    steps = int(steps if steps is not None else tcfg.steps)
    device = torch.device(tcfg.device)

    # The datasets actually supervised: held-out conditions (axis B) are loaded
    # only at evaluation, never here, so they contribute nothing to the loss. This
    # also validates the split (rejects an all-held-out run) before any I/O.
    train_names = training_datasets(cfg)
    heldout = [name for name in resolved_datasets(cfg) if name not in set(train_names)]

    paths.ensure()
    torch.manual_seed(tcfg.seed)

    contexts = _load_joint_datasets(cfg, paths, device, train_names)
    names = [cx.name for cx in contexts]
    model = BubblePINN(cfg, n_cond=N_COND).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)

    equations = registry.enabled_equations(cfg.model.fields)
    rebalanced = registry.rebalanced_terms(equations)
    state = _initial_state(cfg, equations)

    rng = np.random.default_rng(tcfg.seed + state["done"])
    weights = state["w"]
    first_step = state["done"] + 1
    last_step = state["done"] + steps
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

        losses = _joint_losses(model, contexts, equations, tcfg, rng)

        if step % tcfg.rebalance_every == 0:
            _rebalance(weights, _gradient_norms(model, losses, opt), rebalanced)
            log.info("step %5d | rebalanced weights: %s", step, _fmt(weights))

        src_factor, evap_factor = _curriculum(step, tcfg.curriculum_steps)
        schedule = {"src": src_factor, "evap": evap_factor}
        total = sum(
            weights[name] * schedule.get(name, 1.0) * loss for name, loss in losses.items()
        )
        total.backward()
        opt.step()

        if step % tcfg.log_every == 0 or step == first_step:
            record = {name: float(loss.detach()) for name, loss in losses.items()}
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
            "datasets": resolved_datasets(cfg),  # every series the run spans
            "training_datasets": names,  # the ones actually supervised
            "heldout_datasets": heldout,  # kept out for the transfer test (axis B)
            "n_cond": N_COND,  # so evaluation rebuilds the conditioned architecture
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
    data = BubbleDataset(cfg, paths, device=str(device))
    # `n_cond` (0 for a single-dataset checkpoint) rebuilds the right architecture,
    # so a conditioned joint checkpoint loads without a shape mismatch.
    model = BubblePINN(cfg, n_cond=int(ckpt.get("n_cond", 0))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, data, ckpt["state"]


def load_joint(cfg, paths: RunPaths) -> tuple[BubblePINN, list[_JointDataset]]:
    """The conditioned model and its per-dataset contexts, for evaluating a joint
    run. Mirrors :func:`load_model` but returns every dataset the run spans, each
    with its conditioning row, so evaluation scores them all."""
    if not paths.checkpoint.exists():
        raise FileNotFoundError(
            f"{paths.checkpoint} not found -- run the train stage first:\n"
            f"  naviernet stage=train run_name={cfg.run_name}"
        )
    device = torch.device(cfg.training.device)
    ckpt = torch.load(paths.checkpoint, map_location=device, weights_only=False)
    contexts = _load_joint_datasets(cfg, paths, device)
    model = BubblePINN(cfg, n_cond=int(ckpt.get("n_cond", N_COND))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, contexts


def _fmt(weights: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.3g}" for k, v in weights.items())
