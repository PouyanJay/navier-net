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

import numpy as np
import torch

from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BubblePINN
from naviernet.physics import registry
from naviernet.physics.groups import compute_groups
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
        model.load_state_dict(ckpt["model"])
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

        total = sum(weights[name] * loss for name, loss in losses.items())
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


def load_model(cfg, paths: RunPaths) -> tuple[BubblePINN, BubbleDataset, dict]:
    """Load a trained model and its dataset for evaluation or rendering."""
    if not paths.checkpoint.exists():
        raise FileNotFoundError(
            f"{paths.checkpoint} not found -- run the train stage first:\n"
            f"  naviernet stage=train run_name={cfg.run_name}"
        )
    device = torch.device(cfg.training.device)
    data = BubbleDataset(cfg, paths, device=str(device))
    model = BubblePINN(cfg).to(device)
    ckpt = torch.load(paths.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, data, ckpt["state"]


def _fmt(weights: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.3g}" for k, v in weights.items())
