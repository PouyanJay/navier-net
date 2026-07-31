"""Kinematic growth constraints: physics-only functionals on the late time window.

Diagnosed failure these terms target: outside the supervised window the free
dilatation source collapses (S(t) = integral of alpha*s decays or flips sign
starting exactly at the supervision boundary) and the predicted bubble VOLUME
RATE goes negative while the true bubble keeps growing -- the front stalls and
retreats. The evaporation drive stays healthy, so the energy side is not the
problem; the volume budget is.

The remedy is scalar functionals -- not pointwise residuals -- evaluated on a
FIXED quadrature grid (a resampled global scalar has no noise averaging, so the
grid never resamples) over the last fraction of the time window:

- ``mono``:   ReLU(margin - dQ/dt)^2 -- the bubble must keep growing at least a
              margin of the supervised-tail rate (a plain "don't shrink" penalty
              is satisfiable by flattening growth to zero).
- ``vbal``:   (dQ/dt - S)^2 -- volume growth must come from the dilatation
              source (boundary flux measured negligible for this problem).
- ``kevap``:  ReLU(floor*E - S)^2 -- the source must carry at least a floor of
              the (healthy, detached) evaporation drive; sets the magnitude.

dQ/dt is exact for the discrete estimator via autograd (the sum of dalpha/dt
over the grid), which concentrates the constraint gradient in the interface
band -- exactly the front points the terms are meant to push. Every term is
normalized by the dataset's supervised-tail growth rate so it is O(1) against
the other losses. The terms sit OUTSIDE causal weighting (which down-weights
the very window being fixed), RBA, and the gradient-norm rebalancer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from naviernet.physics.residuals import KAPPA_EPS


@dataclass(frozen=True)
class KinematicContext:
    """Per-call inputs shared by the kinematic terms: the model (a pin-bound
    view on a joint run), the training config, the dataset's dimensionless
    groups, and its conditioning row (``None`` for a single-dataset run)."""

    model: object
    tcfg: object
    groups: dict[str, float]
    c: torch.Tensor | None = None


class KinematicSchedule(NamedTuple):
    """Ramp factors in [0, 1] for the two source-reading terms. The monotone
    term is always at full weight; ``balance`` ramps over the call's first
    ``kin_ramp`` steps (the source field is noise early); ``evap`` additionally
    waits out the Stage-B warm-up (the temperature net receives no gradient
    until the warm-up ends, so flooring against it earlier would push the
    source toward an untrained pattern)."""

    balance: float
    evap: float


@dataclass(frozen=True)
class KinematicPlan:
    """Per-run-constant quadrature and references for the kinematic terms.

    ``points`` is the fixed grid, ``(kin_times * kin_grid^2, 3)`` ordered
    ``(x, y, t)``; a fresh ``requires_grad`` leaf is cloned from it each step.
    ``r_ref`` is the dataset's measured supervised-tail growth rate.
    """

    points: torch.Tensor
    n_times: int
    n_space: int
    area: float
    r_ref: float


def build_plan(domain, r_ref: float, tcfg, device) -> KinematicPlan:
    """The fixed quadrature over the late window: ``kin_grid^2`` spatial points
    at ``kin_times`` slices spanning the last ``kin_time_frac`` of the run's
    time domain. Deterministic -- no RNG, so resume reproduces it exactly."""
    n_grid, n_times = int(tcfg.kin_grid), int(tcfg.kin_times)
    xs = torch.linspace(domain.x_min, domain.x_max, n_grid)
    ys = torch.linspace(domain.y_min, domain.y_max, n_grid)
    t_start = domain.t_max - float(tcfg.kin_time_frac) * (domain.t_max - domain.t_min)
    times = torch.linspace(t_start, domain.t_max, n_times)

    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    space = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=1)
    n_space = space.shape[0]
    points = torch.cat(
        [space.repeat(n_times, 1), times.repeat_interleave(n_space).unsqueeze(1)], dim=1
    )
    return KinematicPlan(points.to(device), n_times, n_space, float(domain.area), float(r_ref))


class _StepFields(NamedTuple):
    """One step's shared intermediates: the fresh quadrature leaf, its expanded
    conditioning, the autograd pass over the slice volumes, and the normalized
    per-slice quantities every term reads."""

    x: torch.Tensor
    cx: torch.Tensor | None
    grads: torch.Tensor
    q_dot: torch.Tensor
    s_slices: torch.Tensor


def kinematic_losses(
    ctx: KinematicContext, plan: KinematicPlan, schedule: KinematicSchedule
) -> tuple[torch.Tensor, dict[str, float]]:
    """The weighted kinematic total for one step, plus per-term floats for the
    log. Recorded values are the raw (unramped, unweighted) term magnitudes."""
    tcfg = ctx.tcfg
    fields = _step_fields(ctx, plan)

    mono = (torch.relu(float(tcfg.kin_margin_frac) - fields.q_dot) ** 2).mean()
    vbal = ((fields.q_dot - fields.s_slices) ** 2).mean()

    total = (
        float(tcfg.kin_weight_mono) * mono
        + schedule.balance * float(tcfg.kin_weight_balance) * vbal
    )
    record = {"kin_mono": float(mono.detach()), "kin_vbal": float(vbal.detach())}

    if float(tcfg.kin_weight_evap) > 0:
        kevap = _evap_floor(ctx, plan, fields)
        total = total + schedule.evap * float(tcfg.kin_weight_evap) * kevap
        record["kin_evap"] = float(kevap.detach())
    return total, record


def _step_fields(ctx: KinematicContext, plan: KinematicPlan) -> _StepFields:
    """Evaluate the model on a fresh quadrature leaf and derive the per-slice
    volume rate and source integral (both normalized by ``r_ref``)."""
    x = plan.points.clone().requires_grad_(True)
    cx = ctx.c.expand(x.shape[0], -1) if ctx.c is not None else None

    alpha = ctx.model.alpha(x, cx)
    # Q(t_k) = area * mean_i alpha; summing the slices lets one autograd pass
    # yield every slice's dQ/dt (each point contributes only to its own slice,
    # so grads[:, 2] holds (area/n_space) * dalpha/dt per point).
    q_slices = alpha.reshape(plan.n_times, plan.n_space).mean(dim=1) * plan.area
    grads = torch.autograd.grad(q_slices.sum(), x, create_graph=True)[0]
    q_dot = grads[:, 2].reshape(plan.n_times, plan.n_space).sum(dim=1) / plan.r_ref

    # S(t_k) = area * mean_i alpha*s -- the vapour-side dilatation actually
    # feeding the volume budget (the diagnosed quantity that collapses).
    source = ctx.model.source(x, cx)
    s_slices = (
        (alpha * source).reshape(plan.n_times, plan.n_space).mean(dim=1)
        * plan.area
        / plan.r_ref
    )
    return _StepFields(x, cx, grads, q_dot, s_slices)


def _evap_floor(
    ctx: KinematicContext, plan: KinematicPlan, fields: _StepFields
) -> torch.Tensor:
    """``ReLU(floor*E - S)^2``: the source must carry at least ``kin_evap_floor``
    of the evaporation drive ``E`` -- the budget the diagnosis showed stays
    healthy while ``s`` dies. ``E`` mirrors the energy residual's Hardt-Wondra
    flux (prefactor, Ja, theta/r_int, interfacial density regularized by the
    same ``KAPPA_EPS``) and is DETACHED: the floor pushes the source up, never
    drags the temperature down (same one-way convention as the existing
    evaporation mass closure)."""
    model, groups = ctx.model, ctx.groups
    with torch.no_grad():
        # grads[:, :2] is (area/n_space) * grad(alpha); undo the quadrature scale.
        grad_alpha = fields.grads[:, :2] * (plan.n_space / plan.area)
        delta = torch.sqrt((grad_alpha**2).sum(dim=1, keepdim=True) + KAPPA_EPS**2)
        theta = model.temperature(fields.x, fields.cx)
        flux = (1.0 - 1.0 / groups["rho_ratio"]) * groups["Ja"] * (theta / model.r_int_star)
        drive = (
            (flux * delta).reshape(plan.n_times, plan.n_space).mean(dim=1)
            * plan.area
            / plan.r_ref
        )
    return (torch.relu(float(ctx.tcfg.kin_evap_floor) * drive - fields.s_slices) ** 2).mean()
