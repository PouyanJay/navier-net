"""Growth closure: the vapour the heat makes against the growth the images show.

Integrating ``div u = s`` over the bubble with the kinematic condition on its
edge gives, at every instant::

    dA/dt = int s dA

the bubble's area grows exactly as fast as the source creates vapour. The
pointwise kinematic condition says this only locally, and on Series-1 the slip
at the advancing nose dominates it; nothing scored the total, and with the film
thickness free the fit made more vapour than the bubble holds (1.6 times over
the event). This term scores the total directly at every event frame against
the measured growth, normalised by the mean measured rate so it is O(1).

The source has two parts: the outline flux, which the temperature carries, and
the film under the footprint. The outline part enters detached, so this total
trains the footprint's wetting (its wet share or film thickness) and never
bends the temperature to fix an integral; it is read without a parameter graph,
since no backward pass reaches it. ``training.growth_closure_outline_gradient``
keeps its gradient instead, so the closure also trains the temperature. A functional on a fixed quadrature,
outside causal weighting, RBA and the rebalancer, like the mass balance.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from naviernet.data.dataset import INTERFACE_ALPHA

# The covered floor is widened by this many pixels before it is sampled, so the
# footprint the model draws a little outside the masks is still integrated.
FOOTPRINT_MARGIN_PX = 6


@dataclass(frozen=True)
class GrowthClosurePlan:
    """``points`` are the sampled floor pixels ``(x, y)``, ``cell`` the area each
    stands for, ``times`` the event frames' instants, ``growth`` the measured
    ``dA/dt`` at each, ``scale`` their mean magnitude."""

    points: torch.Tensor
    cell: float
    times: torch.Tensor
    growth: torch.Tensor
    scale: float


def measured_growth_rate(data, row: int) -> float:
    """``dA/dt`` of the measured vapour area at ``row``, in ``L_ref^2 / t_ref``:
    central between the neighbouring frames, one-sided at the ends."""
    pixel = float(data.x[1] - data.x[0]) * float(data.y[1] - data.y[0])
    areas = {
        r: float(((data.alpha[r] > INTERFACE_ALPHA) & (data.valid[r] > 0)).sum()) * pixel
        for r in (row - 1, row, row + 1)
        if 0 <= r < data.n_event
    }
    rows = sorted(areas)
    first, last = rows[0], rows[-1]
    return (areas[last] - areas[first]) / (float(data.t[last]) - float(data.t[first]))


def _covered_floor(data, stride: int) -> np.ndarray:
    """Every ``stride``-th pixel of the floor any event frame shows covered,
    widened by the margin, as a ``(H, W)`` mask."""
    covered = (
        (data.alpha[: data.n_event] > INTERFACE_ALPHA) & (data.valid[: data.n_event] > 0)
    ).any(0)
    grown = covered.copy()
    for _ in range(FOOTPRINT_MARGIN_PX):
        grown[1:, :] |= grown[:-1, :].copy()
        grown[:-1, :] |= grown[1:, :].copy()
        grown[:, 1:] |= grown[:, :-1].copy()
        grown[:, :-1] |= grown[:, 1:].copy()
    sampled = np.zeros_like(grown)
    sampled[::stride, ::stride] = True
    return grown & sampled


def build_plan(data, stride: int, device) -> GrowthClosurePlan:
    if stride < 1:
        raise ValueError(f"training.growth_closure_stride must be >= 1, got {stride}")
    rows, cols = np.nonzero(_covered_floor(data, stride))
    points = np.column_stack([data.x[cols], data.y[rows]]).astype(np.float32)
    cell = abs(float(data.x[1] - data.x[0]) * float(data.y[1] - data.y[0])) * stride**2
    growth = np.array([measured_growth_rate(data, r) for r in range(data.n_event)])
    return GrowthClosurePlan(
        points=torch.tensor(points, device=device),
        cell=cell,
        times=torch.tensor(np.asarray(data.t[: data.n_event], dtype=np.float32), device=device),
        growth=torch.tensor(growth, dtype=torch.float32, device=device),
        scale=max(float(np.abs(growth).mean()), 1e-9),
    )


def outline_vapour(model, instant: torch.Tensor) -> torch.Tensor:
    """The outline flux's part of ``int s dA`` at one instant: the coefficient
    times the flux integrated along the front (the kernel is normalised)."""
    flux, weights, _ = model._front_flux(instant, None)
    return model._source_coefficient * (flux * weights).sum()


@contextmanager
def _without_parameter_graph(model):
    """Within the block no parameter records a graph: a value read here and
    then detached costs no bookkeeping for a backward pass that never comes.
    Every value is the same as outside it."""
    flags = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


def source_totals(
    model, plan: GrowthClosurePlan, outline_gradient: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """``int s dA`` at every event frame, ``(n,)``, split into the footprint's
    part (with its gradient) and the outline's: detached and read without a
    parameter graph, or with its gradient to the temperature under
    ``outline_gradient`` (``training.growth_closure_outline_gradient``)."""
    footprint, outline = [], []
    for instant in plan.times:
        x = torch.cat([plan.points, instant.expand(len(plan.points), 1)], dim=1)
        footprint.append(model.footprint_source(x).sum() * plan.cell)
        if outline_gradient:
            outline.append(outline_vapour(model, instant))
        else:
            with _without_parameter_graph(model):
                outline.append(outline_vapour(model, instant).detach())
    return torch.stack(footprint), torch.stack(outline)


def growth_closure_loss(
    model, plan: GrowthClosurePlan, outline_gradient: bool = False
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean squared gap between the vapour made and the measured growth over
    the event frames, on the mean measured rate; plus its log record."""
    footprint, outline = source_totals(model, plan, outline_gradient)
    gap = (footprint + outline - plan.growth) / plan.scale
    loss = (gap**2).mean()
    return loss, {
        "growth_closure": float(loss.detach()),
        "growth_closure_max": float(gap.detach().abs().max()),
    }
