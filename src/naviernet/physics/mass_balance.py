"""Cross-section mass balance: the integral form of continuity as a functional.

Integrating ``div u = s`` over the channel upstream of a station ``x``, with
impermeable side walls, gives::

    Q(x, t) = Q_in + int_0^x int s dy dx'

the axial volume flux is the inlet flux plus the vapour volume created
upstream. The pointwise residual says this only locally and only where it is
sampled; the published runs lost the whole inlet flux within one channel width
while their local residual looked moderate. This term scores the integral
statement directly, at fixed stations and times, wall to wall, normalised by
``Q_in`` so it is O(1) beside the other losses.

A deterministic quadrature (no RNG, so resume reproduces it), outside causal
weighting, RBA and the gradient-norm rebalancer like the other functionals.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MassBalancePlan:
    """The fixed quadrature: ``points`` are ``(x, y, t)`` rows ordered time-major,
    then station, then transverse point; ``stations`` the x-stations; ``dy`` the
    uniform transverse spacing; ``q_in`` the inlet volume flux over the span."""

    points: torch.Tensor
    stations: torch.Tensor
    n_times: int
    n_stations: int
    n_grid: int
    dy: float
    q_in: float


def build_plan(
    domain,
    span: tuple[float, float],
    groups,
    tcfg,
    device,
    at_times: torch.Tensor | None = None,
) -> MassBalancePlan:
    """``mass_balance_times`` slices across the window (or the given
    ``at_times``), ``mass_balance_stations`` x-stations across the domain,
    ``mass_balance_grid`` points across ``span`` (the wall-to-wall span the run's
    samplers use). ``Q_in`` is the inlet plug velocity over that same span, so the
    balance is consistent with the inlet condition the model is given."""
    lo, hi = span
    n_stations, n_grid = int(tcfg.mass_balance_stations), int(tcfg.mass_balance_grid)
    # Built on the CPU whatever device ``at_times`` lives on (the audit passes a
    # tensor on the model's device), then moved as one: meshgrid refuses mixed
    # devices, which took the GPU audit down after a full run.
    times = (
        torch.linspace(domain.t_min, domain.t_max, int(tcfg.mass_balance_times))
        if at_times is None
        else at_times.detach().reshape(-1).cpu()
    )
    n_times = int(times.numel())
    stations = torch.linspace(domain.x_min, domain.x_max, n_stations)
    ys = torch.linspace(lo, hi, n_grid)
    grid_t, grid_x, grid_y = torch.meshgrid(times, stations, ys, indexing="ij")
    points = torch.stack([grid_x.ravel(), grid_y.ravel(), grid_t.ravel()], dim=1)
    return MassBalancePlan(
        points=points.to(device),
        stations=stations.to(device),
        n_times=n_times,
        n_stations=n_stations,
        n_grid=n_grid,
        dy=float((hi - lo) / (n_grid - 1)),
        q_in=float(groups["u_inlet_star"]) * float(hi - lo),
    )


def axial_flux_defect(
    model, plan: MassBalancePlan, c: torch.Tensor | None = None
) -> torch.Tensor:
    """``(Q(x,t) - Q_in - int_0^x int s dy dx') / Q_in`` on the ``(time, station)`` grid."""
    x = plan.points
    cx = c.expand(x.shape[0], -1) if c is not None else None
    u, _ = model.velocity(x, cx)
    source = model.source(x, cx)
    shape = (plan.n_times, plan.n_stations, plan.n_grid)
    flux = _trapezoid(u.reshape(shape), plan.dy)
    created_per_length = _trapezoid(source.reshape(shape), plan.dy)
    upstream = torch.cumulative_trapezoid(created_per_length, plan.stations, dim=1)
    upstream = torch.cat([torch.zeros_like(upstream[:, :1]), upstream], dim=1)
    return (flux - plan.q_in - upstream) / plan.q_in


def mass_balance_loss(
    model, plan: MassBalancePlan, c: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict[str, float]]:
    """The mean squared normalised defect, plus the floats worth a line in the log:
    the term itself and the worst station's defect."""
    defect = axial_flux_defect(model, plan, c)
    loss = (defect**2).mean()
    return loss, {
        "mass_bal": float(loss.detach()),
        "mass_bal_max": float(defect.detach().abs().max()),
    }


def _trapezoid(values: torch.Tensor, spacing: float) -> torch.Tensor:
    """Composite trapezoid along the last axis of a uniform grid."""
    return spacing * (values.sum(dim=-1) - 0.5 * (values[..., 0] + values[..., -1]))
