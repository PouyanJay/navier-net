"""The bubble's depth beside its outline: what the camera's outline hides.

The images give the bubble's outline from above. Across the channel's depth the
bubble is not a slab filling that outline: it rides on the films it leaves on
both plates, and its sides are half-circles of radius ``r = (H - 2 delta) / 2``
across the depth (the Hele-Shaw bubble on wetted plates). The cross-section
behind the outline is therefore

    A_b = pi r^2 + (w - 2 r) 2 r        (w the outline's width, w >= 2 r)

against the ``w H`` a depth-averaged model assumes. The difference is liquid
the outline passes over without moving it: the films (``2 delta`` of the
depth) and the four corners beside the rounded sides (``(1 - pi/4) r^2`` each).

Two fractions follow from this geometry and Bretherton's film, nothing fitted:

- the head advancing displaces ``eta_nose = A_b / (w H)`` of what the slab
  would (about 0.83 on Series-1-w300): it leaves the films and the corners
  behind it;
- the body widening or narrowing by ``dw`` changes the cross-section by
  ``(H - 2 delta) dw``, so its outline displaces ``eta_body = (H - 2 delta) / H``
  of the slab's (about 0.95): the corners move with the sides.

Under ``model.depth_bypass`` the depth-averaged liquid moves with ``eta V_n`` at
the outline instead of ``V_n``, consistently: the exact flow's potentials, the
kinematic condition and the mass balance see the same reduced displacement, the
interface advection on the liquid side carries the matching sink, and the
thermal conversion from evaporation to outline area is divided by the
growth-weighted ``eta_eff`` (the same evaporation grows the outline faster when
the bubble is thinner than the channel). It engages with the frozen geometry,
so the shared warm-up is untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from naviernet.physics import mass_balance
from naviernet.physics.film import deposited_thickness

# Instants the fractions are tabulated at once the geometry freezes, linear in
# between: the outline's width and its nose speed change slowly over the window.
TABLE_INSTANTS = 16


def cross_section_fractions(width: float, depth: float, film: float) -> tuple[float, float]:
    """``(eta_nose, eta_body)`` for an outline ``width`` wide in a channel
    ``depth`` deep, a film ``film`` thick on each plate, all in one unit. A
    bubble narrower than the depth is read as a circle of its own width."""
    r = min(0.5 * max(depth - 2.0 * film, 0.0), 0.5 * width)
    flat = max(width - 2.0 * r, 0.0)
    area = math.pi * r**2 + flat * 2.0 * r
    eta_nose = area / (width * depth) if width > 0.0 else 1.0
    eta_body = 2.0 * r / depth
    return min(eta_nose, 1.0), min(eta_body, 1.0)


@dataclass(frozen=True)
class BypassTable:
    """The fractions along the window, one row per instant: ``times`` ``(T,)``
    and, per instant, the head's and the body's fractions, the growth-weighted
    ``effective`` one, and ``shoulder``, the x where the nose cap meets the body."""

    times: torch.Tensor
    nose: torch.Tensor
    body: torch.Tensor
    effective: torch.Tensor
    shoulder: torch.Tensor

    def at(self, t: torch.Tensor, name: str) -> torch.Tensor:
        """Column ``name`` at ``t`` (any shape): linear between the rows, held
        at the ends."""
        values, times = getattr(self, name), self.times
        flat = t.detach().reshape(-1).to(times.dtype).clamp(times[0], times[-1])
        hi = torch.bucketize(flat, times).clamp(1, len(times) - 1)
        lo = hi - 1
        weight = (flat - times[lo]) / (times[hi] - times[lo]).clamp(min=1e-12)
        return (values[lo] * (1.0 - weight) + values[hi] * weight).reshape(t.shape)


def build_table(model, groups: dict[str, float]) -> BypassTable:
    """Tabulate the fractions over the frozen geometry's window from its own
    front: the width from the widest point, the film from the nose's speed."""
    # The sampler's own weights; imported here so the physics package does not
    # import the model at load time.
    from naviernet.models.pinn import _arclength_weights

    priors = model.nets["phi"].priors
    device = next(model.parameters()).device
    times = torch.linspace(
        float(priors.t_min), float(priors.t_max), TABLE_INSTANTS, device=device
    )
    n_body, n_cap = _front_counts(model)
    depth = float(groups["H_star"])
    axis = getattr(model, "mirror_axis", None)
    rows = []
    for instant in times:
        front = _front_at(model, instant, n_body, n_cap)
        points = front.points.detach()[:, :2]
        speed = front.normal_speed.detach()[:, 0]
        weights = _arclength_weights(points, n_body, n_cap)[:, 0]
        nose_cap = _nose_cap(front)[:, 0]
        centre = float(points[:, 1].mean()) if axis is None else float(axis)
        width = 2.0 * float((points[:, 1] - centre).abs().max())
        film = float(deposited_thickness(speed.max().reshape(1, 1), groups))
        eta_nose, eta_body = cross_section_fractions(width, depth, film)
        eta = torch.full_like(speed, eta_body)
        eta[nose_cap] = eta_nose
        growth = float((speed * weights).sum())
        if growth > 0.0:
            effective = float((eta * speed * weights).sum()) / growth
        else:
            effective = float((eta * weights).sum() / weights.sum())
        shoulder = (
            float(points[nose_cap, 0].min())
            if bool(nose_cap.any())
            else float(points[:, 0].max())
        )
        rows.append((eta_nose, eta_body, effective, shoulder))
    columns = torch.tensor(rows, dtype=times.dtype, device=device).T
    return BypassTable(times, *columns)


def on_front(model, front) -> torch.Tensor | None:
    """The fraction per front sample, ``(N, 1)``: the head's on the nose cap,
    the body's elsewhere; None when the bypass is not engaged."""
    table = _table(model)
    if table is None:
        return None
    t = front.points[:, 2:3]
    return torch.where(_nose_cap(front), table.at(t, "nose"), table.at(t, "body"))


def at_points(model, x: torch.Tensor) -> torch.Tensor | None:
    """The fraction per point, ``(N, 1)``: on the liquid side the head's past
    the shoulder and the body's behind it, 1 in the vapour; None when the
    bypass is not engaged."""
    table = _table(model)
    if table is None:
        return None
    t = x[:, 2:3]
    head = x[:, 0:1].detach() > table.at(t, "shoulder")
    eta = torch.where(head, table.at(t, "nose"), table.at(t, "body"))
    liquid = model.liquid_side(x).unsqueeze(1)
    return torch.where(liquid, eta, torch.ones_like(eta))


def effective(model, t: torch.Tensor) -> torch.Tensor:
    """``eta_eff(t)``, the fraction weighted by the front's growth, ``t``'s shape;
    1 when the bypass is not engaged."""
    table = _table(model)
    if table is None:
        return torch.ones_like(t)
    return table.at(t, "effective")


def record(model, t: float) -> dict[str, float] | None:
    """The three fractions at ``t`` for a report; None when not engaged."""
    table = _table(model)
    if table is None:
        return None
    instant = torch.tensor(float(t), device=table.times.device)
    return {
        "eta_nose": float(table.at(instant, "nose")),
        "eta_body": float(table.at(instant, "body")),
        "eta_eff": float(table.at(instant, "effective")),
    }


def mass_balance_sink(model, plan: mass_balance.MassBalancePlan) -> torch.Tensor:
    """The liquid the outline passes over upstream of each station,
    ``sum (1 - eta) V_n dl`` over the front's samples at ``x`` below it, on the
    plan's ``(time, station)`` grid."""
    from naviernet.models.pinn import _arclength_weights

    n_body, n_cap = _front_counts(model)
    shape = (plan.n_times, plan.n_stations, plan.n_grid)
    times = plan.points[:, 2].reshape(shape)[:, 0, 0]
    sink = torch.zeros(
        plan.n_times, plan.n_stations, dtype=plan.points.dtype, device=plan.points.device
    )
    for i, instant in enumerate(times):
        front = _front_at(model, instant, n_body, n_cap)
        points = front.points.detach()
        weights = _arclength_weights(points[:, :2], n_body, n_cap)
        stored = (1.0 - on_front(model, front)) * front.normal_speed.detach() * weights
        upstream = points[:, 0:1] < plan.stations[None, :]
        sink[i] = (stored * upstream).sum(dim=0)
    return sink


def flux_defect(
    model, plan: mass_balance.MassBalancePlan, c: torch.Tensor | None = None
) -> torch.Tensor:
    """:func:`mass_balance.axial_flux_defect` with the stored liquid taken out of
    what each station must carry; the plain defect when the bypass is off."""
    defect = mass_balance.axial_flux_defect(model, plan, c)
    if _table(model) is None:
        return defect
    return defect + mass_balance_sink(model, plan) / plan.q_in


def mass_balance_loss(
    model, plan: mass_balance.MassBalancePlan, c: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict[str, float]]:
    """:func:`mass_balance.mass_balance_loss` on :func:`flux_defect`."""
    defect = flux_defect(model, plan, c)
    loss = (defect**2).mean()
    return loss, {
        "mass_bal": float(loss.detach()),
        "mass_bal_max": float(defect.detach().abs().max()),
    }


def _table(model) -> BypassTable | None:
    if not getattr(model, "bypass_engaged", False):
        return None
    table = getattr(model, "_bypass_table", None)
    if table is None:
        raise RuntimeError(
            "model.depth_bypass is engaged without its table; the trainer builds it "
            "when the geometry freezes"
        )
    return table


def _nose_cap(front) -> torch.Tensor:
    """``(N, 1)`` bool: the samples on the nose cap (the audit's segment too)."""
    return (front.on_cap.detach() > 0.5) & (front.u.detach() >= 0.5)


def _front_counts(model) -> tuple[int, int]:
    model_cfg = model.cfg.model
    return int(model_cfg.front_body_samples), int(model_cfg.front_cap_samples)


def _front_at(model, instant: torch.Tensor, n_body: int, n_cap: int):
    """The front at one instant with its speed (which needs the graph in ``t``)."""
    with torch.enable_grad():
        t = instant.detach().reshape(1, 1).clone().requires_grad_(True)
        return model.front(t, n_body, n_cap)
