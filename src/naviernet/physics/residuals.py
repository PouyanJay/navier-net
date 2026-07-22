"""PDE residuals as composable terms. All quantities are non-dimensional.

Stage A (active)
----------------
VOF transport::

    r_vof = alpha_t + u alpha_x + v alpha_y

Phase-change source terms live *off* the interface under the Hardt-Wondra
treatment, so the interface itself is advected by the local velocity alone.

Continuity with an inferred dilatation source::

    r_div = u_x + v_y - s

``s(x, y, t)`` accounts for phase-change dilatation, of order
``mdot (1/rho_v - 1/rho_l)``. It is penalised away from the interface (see
:func:`source_penalty`) so it cannot degenerate into a free sink absorbing
divergence errors wherever the network finds it convenient.

Stage B (next)
--------------
Momentum, including the Hele-Shaw drag that represents the unresolved
depth direction, and continuum surface tension::

    r_u = rho*(u_t + u u_x + v u_y) + p_x - (1/Re) lap(u)
          + hele_shaw * mu*(alpha) u - (1/We) kappa alpha_x

plus the analogous ``r_v``; energy with the wall source ``q''/(rho cp H)``; and
the Hardt-Wondra evaporation closure ``j_evap = (T_int - T_sat)/(R_int h_lv)``
replacing the free source ``s``. The property fields ``rho*(alpha)`` and
``mu*(alpha)`` are arithmetic mixtures built from ``rho_ratio`` and
``mu_ratio`` in :mod:`naviernet.physics.groups`.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class StageAResiduals(NamedTuple):
    """Residual fields evaluated at a batch of collocation points."""

    vof: torch.Tensor  # volume-fraction transport residual
    div: torch.Tensor  # continuity residual, net of the dilatation source
    source: torch.Tensor  # the inferred source itself, for penalisation
    interface_weight: torch.Tensor  # 1 on the interface, 0 in the bulk (detached)


def gradients(f: torch.Tensor, x: torch.Tensor):
    """Return ``(f_x, f_y, f_t)`` for a scalar field ``f`` evaluated at ``x``.

    ``x`` must have ``requires_grad=True`` and columns ordered ``(x, y, t)``.
    """
    grad = torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]
    return grad[:, 0:1], grad[:, 1:2], grad[:, 2:3]


def interface_indicator(alpha: torch.Tensor) -> torch.Tensor:
    """``4a(1-a)``: unity at the alpha=0.5 interface, decaying to zero in the bulk."""
    return 4.0 * alpha * (1.0 - alpha)


def stage_a_residuals(model, x: torch.Tensor) -> StageAResiduals:
    """Evaluate the Stage-A residuals at collocation points ``x``."""
    alpha = model.alpha(x)
    u, v = model.velocity(x)
    source = model.source(x)

    a_x, a_y, a_t = gradients(alpha, x)
    u_x, _, _ = gradients(u, x)
    _, v_y, _ = gradients(v, x)

    return StageAResiduals(
        vof=a_t + u * a_x + v * a_y,
        div=u_x + v_y - source,
        source=source,
        interface_weight=interface_indicator(alpha).detach(),
    )


def source_penalty(residuals: StageAResiduals) -> torch.Tensor:
    """Penalise dilatation away from the interface, where it is unphysical."""
    return (((1.0 - residuals.interface_weight) * residuals.source) ** 2).mean()


def boundary_losses(model, inlet_x, wall_x, u_inlet: float) -> torch.Tensor:
    """Inlet plug velocity and no-slip side walls (Stage A: velocity only)."""
    u_in, v_in = model.velocity(inlet_x)
    u_wall, v_wall = model.velocity(wall_x)

    inlet = ((u_in - u_inlet) ** 2).mean() + (v_in**2).mean()
    wall = (u_wall**2).mean() + (v_wall**2).mean()
    return inlet + wall
