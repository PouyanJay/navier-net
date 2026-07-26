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


# --- Stage B: momentum + surface tension, energy + evaporation --------------

# Numerical floor on |grad alpha| in the interface-normal and area-density
# calculations, so curvature and the interfacial delta stay finite where alpha
# is flat (0/0). This is a solver safeguard, not a physical parameter -- it is
# dataset-independent and only needs to sit well below a resolved interface's
# gradient magnitude, so it is a fixed constant rather than a cfg value.
KAPPA_EPS = 1e-3


class StageBResiduals(NamedTuple):
    """Stage-B residual fields evaluated at a batch of collocation points."""

    mom_x: torch.Tensor  # x-momentum residual
    mom_y: torch.Tensor  # y-momentum residual
    energy: torch.Tensor  # energy + evaporation residual
    src_closure: torch.Tensor  # source(x) minus the evaporation closure
    kappa: torch.Tensor  # interface curvature, for inspection
    interface_delta: torch.Tensor  # |grad alpha| area-density delta, for inspection


def mixture(alpha: torch.Tensor, ratio: float) -> torch.Tensor:
    """Arithmetic property blend, scaled so liquid (alpha=0) reads 1.

    ``alpha`` is the vapour fraction, so the blend runs from 1 in liquid to
    ``1/ratio`` in vapour (``ratio`` is the liquid/vapour property ratio).
    """
    return (1.0 - alpha) + alpha / ratio


_Normal = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _interface_normal(alpha: torch.Tensor, x: torch.Tensor, eps: float) -> _Normal:
    """``(a_x, a_y, n_x, n_y, |grad alpha|)``: the interface gradient, its unit
    normal ``grad alpha / |grad alpha|``, and the floored magnitude (the
    interfacial area density)."""
    a_x, a_y, _ = gradients(alpha, x)
    grad_mag = torch.sqrt(a_x**2 + a_y**2 + eps**2)
    return a_x, a_y, a_x / grad_mag, a_y / grad_mag, grad_mag


def _normal_divergence(nx: torch.Tensor, ny: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``div(n) = n_x,x + n_y,y`` -- the one implementation both the standalone
    ``curvature`` and the momentum residual share."""
    nx_x, _, _ = gradients(nx, x)
    _, ny_y, _ = gradients(ny, x)
    return nx_x + ny_y


def curvature(alpha: torch.Tensor, x: torch.Tensor, eps: float = KAPPA_EPS) -> torch.Tensor:
    """Interface curvature ``kappa = -div(grad alpha / |grad alpha|)``.

    The sign is chosen so a compact vapour bubble (alpha high inside) yields a
    positive curvature, hence a positive Laplace pressure jump.
    """
    _, _, nx, ny, _ = _interface_normal(alpha, x, eps)
    return -_normal_divergence(nx, ny, x)


def _laplacian(field: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    f_x, f_y, _ = gradients(field, x)
    f_xx, _, _ = gradients(f_x, x)
    _, f_yy, _ = gradients(f_y, x)
    return f_xx + f_yy


class MomentumResiduals(NamedTuple):
    mom_x: torch.Tensor
    mom_y: torch.Tensor
    kappa: torch.Tensor


class EnergyResiduals(NamedTuple):
    energy: torch.Tensor  # energy + evaporation latent sink
    src_closure: torch.Tensor  # source(x) minus the evaporation mass closure
    interface_delta: torch.Tensor  # |grad alpha| area density, for inspection


def momentum_residuals(
    model, x: torch.Tensor, groups: dict[str, float], eps: float = KAPPA_EPS
) -> MomentumResiduals:
    """x/y momentum with Hele-Shaw drag and CSF surface tension. Needs ``p``.

    Every coefficient comes from ``groups`` (never a literal).
    """
    re = groups["Re"]
    we = groups["We"]
    c_hs = groups["hele_shaw"]
    alpha = model.alpha(x)
    rho_t = mixture(alpha, groups["rho_ratio"])
    mu_t = mixture(alpha, groups["mu_ratio"])

    a_x, a_y, nx, ny, _ = _interface_normal(alpha, x, eps)
    kappa = -_normal_divergence(nx, ny, x)

    u, v = model.velocity(x)
    u_x, u_y, u_t = gradients(u, x)
    v_x, v_y, v_t = gradients(v, x)
    p_x, p_y, _ = gradients(model.pressure(x), x)
    mom_x = (
        rho_t * (u_t + u * u_x + v * u_y)
        + p_x
        - (1.0 / re) * _laplacian(u, x)
        + c_hs * mu_t * u
        - (1.0 / we) * kappa * a_x
    )
    mom_y = (
        rho_t * (v_t + u * v_x + v * v_y)
        + p_y
        - (1.0 / re) * _laplacian(v, x)
        + c_hs * mu_t * v
        - (1.0 / we) * kappa * a_y
    )
    return MomentumResiduals(mom_x, mom_y, kappa)


def energy_residuals(
    model, x: torch.Tensor, groups: dict[str, float], r_int_star, eps: float = KAPPA_EPS
) -> EnergyResiduals:
    """Energy advection-diffusion with wall heating and the two-way evaporation
    closure. Needs ``T`` (and the ``s`` field for the mass consistency).

    ``r_int_star`` is the non-dimensional interfacial resistance closing the
    evaporation flux -- a trainable inverse unknown supplied by the trainer.
    """
    pe = groups["Pe"]
    q_wall = groups["q_wall_star"]
    rho_ratio = groups["rho_ratio"]

    alpha = model.alpha(x)
    a_x, a_y, _ = gradients(alpha, x)
    delta = torch.sqrt(a_x**2 + a_y**2 + eps**2)  # interfacial area density

    theta = model.temperature(x)  # non-dimensional superheat (T - T_sat)/dT_ref
    # Hardt-Wondra flux; the same flux dilates mass and removes latent heat.
    evap = groups["Ja"] * (theta / r_int_star) * delta
    src_closure = model.source(x) - (rho_ratio - 1.0) * evap

    u, v = model.velocity(x)
    t_x, t_y, t_t = gradients(theta, x)
    energy = (
        t_t
        + u * t_x
        + v * t_y
        - (1.0 / pe) * _laplacian(theta, x)
        - q_wall * (1.0 - alpha)  # wall heating, gated by liquid contact
        + evap  # latent-heat sink at the interface
    )
    return EnergyResiduals(energy, src_closure, delta)


def stage_b_residuals(
    model, x: torch.Tensor, groups: dict[str, float], r_int_star=1.0, eps: float = KAPPA_EPS
) -> StageBResiduals:
    """Convenience combiner of momentum + energy, for the full Stage-B set."""
    mom = momentum_residuals(model, x, groups, eps)
    en = energy_residuals(model, x, groups, r_int_star, eps)
    return StageBResiduals(
        mom.mom_x, mom.mom_y, en.energy, en.src_closure, mom.kappa, en.interface_delta
    )
