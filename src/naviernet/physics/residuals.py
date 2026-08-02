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


def _ctx(c: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
    """Broadcast a dataset's conditioning row ``(1, n_cond)`` to this point batch.

    A residual is evaluated on one dataset's points, so its context is a single
    row expanded to match ``x``'s count -- which lets the same context serve
    collocation, inlet, and wall batches of different sizes. ``None`` passes
    through unchanged, leaving the unconditioned (single-dataset) path untouched.
    """
    return None if c is None else c.expand(x.shape[0], -1)


def interface_indicator(alpha: torch.Tensor) -> torch.Tensor:
    """``4a(1-a)``: unity at the alpha=0.5 interface, decaying to zero in the bulk."""
    return 4.0 * alpha * (1.0 - alpha)


def stage_a_residuals(model, x: torch.Tensor, c: torch.Tensor | None = None) -> StageAResiduals:
    """Evaluate the Stage-A residuals at collocation points ``x``.

    ``c`` is the points' dataset's conditioning row (``None`` for an
    unconditioned single-dataset model).
    """
    cx = _ctx(c, x)
    alpha = model.alpha(x, cx)
    u, v = model.velocity(x, cx)
    source = model.source(x, cx)

    a_x, a_y, a_t = gradients(alpha, x)
    u_x, _, _ = gradients(u, x)
    _, v_y, _ = gradients(v, x)

    return StageAResiduals(
        vof=a_t + u * a_x + v * a_y,
        div=u_x + v_y - source,
        source=source,
        interface_weight=interface_indicator(alpha).detach(),
    )


def source_penalty_sq(residuals: StageAResiduals) -> torch.Tensor:
    """Per-point squared dilatation penalty away from the interface, where it is
    unphysical (shape ``(n, 1)``). The registry's ``src`` collocation term reads this."""
    return ((1.0 - residuals.interface_weight) * residuals.source) ** 2


def source_penalty(residuals: StageAResiduals) -> torch.Tensor:
    """Penalise dilatation away from the interface, where it is unphysical."""
    return source_penalty_sq(residuals).mean()


def boundary_losses(
    model, inlet_x, wall_x, u_inlet: float, c: torch.Tensor | None = None
) -> torch.Tensor:
    """Inlet plug velocity and no-slip side walls (Stage A: velocity only)."""
    u_in, v_in = model.velocity(inlet_x, _ctx(c, inlet_x))
    u_wall, v_wall = model.velocity(wall_x, _ctx(c, wall_x))

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
    model,
    x: torch.Tensor,
    groups: dict[str, float],
    eps: float = KAPPA_EPS,
    c: torch.Tensor | None = None,
) -> MomentumResiduals:
    """x/y momentum with Hele-Shaw drag and CSF surface tension. Needs ``p``.

    Every coefficient comes from ``groups`` (never a literal). ``c`` is the
    points' dataset's conditioning row (``None`` when unconditioned).
    """
    re = groups["Re"]
    we = groups["We"]
    c_hs = groups["hele_shaw"]
    cx = _ctx(c, x)
    alpha = model.alpha(x, cx)
    rho_t = mixture(alpha, groups["rho_ratio"])
    mu_t = mixture(alpha, groups["mu_ratio"])

    a_x, a_y, nx, ny, _ = _interface_normal(alpha, x, eps)
    kappa = -_normal_divergence(nx, ny, x)

    u, v = model.velocity(x, cx)
    u_x, u_y, u_t = gradients(u, x)
    v_x, v_y, v_t = gradients(v, x)
    p_x, p_y, _ = gradients(model.pressure(x, cx), x)
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


class NucleationPulse(NamedTuple):
    """A localized initiation heat pulse at the fixed nucleation cavity.

    The real experiment seeds the bubble with a brief, spatially localized heat pulse at
    a fixed cavity (in addition to the uniform bottom-wall flux). Encoding it gives the
    model the fixed, time-anchored *cause* of the bubble it otherwise lacks -- a
    streamwise Gaussian at ``x_pin`` (width ``sigma``), a step for ``t < t0``, with a
    learnable magnitude ``q_pulse`` (the heater power is unknown -- V is known but its
    resistance is not -- so only the strength is fit; location, timing and width are
    fixed priors). All quantities non-dimensional.
    """

    x_pin: float
    t0: float
    sigma: float
    q_pulse: torch.Tensor  # learnable magnitude (softplus >= 0)


def nucleation_pulse_source(
    x: torch.Tensor, x_pin: float, t0: float, sigma: float, q_pulse
) -> torch.Tensor:
    """The nucleation pulse's per-point heat ``(N, 1)`` (before the liquid ``(1-alpha)``
    gate): a streamwise Gaussian centred at the fixed cavity ``x_pin`` and gated to the
    brief initiation burst ``t < t0``. y-independent (a band across the channel height),
    since the cavity's channel-height position is not measured."""
    streamwise = x[:, 0:1]
    time = x[:, 2:3]
    spatial = torch.exp(-((streamwise - x_pin) ** 2) / (2.0 * sigma**2))
    active = (time < t0).to(x.dtype)
    return q_pulse * spatial * active


def energy_residuals(
    model,
    x: torch.Tensor,
    groups: dict[str, float],
    r_int_star,
    eps: float = KAPPA_EPS,
    c: torch.Tensor | None = None,
    pulse: NucleationPulse | None = None,
) -> EnergyResiduals:
    """Energy advection-diffusion with wall heating and the two-way evaporation
    closure. Needs ``T`` (and the ``s`` field for the mass consistency).

    ``r_int_star`` is the non-dimensional interfacial resistance closing the
    evaporation flux -- a trainable inverse unknown supplied by the trainer.
    ``c`` is the points' dataset's conditioning row (``None`` when unconditioned).
    ``pulse`` adds the localized nucleation heat pulse (:class:`NucleationPulse`); when
    ``None`` (the default) the energy residual is unchanged.
    """
    pe = groups["Pe"]
    q_wall = groups["q_wall_star"]
    rho_ratio = groups["rho_ratio"]

    cx = _ctx(c, x)
    alpha = model.alpha(x, cx)
    a_x, a_y, _ = gradients(alpha, x)
    delta = torch.sqrt(a_x**2 + a_y**2 + eps**2)  # interfacial area density

    theta = model.temperature(x, cx)  # non-dimensional superheat (T - T_sat)/dT_ref
    # Hardt-Wondra flux; the same flux dilates mass and removes latent heat.
    evap = groups["Ja"] * (theta / r_int_star) * delta
    # Mass closure: the free dilatation source ``s`` (which enters continuity as
    # u_x + v_y - s) must equal the volume the phase change creates. That volume is
    # mdot*(1/rho_v - 1/rho_l); in the model's vapour-scaled source the prefactor is
    # the O(1) fraction (1 - rho_v/rho_l), NOT (rho_l/rho_v - 1) ~ rho_ratio, which
    # over-scales the source ~120x -> unphysical interface velocity that collapses
    # alpha (see tests). Detach the flux target so this one-way penalty trains ``s``
    # alone and cannot flatten the interface (delta) or perturb theta to cheat.
    src_closure = model.source(x, cx) - (1.0 - 1.0 / rho_ratio) * evap.detach()

    # Wall heating gated by liquid contact: the uniform bottom-wall flux plus, when
    # enabled, the localized nucleation pulse that seeds the bubble at the fixed cavity.
    wall_heat = q_wall
    if pulse is not None:
        wall_heat = wall_heat + nucleation_pulse_source(
            x, pulse.x_pin, pulse.t0, pulse.sigma, pulse.q_pulse
        )

    u, v = model.velocity(x, cx)
    t_x, t_y, t_t = gradients(theta, x)
    energy = (
        t_t
        + u * t_x
        + v * t_y
        - (1.0 / pe) * _laplacian(theta, x)
        - wall_heat * (1.0 - alpha)  # heating, gated by liquid contact
        + evap  # latent-heat sink at the interface
    )
    return EnergyResiduals(energy, src_closure, delta)


# --- R4: sharp-interface conditions on the explicit front --------------------


def darcy_residuals(
    model, x: torch.Tensor, groups: dict[str, float], c: torch.Tensor | None = None
) -> MomentumResiduals:
    """Depth-averaged (Hele-Shaw) momentum, the leading-order balance here::

        grad p = -C_HS mu*(alpha) u

    paired with the unchanged continuity ``u_x + v_y = s``.

    Why this and not :func:`momentum_residuals`. Darcy is the depth-averaged
    limit of the *3-D* problem, not a special case of the 2-D one: the dominant
    force in a 198 um channel is the wall shear in the GAP direction, which a
    2-D (x, y) formulation does not contain at all -- which is why the 2-D
    residual has to carry ``hele_shaw`` as a bolted-on stand-in for it. Measured
    on the R3 baseline, that formulation's in-plane inertia ran at RMS 0.34
    against the drag's 0.05, so the optimiser spent its effort on terms that are
    O(eps) in this regime (Ca = 0.011, Bo = 0.073, Re_in = 22) while the actual
    leading balance sat in the noise.

    No surface-tension body force. In a sharp-interface formulation capillarity
    is a BOUNDARY CONDITION (:func:`laplace_jump_residual`), not a volumetric
    term; the CSF force exists only because a diffuse interface has no boundary
    to put it on. Removing it here is what stops a free ``p`` from absorbing it.

    Returns the same shape as :func:`momentum_residuals` so both momentum-family
    equations read identically to the registry, with ``kappa`` left at zero:
    there is no curvature in this balance.
    """
    cx = _ctx(c, x)
    drag = groups["hele_shaw"] * mixture(model.alpha(x, cx), groups["mu_ratio"])
    u, v = model.velocity(x, cx)
    p_x, p_y, _ = gradients(model.pressure(x, cx), x)
    return MomentumResiduals(p_x + drag * u, p_y + drag * v, torch.zeros_like(u))


# Bretherton's front-meniscus correction, 1.29 (3 Ca)^{2/3} = 2.68 Ca^{2/3}
# (Bretherton 1961): the dynamic thickening of the capillary pressure across an
# advancing meniscus that has laid down a lubrication film behind it. A fixed
# physical coefficient, not a tunable.
BRETHERTON_COEFF = 1.29 * 3.0 ** (2.0 / 3.0)


def gap_curvature(normal_speed: torch.Tensor, groups: dict[str, float]) -> torch.Tensor:
    """Out-of-plane (gap-direction) interface curvature, ``(N, 1)``::

        kappa_perp = (2 / H*) (1 + 2.68 Ca_local^{2/3})

    A depth-averaged model has no z direction, so this curvature cannot be
    computed from the in-plane shape -- it has to be supplied. It matters twice
    over: it is the LARGER principal curvature here (``2/H* = 4`` against an
    in-plane O(1)), and, through the local capillary number
    ``Ca_local = Ca * v_n``, it is the only place the front's own SPEED enters
    the capillary pressure.

    That speed dependence is the mechanism the whole sharp-interface change
    exists to restore: a fast-advancing nose carries more capillary pressure
    than a slow mid-body, so vapour is driven forward and the middle thins. With
    a speed-independent capillary pressure every station along the bubble is
    interchangeable and no neck can be selected.

    Only an ADVANCING front deposits a film, so a receding section takes the
    static ``2/H*`` (and a negative capillary number never reaches the 2/3
    power).
    """
    capillary = (groups["Ca"] * normal_speed).clamp(min=0.0)
    return (2.0 / groups["H_star"]) * (1.0 + BRETHERTON_COEFF * capillary ** (2.0 / 3.0))


def laplace_jump_residual(model, front, groups: dict[str, float]) -> torch.Tensor:
    """Young-Laplace across the explicit interface, per front sample ``(N, 1)``::

        p_v(t) - p_liq(Gamma) = (1/We) (kappa_par + kappa_perp)

    The vapour is higher-pressure inside a convex bubble, which is the sign the
    diffuse ``curvature`` above already uses (a compact vapour region reads
    positive). ``p_v`` is space-independent, so this one condition ties the
    liquid pressure at every point of the front to the LOCAL total curvature --
    the constraint that makes shape a consequence of forces.

    Why here and not in the bulk: with a free ``p`` field, the CSF term in
    :func:`momentum_residuals` can be absorbed by the pressure up to its curl, so
    the bulk residual constrains ``p``, not the interface. Evaluated ON the front,
    there is nothing left to absorb it.
    """
    t = front.points[:, 2:3]
    kappa = front.kappa_par + gap_curvature(front.normal_speed, groups)
    return model.p_vapor(t) - model.pressure(front.points) - kappa / groups["We"]


def stage_b_residuals(
    model,
    x: torch.Tensor,
    groups: dict[str, float],
    r_int_star=1.0,
    eps: float = KAPPA_EPS,
    c: torch.Tensor | None = None,
) -> StageBResiduals:
    """Convenience combiner of momentum + energy, for the full Stage-B set."""
    mom = momentum_residuals(model, x, groups, eps, c)
    en = energy_residuals(model, x, groups, r_int_star, eps, c)
    return StageBResiduals(
        mom.mom_x, mom.mom_y, en.energy, en.src_closure, mom.kappa, en.interface_delta
    )
