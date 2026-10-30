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

import math
from contextlib import contextmanager
from typing import NamedTuple

import torch

from naviernet.physics import depth
from naviernet.physics.film import film_surface_pressure
from naviernet.physics.heating import wall_flux_ratio


class StageAResiduals(NamedTuple):
    """Residual fields evaluated at a batch of collocation points."""

    vof: torch.Tensor  # volume-fraction transport residual
    div: torch.Tensor  # continuity residual, net of the dilatation source
    source: torch.Tensor  # the inferred source itself, for penalisation
    interface_weight: torch.Tensor  # 1 on the interface, 0 in the bulk (detached)


# While set (sharing_gradients), each (field, points) derivative is taken once
# and shared: a training step's equations differentiate the same velocity,
# temperature and pressure at the same points, each on its own before.
_GRADIENT_CACHE: dict | None = None


@contextmanager
def sharing_gradients():
    """Within the block, :func:`gradients` returns the derivative it already
    took of the same field at the same points instead of taking it again. The
    value is the same either way; the step's graph, and its backward pass, is
    not built twice. Keyed on the tensors themselves, held for the block."""
    global _GRADIENT_CACHE
    _GRADIENT_CACHE = {}
    try:
        yield
    finally:
        _GRADIENT_CACHE = None


def gradients(f: torch.Tensor, x: torch.Tensor):
    """Return ``(f_x, f_y, f_t)`` for a scalar field ``f`` evaluated at ``x``.

    ``x`` must have ``requires_grad=True`` and columns ordered ``(x, y, t)``.
    """
    cache = _GRADIENT_CACHE
    if cache is not None:
        hit = cache.get((id(f), id(x)))
        if hit is not None and hit[0] is f and hit[1] is x:
            return hit[2]
    grad = torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]
    parts = grad[:, 0:1], grad[:, 1:2], grad[:, 2:3]
    if cache is not None:
        cache[(id(f), id(x))] = (f, x, parts)
    return parts


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

    vof = a_t + u * a_x + v * a_y
    # Under model.depth_bypass the liquid moves with eta V_n at the outline
    # (physics.depth); the rest of the outline's advance is liquid it passes
    # over, which the advection on the liquid side must not count. V_n |grad
    # alpha| is alpha_t exactly, so the sink is (1 - eta) alpha_t.
    factor = depth.at_points(model, x)
    if factor is not None:
        vof = factor * a_t + u * a_x + v * a_y
    div = u_x + v_y - source
    if getattr(model, "exact_liquid", False):
        # The liquid is divergence-free by construction: there the residual is
        # only the source's tail, and scoring it would pull the interface flux
        # to zero. Continuity is the vapour's alone.
        liquid = model.liquid_side(x).unsqueeze(1)
        div = torch.where(liquid, torch.zeros_like(div), div)
    return StageAResiduals(
        vof=vof,
        div=div,
        source=source,
        interface_weight=interface_indicator(alpha).detach(),
    )


def source_penalty_sq(residuals: StageAResiduals) -> torch.Tensor:
    """Per-point squared dilatation penalty away from the interface, where it is
    unphysical (shape ``(n, 1)``). The registry's ``src`` collocation term reads this."""
    return ((1.0 - residuals.interface_weight) * residuals.source) ** 2


def footprint_source_penalty_sq(
    residuals: StageAResiduals, vapour: torch.Tensor
) -> torch.Tensor:
    """Per-point squared dilatation penalty OUTSIDE the bubble, ``(n, 1)``.

    :func:`source_penalty_sq`'s liquid-film sibling: under ``model.liquid_film``
    the mass legitimately appears across the whole footprint the bubble covers
    -- the films evaporate there -- so the interface-band rule would penalise
    exactly where the film's own closure puts the source. ``vapour`` is the
    detached volume fraction; only the liquid outside the bubble stays forbidden.
    """
    return ((1.0 - vapour) * residuals.source) ** 2


def source_penalty(residuals: StageAResiduals) -> torch.Tensor:
    """Penalise dilatation away from the interface, where it is unphysical."""
    return source_penalty_sq(residuals).mean()


def developed_inlet_profile(
    y: torch.Tensor, y_lo: float, y_hi: float, mean: float, groups: dict[str, float]
) -> torch.Tensor:
    """The fully developed depth-averaged Brinkman profile across the width,
    ``(N, 1)``, with mean ``mean``.

    In a straight channel the law ``-dp/dx = C_HS u - (1/Re) u''`` with no slip
    at the side walls gives ``u ~ 1 - cosh(k y) / cosh(k w/2)`` about the
    centreline, ``k = sqrt(C_HS Re) = sqrt(12) L/h``: flat across the middle and
    falling to zero over ``h / sqrt(12)`` (43 um at a 150 um depth). Its mean
    over the width is ``1 - tanh(k w/2) / (k w/2)`` of the peak.
    """
    k = math.sqrt(groups["hele_shaw"] * groups["Re"])
    half = 0.5 * (y_hi - y_lo)
    shape = 1.0 - torch.cosh(k * (y - 0.5 * (y_lo + y_hi))) / math.cosh(k * half)
    return mean * shape / (1.0 - math.tanh(k * half) / (k * half))


def boundary_losses(
    model,
    inlet_x,
    wall_x,
    u_inlet: float | torch.Tensor,
    c: torch.Tensor | None = None,
    theta_in: torch.Tensor | None = None,
    wall_condition: str = "no_slip",
    outlet_x: torch.Tensor | None = None,
    outlet_pressure: float | None = None,
) -> torch.Tensor:
    """Inlet plug velocity, the side-wall condition, and the outlet condition.

    ``wall_condition`` is ``"no_slip"`` (both components vanish at the wall) or
    ``"no_penetration"`` (only the wall-normal ``v``; the tangential velocity is
    free, which is all a first-order Darcy law can honour). ``outlet_x``, when
    given, adds :func:`outlet_loss` at those points, with the liquid pressure
    held at ``outlet_pressure`` there when that is given. ``u_inlet`` is the
    plug's speed, or one target per inlet point (the developed profile).

    ``theta_in`` adds the inlet temperature condition. It is passed only when the
    superheat is depletable: with the default parameterization ``theta_in`` is
    already the field's floor and anchoring it here would say nothing, but once
    the floor is saturation the liquid arriving at the inlet still has to arrive
    at its inlet superheat.
    """
    u_in, v_in = model.velocity(inlet_x, _ctx(c, inlet_x))
    u_wall, v_wall = model.velocity(wall_x, _ctx(c, wall_x))

    inlet = ((u_in - u_inlet) ** 2).mean() + (v_in**2).mean()
    wall = (u_wall**2).mean() + (v_wall**2).mean()
    if wall_condition == "no_penetration":
        wall = (v_wall**2).mean()
    if theta_in is not None:
        inlet = inlet + ((model.temperature(inlet_x, _ctx(c, inlet_x)) - theta_in) ** 2).mean()
    total = inlet + wall
    if outlet_x is not None:
        total = total + outlet_loss(model, outlet_x, c, outlet_pressure)
    return total


def outlet_loss(
    model,
    outlet_x: torch.Tensor,
    c: torch.Tensor | None = None,
    pressure: float | None = None,
) -> torch.Tensor:
    """Zero streamwise velocity gradient at the outlet, ``mean(u_x^2 + v_x^2)``,
    and the liquid pressure there held at ``pressure`` when given.

    The channel continues beyond the field of view, so the flow leaves the imaged
    region as it arrived there. With the pressure given the outflow is
    stress-free at that pressure (the viscous normal stress is zero with
    ``u_x``); without it the pressure's level is left to the vapour pressure.
    ``outlet_x`` must carry ``requires_grad`` (the sampler sets it).
    """
    cx = _ctx(c, outlet_x)
    total = outlet_x.new_zeros(())
    # Under model.exact_continuity the liquid leaves as the potential flow
    # continues past the field of view, not as a developed one (2% uneven at
    # frame 9, more as the nose nears the outlet): only the pressure is held.
    if not getattr(model, "exact_liquid", False):
        u, v = model.velocity(outlet_x, cx)
        u_x, _, _ = gradients(u, outlet_x)
        v_x, _, _ = gradients(v, outlet_x)
        total = (u_x**2).mean() + (v_x**2).mean()
    if pressure is not None:
        total = total + ((liquid_pressure(model, outlet_x, cx) - pressure) ** 2).mean()
    return total


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


def confinement_drag(alpha: torch.Tensor, groups: dict[str, float]) -> torch.Tensor:
    """The depth-averaged Hele-Shaw drag coefficient, ``C_HS mu*(alpha)``.

    One definition shared by both momentum formulations: it is the same physical
    closure whether it appears as a term in the 2-D residual or as the whole of
    the Darcy balance, and a change to one that missed the other would be a
    silent inconsistency between the two paths.
    """
    return groups["hele_shaw"] * mixture(alpha, groups["mu_ratio"])


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
    cx = _ctx(c, x)
    alpha = model.alpha(x, cx)
    rho_t = mixture(alpha, groups["rho_ratio"])
    drag = confinement_drag(alpha, groups)

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
        + drag * u
        - (1.0 / we) * kappa * a_x
    )
    mom_y = (
        rho_t * (v_t + u * v_x + v * v_y)
        + p_y
        - (1.0 / re) * _laplacian(v, x)
        + drag * v
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
    two_way_closure: bool = False,
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
    # Which way the closure can push. Detaching the whole target makes it
    # ONE-WAY: it pulls the source down to the drive and can never raise the
    # drive, so no weight on it will ever move theta. With `two_way_closure` only
    # the interfacial area is detached, so the closure may raise the temperature
    # to justify the source it observes -- but still cannot flatten the interface
    # to satisfy itself, which is what the detach was there to prevent.
    target = (
        groups["Ja"] * (theta / r_int_star) * delta.detach()
        if two_way_closure
        else evap.detach()
    )
    src_closure = model.source(x, cx) - (1.0 - 1.0 / rho_ratio) * target

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


# --- WP4: the sharp thermal model -------------------------------------------
#
# The interface is AT saturation, so the latent heat is a boundary condition
# rather than a volumetric sink: the liquid loses heat by conduction into the
# front, and that flux, g = d theta/dn on the liquid side, is the one mass flux
# that creates vapour volume (``groups["mass_flux_to_source"] * g`` per unit
# front length) and moves the interface relative to the liquid
# (``groups["mass_flux_to_speed"] * g``). Under the gated parameterisation theta
# is exactly zero on the front and in the vapour, so no interface or vapour
# penalty is needed; the liquid energy equation below is weighted by the liquid
# fraction so it is scored where it holds.


def energy_liquid_residual(
    model,
    x: torch.Tensor,
    groups: dict[str, float],
    c: torch.Tensor | None = None,
    one_way: bool = False,
) -> torch.Tensor:
    """``(1 - alpha) [theta_t + u.grad theta - lap(theta)/Pe - q* (1 - alpha)]``.

    The heating ``q*`` is the series' wall flux, gated by liquid contact: under
    the bubble the same flux evaporates the film instead (the footprint source
    on the model), so one wall flux is split by phase. The latent sink is absent
    because the interface temperature is the boundary condition that carries it.

    ``one_way`` detaches the velocity in the advection term, so this equation
    trains the temperature and never the flow. Physically the coupling is one
    way here: the liquid carries the heat, and nothing in the model lets the
    temperature move the liquid (constant properties, no buoyancy); its only
    route back is the evaporation source, which continuity reads. Without the
    detach the equation's error reaches the velocity network through
    ``u.grad(theta)``, and measured on the trusted recipe it pulled the flow
    ahead of the bubble 2 to 30 times harder than the momentum equation did,
    bending it into the lopsided swirl the mirror test failed on.
    """
    return energy_liquid_balance(model, x, groups, c, one_way)[0]


def energy_liquid_balance(
    model,
    x: torch.Tensor,
    groups: dict[str, float],
    c: torch.Tensor | None = None,
    one_way: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`energy_liquid_residual` with the size of its terms: the liquid-
    weighted sum of their magnitudes ``|theta_t| + |u.grad theta| + |lap/Pe| +
    |q*|``, detached, the scale the relative error is measured against."""
    cx = _ctx(c, x)
    alpha = model.alpha(x, cx)
    theta = model.temperature(x, cx)
    u, v = model.velocity(x, cx)
    if one_way:
        u, v = u.detach(), v.detach()
    t_x, t_y, t_t = gradients(theta, x)
    heat = groups["q_wall_star"] * wall_flux_ratio(x, groups) * (1.0 - alpha)
    advection = u * t_x + v * t_y
    diffusion = _laplacian(theta, x) / groups["Pe"]
    liquid = 1.0 - alpha.detach()
    residual = liquid * (t_t + advection - diffusion - heat)
    scale = liquid * (t_t.abs() + advection.abs() + diffusion.abs() + heat.abs())
    return residual, scale.detach()


class ThermalBatches(NamedTuple):
    """The sampled points the temperature conditions are scored on: the inlet
    line, the first instant, and the side walls (a leaf, for the adiabatic
    gradient)."""

    inlet: torch.Tensor
    initial: torch.Tensor
    walls: torch.Tensor


def thermal_boundary_losses(
    model, groups: dict[str, float], batches: ThermalBatches, c: torch.Tensor | None = None
) -> torch.Tensor:
    """The inlet temperature, the initial liquid temperature (where there is
    liquid, and farther from the frame-1 outline than the layer the bubble
    already has, ``init_layer_star``), and the side-wall condition: adiabatic
    (``theta_y = 0``) unless the series gives a wall temperature."""
    inlet = (
        (model.temperature(batches.inlet, _ctx(c, batches.inlet)) - groups["theta_inlet"]) ** 2
    ).mean()
    total = inlet
    if "theta_init" in groups:
        cx = _ctx(c, batches.initial)
        liquid = 1.0 - model.alpha(batches.initial, cx).detach()
        layer = groups.get("init_layer_star", 0.0)
        if layer > 0.0:
            # Inside the layer the heat equation decides; the distance reads the
            # (frozen) geometry and carries no gradient.
            distance = model._liquid_distance(batches.initial, cx).detach()
            liquid = liquid * (distance >= layer).to(liquid.dtype)
        total = (
            total
            + (
                liquid * (model.temperature(batches.initial, cx) - groups["theta_init"]) ** 2
            ).mean()
        )
    walls = batches.walls.detach().clone().requires_grad_(True)
    theta_wall = model.temperature(walls, _ctx(c, walls))
    if "theta_wall" in groups:
        return total + ((theta_wall - groups["theta_wall"]) ** 2).mean()
    _, theta_y, _ = gradients(theta_wall, walls)
    return total + (theta_y**2).mean()


def evap_flux_closure(model, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
    """``s_net - s_thermal``: the free source must be the interface flux spread
    over the band. Nothing is detached, so the flux can also learn from the
    growth the source carries (the two-way coupling the legacy closure lacked)."""
    cx = _ctx(c, x)
    return model.nets["s"](x, cx) - model.thermal_source(x, cx)


def interface_mass_flux(model, front, groups: dict[str, float]) -> torch.Tensor:
    """The liquid-side kinematic correction on the front, ``(N, 1)``:
    ``mass_flux_to_speed * g``. O(1e-3) of the front speed here, kept so the
    kinematic condition and the source agree about the same evaporation."""
    return groups["mass_flux_to_speed"] * model.interface_flux_at(front)


# --- R4: sharp-interface conditions on the explicit front --------------------


def liquid_pressure(model, x: torch.Tensor, cx: torch.Tensor | None) -> torch.Tensor:
    """The liquid-side pressure the momentum law and the jump condition read: the
    liquid branch under the two-phase representation, the one field otherwise.
    A model without the accessor (a test stand-in) is read through ``pressure``."""
    branch = getattr(model, "liquid_pressure", None)
    return branch(x, cx) if branch is not None else model.pressure(x, cx)


def liquid_weight(model, alpha: torch.Tensor) -> torch.Tensor | float:
    """``1 - alpha`` (detached) under the two-phase pressure, so the momentum law is
    scored where the liquid pressure is defined; 1 everywhere otherwise."""
    if getattr(model, "pressure_representation", "single") == "two_phase":
        return 1.0 - alpha.detach()
    return 1.0


def vapour_potential_residual(
    model, x: torch.Tensor, c: torch.Tensor | None = None
) -> torch.Tensor:
    """``alpha (v_x - u_y)``: the vapour velocity is curl-free, ``(N, 1)``.

    Under the two-phase pressure the vapour pressure is uniform, so the Darcy
    law inside the bubble says nothing about the vapour velocity (its drag is
    1/37 of the liquid's) and leaves it free -- the null space the audit found.
    The same law also says the velocity is a gradient; imposing that directly,
    with the kinematic condition on the front, makes the vapour flow the unique
    potential flow. Weighted by the (detached) indicator, so it acts in the
    vapour and fades through the interface band.
    """
    cx = _ctx(c, x)
    u, v = model.velocity(x, cx)
    _, u_y, _ = gradients(u, x)
    v_x, _, _ = gradients(v, x)
    weight = model.alpha(x, cx).detach()
    cfg = getattr(model, "cfg", None)
    region = (
        getattr(cfg.model, "vapour_potential_region", "band") if cfg is not None else "band"
    )
    if str(region) == "vapour":
        weight = (weight > 0.5).to(weight.dtype)
    return weight * (v_x - u_y)


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
    residual has to carry ``hele_shaw`` as a bolted-on stand-in for it.

    CAUTION on the original justification, which was wrong. It read: "in-plane
    inertia ran at RMS 0.34 against the drag's 0.05, so the optimiser spent its
    effort on terms that are O(eps)". Those are RESIDUAL norms -- they measure how
    badly each term was SATISFIED, not how large it is. A large inertia residual
    is evidence the term was poorly constrained, not evidence it is negligible,
    so nothing in that measurement supported dropping it.

    The defensible criterion is the ratio of convective inertia to Darcy drag,
    ``Re_b (b/L) / 12`` with ``Re_b = rho U b / mu``. At our numbers (Re_Dh = 22,
    b = 150 um) that is ~0.21 over a 1 mm bubble and ~0.7 near the nose, where L
    collapses -- small, but not the O(eps) the docstring claimed. Unsteady inertia
    is the larger exposure: Darcy assumes the gap-wise profile is quasi-steady,
    which needs sqrt(nu T) >> b/2, and that fails for T below ~20 ms. Millisecond
    growth is inside the violated regime.

    Darcy is still the right LEADING order here. The point is that it is a leading
    order with a known, quantified error, not an exact statement -- and the places
    the error concentrates (the nose, the sidewalls, a pinching neck) are exactly
    the features this model fails to reproduce.

    No surface-tension body force. In a sharp-interface formulation capillarity
    is a BOUNDARY CONDITION (:func:`laplace_jump_residual`), not a volumetric
    term; the CSF force exists only because a diffuse interface has no boundary
    to put it on. Removing it here is what stops a free ``p`` from absorbing it.

    Returns the same shape as :func:`momentum_residuals` so both momentum-family
    equations read identically to the registry, with ``kappa`` left at zero:
    there is no curvature in this balance.
    """
    cx = _ctx(c, x)
    alpha = model.alpha(x, cx)
    drag = confinement_drag(alpha, groups)
    weight = liquid_weight(model, alpha)
    u, v = model.velocity(x, cx)
    p_x, p_y, _ = gradients(liquid_pressure(model, x, cx), x)
    return MomentumResiduals(
        weight * (p_x + drag * u), weight * (p_y + drag * v), torch.zeros_like(u)
    )


# The convective term of the depth-averaged momentum carries the mean of u^2
# over the depth over the square of the mean: 6/5 for the parabolic profile the
# Hele-Shaw drag already assumes (Gondret & Rabaud 1997, Phys. Fluids 9, 3267).
DEPTH_PROFILE_CONVECTION = 6.0 / 5.0


def brinkman_residuals(
    model,
    x: torch.Tensor,
    groups: dict[str, float],
    c: torch.Tensor | None = None,
    inertia: bool = False,
) -> MomentumResiduals:
    """Depth-averaged momentum with the Brinkman in-plane viscous term::

        grad p = -C_HS mu*(alpha) u + (mu*(alpha) / Re) lap(u)

    :func:`darcy_residuals` is first order in ``u`` and can honour only a
    no-penetration condition at a side wall. The viscous term restores the O(h)
    wall layer in which the tangential velocity falls to zero (Brinkman 1949; Zeng,
    Yortsos & Salin 2003 for Hele-Shaw cells), so no-slip side walls become
    admissible. Away from the walls, where ``lap(u)`` is small, it is Darcy.
    Same return shape as the other momentum families; ``kappa`` stays zero.

    ``inertia`` (``model.momentum_inertia``) adds the depth-averaged inertia
    ``rho~(alpha) (u_t + 6/5 u.grad u)`` on the left, making the law the
    depth-averaged Navier-Stokes of a Hele-Shaw cell. Pressure is on
    ``rho_l U^2`` and time on ``L/U``, so the coefficient is one in the liquid.
    The time derivative is the model's own (:meth:`BubblePINN.velocity_rate`):
    under exact continuity the liquid's flow is solved per instant, so autograd
    in ``t`` would miss all of it but the stream net's part.
    No surface-tension body force: under the sharp interface the Young-Laplace
    jump on the front carries it, and the liquid pressure is smooth up to it.
    """
    cx = _ctx(c, x)
    alpha = _momentum_phase(model, x, cx)
    drag = confinement_drag(alpha, groups)
    viscosity = mixture(alpha, groups["mu_ratio"]) / groups["Re"]
    weight = liquid_weight(model, alpha)
    u, v = model.velocity(x, cx)
    p_x, p_y, _ = gradients(liquid_pressure(model, x, cx), x)
    mom_x = p_x + drag * u - viscosity * _laplacian(u, x)
    mom_y = p_y + drag * v - viscosity * _laplacian(v, x)
    if inertia:
        u_x, u_y, u_t = gradients(u, x)
        v_x, v_y, v_t = gradients(v, x)
        rate = getattr(model, "velocity_rate", None)
        if rate is not None:
            u_t, v_t = rate(x, u_t, v_t)
        density = mixture(alpha, groups["rho_ratio"])
        convect = DEPTH_PROFILE_CONVECTION
        mom_x = mom_x + density * (u_t + convect * (u * u_x + v * u_y))
        mom_y = mom_y + density * (v_t + convect * (u * v_x + v * v_y))
    return MomentumResiduals(weight * mom_x, weight * mom_y, torch.zeros_like(u))


def _momentum_phase(model, x: torch.Tensor, cx: torch.Tensor | None) -> torch.Tensor:
    """The vapour fraction the momentum law takes its properties and its weight
    from: the smoothed ``alpha``, or under ``model.momentum_phases=sharp`` the
    explicit front's own side (0 in the liquid, 1 in the vapour).

    The smoothed fraction spans 15 um either side of the front, so across a
    34 um gap beside the head (0.5 at the edge, 0.25 mid-gap, 0.1 at the wall)
    it understates the liquid's friction and viscosity by up to half, and
    scores the law at up to half weight where the liquid flows backward."""
    cfg = getattr(model, "cfg", None)
    phases = getattr(cfg.model, "momentum_phases", "mixture") if cfg is not None else "mixture"
    if str(phases) != "sharp":
        return model.alpha(x, cx)
    return (~model.liquid_side(x)).to(x.dtype)[:, None]


def interface_shear_residual(
    model, front, groups: dict[str, float], c: torch.Tensor | None = None
) -> torch.Tensor:
    """The tangential viscous stress the liquid puts on the front, per sample
    ``(N, 1)``, on the pressure scale ``rho_l U^2``::

        (1/Re) t . (grad u + grad u^T) . n = 0

    A clean interface against a vapour 37 times less viscous carries next to no
    shear. The Brinkman law is second order in ``u``, so the front needs this
    tangential condition beside the kinematic (normal) one; without it the
    velocity can slide along the front at any speed, and its in-plane viscous
    term then drives the liquid beside the front instead of the pressure
    (exact4-s0: the liquid on the edge of the gap at -590 mm/s, its pressure
    flat). Read on the liquid side, where the front's samples evaluate. The
    points are detached: this is a condition on the flow, not on the outline.
    """
    x = front.points.detach().requires_grad_(True)
    u, v = model.velocity(x, _ctx(c, x))
    u_x, u_y, _ = gradients(u, x)
    v_x, v_y, _ = gradients(v, x)
    n_x, n_y = front.normal[:, 0:1].detach(), front.normal[:, 1:2].detach()
    t_x, t_y = -n_y, n_x
    strain = (
        2.0 * n_x * t_x * u_x + (n_x * t_y + n_y * t_x) * (u_y + v_x) + 2.0 * n_y * t_y * v_y
    )
    return strain / groups["Re"]


# Dynamic correction to the gap-direction capillary pressure across an advancing
# meniscus that has laid down a lubrication film behind it.
#
# 1.79 (3 Ca)^{2/3} = 3.72 Ca^{2/3}, the coefficient Park & Homsy (1984, JFM 139,
# 291) derive for the PRESSURE JUMP. This previously carried 1.29, which is close
# to Bretherton's FILM-THICKNESS coefficient (1.34) rather than his pressure one,
# and it made the correction 1.39x too small -- +13.0% instead of +18.1% at our
# Ca = 0.011. Two different constants belong to two different quantities; the
# jump condition needs the jump one.
BRETHERTON_COEFF = 1.79 * 3.0 ** (2.0 / 3.0)

# The RECEDING branch of the same correction (`model.receding_cap`): a rear
# meniscus is FLATTER than static, kappa R = 1 + beta (3 Ca)^{2/3} with
# beta_rear = -1.13 (Bretherton 1961; Balestra, Zhu & Gallaire 2018; the
# receding coefficient as quoted by Lu, Glasner, Bertozzi & Kim 2007). Without
# it a receding section is clamped TO static -- an O(10%) overstatement at our
# Ca on exactly the receding root body.
BRETHERTON_RECEDING_COEFF = 1.13 * 3.0 ** (2.0 / 3.0)

# Park & Homsy's weight on the IN-PLANE curvature in the same depth-averaged jump
# condition: `dp = gamma[(2/b)(1 + 3.8 Ca^{2/3}) + (pi/4) kappa_par]`. It accounts
# for how the principal radii vary across the gap for a perfectly wetting liquid,
# which ours are (contact angles ~5-10 deg). Omitting it over-weighted the
# in-plane term by 4/pi = 1.27x -- and that term is the one that HEALS a waist in
# a Darcy flow, so the error pushed directly against the feature we want.
IN_PLANE_WEIGHT = math.pi / 4.0


def gap_curvature(
    normal_speed: torch.Tensor, groups: dict[str, float], receding: bool = False
) -> torch.Tensor:
    """Out-of-plane (gap-direction) interface curvature, ``(N, 1)``::

        kappa_perp = (2 / H*) (1 + 3.72 Ca_local^{2/3})

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
    power) -- unless ``receding`` (`model.receding_cap`) admits the rear
    branch, where a receding meniscus reads FLATTER than static with the
    receding coefficient, floored at zero: a curvature is a shape property and
    cannot go negative however fast the section retreats.
    """
    capillary = (groups["Ca"] * normal_speed).clamp(min=0.0)
    advancing = (2.0 / groups["H_star"]) * (1.0 + BRETHERTON_COEFF * capillary ** (2.0 / 3.0))
    if not receding:
        return advancing
    rear = (groups["Ca"] * (-normal_speed)).clamp(min=0.0)
    reduction = (2.0 / groups["H_star"]) * BRETHERTON_RECEDING_COEFF * rear ** (2.0 / 3.0)
    return (advancing - reduction).clamp(min=0.0)


def jump_gap_curvature(model, front, groups: dict[str, float]) -> torch.Tensor:
    """The gap-direction curvature the jump condition sees for THIS model,
    ``(N, 1)``: the free-meniscus branch (receding correction included when
    trained), blended toward the ATTACHED value over the patch footprint::

        kappa_perp = (1 - a) * kappa_perp_free + a * (1 + cos theta_app)/H*

    Over a dry contact patch the contact-angle cosines ADD across the gap
    (Park & Homsy's 2/H* is the both-plates-wetted limit; Lu et al. 2007 the
    both-contacted one; our one-patch-one-film hybrid is bracketed between
    them, so the free side keeps its Bretherton correction). Lower capillary
    pressure over the patch -> lower demanded curvature -> a blunter root:
    the measured sign, delivered through the physics rather than asserted.

    With both flags off this IS :func:`gap_curvature` -- the single definition
    the residual, the diagnostics and the shape solve all read, so a trained
    attachment run is scored by the condition it actually trained on.
    """
    kappa = gap_curvature(front.normal_speed, groups, getattr(model, "receding_cap", False))
    if not getattr(model, "root_attachment", False):
        return kappa
    attached_fraction = model.attachment_fraction(front.points)
    theta = torch.deg2rad(model.theta_app_deg(groups))
    attached = (1.0 + torch.cos(theta)) / groups["H_star"]
    return kappa + attached_fraction * (attached - kappa)


def jump_total_curvature(model, front, groups: dict[str, float]) -> torch.Tensor:
    """:func:`total_curvature`, with the model's own gap-term treatment."""
    return IN_PLANE_WEIGHT * front.kappa_par + jump_gap_curvature(model, front, groups)


def kinematic_residual(
    model,
    front,
    c: torch.Tensor | None = None,
    mass_flux: torch.Tensor | None = None,
    bypass: torch.Tensor | None = None,
) -> torch.Tensor:
    """The kinematic condition on the explicit interface, per sample ``(N, 1)``::

        v_n = u . n  (+ the liquid-side mass-flux correction, when given)

    The front advances at the normal component of the local fluid velocity. Only
    the normal component can move it: a tangential slip along the interface
    transports nothing across it, which is why this is projected rather than
    compared component-wise. Under the legacy treatment phase change contributes
    no extra interface velocity (the source lives OFF the interface, the same
    reason the VOF residual is a pure advection); the sharp thermal model passes
    :func:`interface_mass_flux` so the front also recedes from the liquid at the
    rate the evaporation flux demands.

    The front-sampled counterpart of the ``vof`` collocation residual, and the
    reason it is worth having twice: ``vof`` is evaluated at bulk points, where
    ``grad alpha`` is ~0 and the equation is satisfied by doing nothing. Here
    every sample is on the interface, so every sample constrains it.

    ``bypass`` (:mod:`naviernet.physics.depth`) scales the front's speed to the
    share of its displacement the depth-averaged liquid carries.
    """
    u, v = model.velocity(front.points, _ctx(c, front.points))
    advance = u * front.normal[:, 0:1] + v * front.normal[:, 1:2]
    if mass_flux is not None:
        advance = advance + mass_flux
    speed = front.normal_speed if bypass is None else front.normal_speed * bypass
    return speed - advance


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
    kappa = jump_total_curvature(model, front, groups)
    # Deliberately NOT written as `(pressure_implied_curvature(...) - kappa)/We`,
    # though that is the same equation: the two differ in float32 (a multiply by
    # We and a divide back do not cancel), and this one TRAINS. A shared helper
    # is not worth perturbing a residual's arithmetic; the test below pins the
    # two readings to each other instead.
    liquid = _liquid_pressure(model, front, groups)
    return model.p_vapor(t) - liquid - kappa / groups["We"]


def _liquid_pressure(model, front, groups: dict[str, float]) -> torch.Tensor:
    """The liquid pressure the meniscus actually faces.

    On the body that is the Bretherton film, not the bulk the depth-averaged
    ``p`` represents (see BubblePINN.film_offset); the offset is zero on the caps
    and zero entirely when the feature is off.

    Under ``model.liquid_film`` the film surface's own capillary pressure
    perturbation is added on the body: where the film locally thins, its
    pressure rises toward the vapour's, and that x-VARYING departure -- which
    the scalar ``film_offset`` structurally cannot carry -- is what lets the
    jump condition select a station. A detached forcing -- the geometry
    answers it through the curvature side of the jump.
    """
    liquid = liquid_pressure(model, front.points, None) + model.film_offset(front.on_cap)
    if getattr(model, "liquid_film", False):
        liquid = liquid + film_surface_pressure(model, front, groups)
    return liquid


def pressure_implied_curvature(
    model, front, groups: dict[str, float], p_vapor: torch.Tensor | None = None
) -> torch.Tensor:
    """The TOTAL curvature the pressure field demands, per front sample ``(N, 1)``::

        kappa_demanded = We (p_v(t) - p_liq(Gamma))

    The same equation :func:`laplace_jump_residual` scores, read the other way
    round. That residual asks "does this shape satisfy the pressure?"; this asks
    "what shape would?" -- and the difference is the whole distinction between
    the physics being a penalty on a learned shape and the physics DETERMINING
    one. Both callers share this definition so the two can never drift.

    Subtract :func:`gap_curvature` to get the in-plane part a 2-D front can
    actually carry; the gap-direction half is a property of the channel, not of
    the curve, and no shape can change it.

    ``p_vapor`` overrides the model's own trained unknown. A front-geometry run
    that is not sharp has no such unknown to read -- but it is exactly the
    baseline this quantity exists to compare against, so the caller supplies the
    estimate rather than being refused an answer.
    """
    liquid = _liquid_pressure(model, front, groups)
    vapour = model.p_vapor(front.points[:, 2:3]) if p_vapor is None else p_vapor
    return groups["We"] * (vapour - liquid)


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
