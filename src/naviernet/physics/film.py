"""The liquid film between the bubble and the gap walls (``model.liquid_film``).

In a 150 um gap the bubble spans the channel, and the liquid it displaces does
not vanish -- it is left behind as a thin film against the walls. A
depth-averaged model integrates that film away, and with it three separate
things: the dominant heat-transfer path (film evaporation is >70% of the local
heat transfer in this confinement), the squeezing pressure that pinches a bubble
in the low-Ca regime, and the one property axis that actually separates the
working fluids (film thickness scales as ``(mu/sigma)^{2/3}``, ~4x between the
dielectrics and water, where their kinematic viscosities nearly coincide).

The field is ``delta(x, t)``: film thickness at lab-frame axial position ``x``,
evaluated on the SAME explicit front the sharp-interface conditions already
sample (:meth:`naviernet.models.geometry.GeometricInterface.front`) -- one
small net, no new sampler. Lab frame because the film is attached to the wall:
measured on a trained run the bubble's body recedes at essentially every
station (only the nose advances), so the film beside a flank is the one the
NOSE left behind when it passed that position -- deposition is a condition at
the advancing meniscus, and what happens behind it is depletion, not fresh
deposit.

Deposition follows Aussillous & Quere's saturating form on the LOCAL capillary
number. Deliberately NOT the boundary-layer ``delta0 = C sqrt(nu t)``: with
``nu = 2.6-3.2e-7 m^2/s`` for all four working fluids that correlation is
fluid-blind, and adopting it would build in exactly the blindness this field
exists to remove.
"""

from __future__ import annotations

import torch

# Aussillous & Quere's saturating deposition law, with the pancake-bubble fit of
# Shukla et al. (2019):
#
#     delta0 / (h/2) = P (3 Ca_n)^{2/3} / (1 + P Q (3 Ca_n)^{2/3})
#
# ``Ca_n = mu u_n / sigma`` on the LOCAL normal speed -- the locality is what
# makes the film non-uniform along the front, and the non-uniformity is the
# whole point. The saturation (Bretherton's unbounded 1.34 Ca^{2/3} capped at
# (h/2)/Q) is what keeps a fast nose from depositing more liquid than the gap
# holds. Literature constants, not config: they are properties of the fitted
# law, and a run has no business tuning them.
AQ_P = 0.544
AQ_Q = 2.061


def deposited_thickness(normal_speed: torch.Tensor, groups: dict[str, float]) -> torch.Tensor:
    """The freshly deposited film thickness per front sample, ``(N, 1)``.

    Non-dimensional (lengths on ``L_ref``, so the half-gap is ``H*/2``). The
    local capillary number is ``Ca * v_n`` exactly as the Bretherton pressure
    correction reads it (:func:`naviernet.physics.residuals.gap_curvature`);
    only an ADVANCING front deposits, so a receding section leaves zero film
    (and a negative capillary number never reaches the 2/3 power).
    """
    capillary = (groups["Ca"] * normal_speed).clamp(min=0.0)
    aq_term = AQ_P * (3.0 * capillary) ** (2.0 / 3.0)
    return 0.5 * groups["H_star"] * aq_term / (1.0 + AQ_Q * aq_term)


# Width of the smooth dryout gate, as a fraction of the roughness scale. Small
# enough that the gate acts only within a hair of dryout (the drain dynamics
# above it are the ungated law), large enough to keep the transition
# differentiable for the trainer.
DRY_GATE_WIDTH = 0.2


def film_flux(
    delta: torch.Tensor,
    theta: torch.Tensor,
    groups: dict[str, float],
    resistance: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """The film's evaporative thinning rate, ``(N, 1)``, non-dimensional::

        E_film * theta / (delta + R_gamma) * gate(delta)

    ``E_film = Ja/Pe`` and the default ``R_gamma`` is the Schrage kinetic
    resistance as an equivalent liquid length (see
    :mod:`naviernet.physics.groups`). The resistance in series is NOT optional:
    without it the flux is ``k dT/delta`` and DIVERGES as the film thins -- the
    same pathology the ``|grad alpha|`` source has -- while here it is bounded
    by ``E_film theta / R_gamma`` whatever the film does.

    ``resistance`` overrides the literature value: the trainer passes the
    model's own :meth:`~naviernet.models.pinn.BubblePINN.film_resistance`, the
    literature value scaled by the trained accommodation unknown. Measured
    accommodation coefficients span 1e-3 to 1 (Marek & Straub 2001) -- the
    largest single uncertainty in any thermal closure -- so its magnitude is an
    inverse unknown exactly like ``r_int_star``, while the fluid-to-fluid
    RATIO stays the computed physics.

    The gate takes a station smoothly to zero at the wall-roughness scale:
    a dry station has no continuous liquid left, so it contributes no
    evaporation -- which is what makes dryout a stable end state instead of a
    boundary the ODE keeps pushing through.
    """
    if resistance is None:
        resistance = groups["R_gamma_star"]
    conduction = groups["film_depletion"] * theta / (delta + resistance)
    dry = groups["film_dryout_star"]
    return conduction * torch.sigmoid((delta - dry) / (DRY_GATE_WIDTH * dry))


def depletion_residual(
    model,
    times: torch.Tensor,
    groups: dict[str, float],
    stations: int,
    c: torch.Tensor | None = None,
) -> torch.Tensor:
    """The depletion ODE behind the front, per quadrature point ``(M*stations, 1)``::

        d delta/dt |_x  +  film_flux(delta, theta)  =  0

    The time derivative is at FIXED lab position -- the film is attached to the
    wall -- which is exactly what the ``delta(x, t)`` parameterization makes a
    plain ``d/dt``: no advection term, no station relabeling.

    Quadrature: ``stations`` midpoints between the measured root and the nose
    at each of ``times`` -- the film's footprint, deterministic so the term is
    reproducible across resume. The driving superheat is read at the root
    height (the centerline's anchor; the centerline swings little), and it is
    DETACHED, as is the nose: the film reads the temperature and the front,
    never writes them. Until the coupling stages (mass source, squeezing
    pressure) deliberately open a channel, the film field is a pure record.
    """
    x_root, y_root = model.film_root()
    nose = model.apex(times)[:, 0:1].detach()
    fractions = (
        torch.arange(stations, dtype=times.dtype, device=times.device) + 0.5
    ) / stations
    x = (x_root + (nose - x_root) * fractions.reshape(1, -1)).reshape(-1, 1)
    t = times.repeat_interleave(stations, dim=0).detach().requires_grad_(True)
    cx = None if c is None else c.expand(x.shape[0], -1)

    delta = model.film(x, t, cx)
    ddt = torch.autograd.grad(delta, t, torch.ones_like(delta), create_graph=True)[0]
    points = torch.cat([x, torch.full_like(x, y_root), t.detach()], dim=1)
    theta = model.temperature(points, cx).detach()
    return ddt + film_flux(delta, theta, groups, model.film_resistance(groups))


def advancing_mask(front) -> torch.Tensor:
    """1 where the front advances into liquid, 0 elsewhere, ``(N, 1)``, detached.

    Only an advancing meniscus deposits a film -- and on the real bubble that is
    the nose cap (the body measurably recedes as the capsule elongates), which
    is why the caps are IN: the nose meniscus is the depositing Bretherton
    meniscus. Where the front is static or receding the film is whatever was
    deposited earlier, less what has evaporated -- depletion's job, not this
    term's -- so those samples are left unconstrained here.
    """
    return (front.normal_speed > 0.0).to(front.normal_speed.dtype).detach()


def deposition_residual(
    model, front, groups: dict[str, float], c: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-sample film deposition residual on the explicit front, ``(N, 1)``::

        delta(x, t) - delta_0(Ca_n)    where the front advances;  0 elsewhere

    Both the target and the film net's coordinates are DETACHED: the film
    learns from the front's motion, never the reverse. Deposition is a one-way
    record of where the meniscus went and how fast; letting this term pull the
    front toward whatever the film net happens to hold would couple the shape
    to an auxiliary field before the physics that earns that coupling (the
    squeezing pressure) exists.
    """
    target = deposited_thickness(front.normal_speed, groups).detach()
    cx = None if c is None else c.expand(front.points.shape[0], -1)
    return advancing_mask(front) * (model.film_thickness(front, cx) - target)


# The bubble deposits a film on BOTH gap walls, and both evaporate. The
# deposition law's delta is per wall, so every volume budget carries the two.
BOTH_WALLS = 2.0


def film_source_density(
    delta: torch.Tensor,
    theta: torch.Tensor,
    groups: dict[str, float],
    resistance: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """The depth-averaged dilatation the films' evaporation creates, ``(N, 1)``::

        s = (2 / H*) (rho_ratio - 1) film_flux(delta, theta)

    Derivation, and why each factor is a mass-conservation statement: each wall's
    film thins at ``film_flux`` (liquid volume per footprint area per time), and
    there are two walls; each unit of liquid volume becomes ``rho_l/rho_v`` of
    vapour, for a NET volume creation of ``(rho_ratio - 1)`` per unit consumed;
    and continuity integrates ``div(u)`` across the gap, so a per-area volume
    rate becomes a dilatation by ``1/H*``. Drop any factor and the source no
    longer creates the volume the films actually give up.
    """
    return (
        BOTH_WALLS
        * (groups["rho_ratio"] - 1.0)
        / groups["H_star"]
        * film_flux(delta, theta, groups, resistance)
    )


def film_source_target(
    model, x: torch.Tensor, groups: dict[str, float], c: torch.Tensor | None = None
) -> torch.Tensor:
    """What the mass source should be at bulk points ``x``, per the film, ``(N, 1)``.

    The film's evaporation happens across the whole FOOTPRINT -- wherever the
    bubble covers the walls -- so the density is gated by the vapour fraction,
    not by ``|grad alpha|``: this is exactly the distribution change the film
    exists to make. The old closure put the mass where the interface is
    sharpest; this one puts it where the film is present and thin (the flux
    RISES as the film thins, until dryout shuts the station off).

    The FIELDS are detached -- the closure is one-way in them, the same design
    as the shipped evap closure and for the same reason: ``s`` learns from the
    film, and nothing can flatten the interface, cool the superheat, or rewrite
    the film to satisfy the source it observes. The one live parameter is the
    model's trained kinetic resistance: the accommodation coefficient is the
    admitted unknown, and this closure -- where the film's implied growth meets
    the growth the data actually shows -- is exactly where it is identifiable.
    ``r_int_star`` plays the same role in the closure this one replaces.
    """
    cx = None if c is None else c.expand(x.shape[0], -1)
    with torch.no_grad():
        delta = model.film(x[:, 0:1], x[:, 2:3], cx)
        theta = model.temperature(x, cx)
        alpha = model.alpha(x, cx)
    return film_source_density(delta, theta, groups, model.film_resistance(groups)) * alpha


def film_source_residual(
    model, x: torch.Tensor, groups: dict[str, float], c: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-point closure residual: ``s(x) - film_source_target(x)``, ``(N, 1)``."""
    cx = None if c is None else c.expand(x.shape[0], -1)
    return model.source(x, cx) - film_source_target(model, x, groups, c)


def film_surface_pressure(
    model, front, groups: dict[str, float], c: torch.Tensor | None = None
) -> torch.Tensor:
    """The film's capillary pressure perturbation at each front sample,
    ``(N, 1)``, DETACHED -- the x-varying part of the pressure the BODY
    meniscus faces.

    Why capillary-static and not an axial-flux solve. The first formulation
    here WAS the lubrication BVP for the film's axial flux -- and it measured
    itself out of the model: with g ~ Re delta^3/3 ~ 1e-4, even a
    well-trained depletion residual (~1e-3) implies pressures ~50x the
    capillary scale, saturating any physical cap into a constant offset with
    no spatial selectivity. That stiffness IS the physics: a micron film over
    a millimetre body is axially SEALED, which is Bretherton's leading order
    -- the film does not flow, and its pressure is set by its own surface's
    capillary equilibrium::

        p_film(x, t) = p_v(t) - kappa_surface / We,
        kappa_surface ~ (gap part) - d^2 delta / dx^2

    The gap part and ``p_v`` are x-uniform (the scalar ``film_offset``'s
    territory); the term that can SELECT a station is the film-thickness
    curvature, and the sign does the necking: at a locally THINNING patch
    (a delta minimum, ``delta_xx > 0`` -- incipient dryout) the film pressure
    rises toward ``p_v``, the jump the body must satisfy shrinks, and the
    demanded interface curvature turns concave -- the waist forms where the
    film thins. Richards & Pegler's route: pinch-off is controlled by the
    film reaching the wall, with Re and We absent.

    Returned mean-free per time (the constant belongs to ``film_offset``),
    soft-capped at the film's capillary relief scale ``(2/H*)/We`` (beyond
    that the film deforms rather than transmitting), zero on the caps (they
    face bulk liquid), and DETACHED: a computed forcing, like the solved
    shape coefficients -- the geometry answers through the curvature side of
    the jump, and gradients back into the film would let the jump rewrite
    the film to excuse the shape.
    """
    body = (1.0 - front.on_cap).squeeze(1)
    x = front.points[:, 0:1].detach().requires_grad_(True)
    t = front.points[:, 2:3].detach()
    cx = None if c is None else c.expand(x.shape[0], -1)

    # The curvature needs autograd on this function's OWN leaf whatever the
    # caller's grad mode -- diagnostics legitimately evaluate the jump under
    # no_grad, and the forcing is detached on return either way.
    with torch.enable_grad():
        delta = model.film(x, t, cx)
        d_x = torch.autograd.grad(delta, x, torch.ones_like(delta), create_graph=True)[0]
        d_xx = torch.autograd.grad(d_x, x, torch.ones_like(d_x))[0].detach().squeeze(1)

    # Mean-free within each time row, over the body samples only.
    times = torch.unique(t)
    row = (t.squeeze(1).reshape(-1, 1) - times.reshape(1, -1)).abs().argmin(dim=1)
    weight = torch.zeros(times.shape[0], device=t.device).index_add(0, row, body)
    total = torch.zeros(times.shape[0], device=t.device).index_add(0, row, d_xx * body)
    centred = (d_xx - (total / weight.clamp(min=1.0))[row]) * body

    relief = (2.0 / groups["H_star"]) / groups["We"]
    return (relief * torch.tanh(centred / groups["We"] / relief)).reshape(-1, 1)
