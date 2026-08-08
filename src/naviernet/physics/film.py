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
