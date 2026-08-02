"""Physics diagnostics: whether the solution obeys the physics, not just the pixels.

An overlap metric cannot see a violated force balance. The R3 baseline scored
IoU 0.929 on its trained frames and 0.866 on held-out ones while its momentum
residual never descended (flat ~4.7 over 1500 steps), its Young-Laplace jump at
the nose was ~20x too small with the WRONG SIGN, and the axial capillary
pressure gradient along the bubble was identically zero -- so the localized
necking that precedes detachment was not merely unlearned but *dynamically
impossible*. These are the numbers that show that.

Every quantity here is non-dimensional and read from the model and the measured
masks; nothing is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from naviernet.physics.groups import compute_groups
from naviernet.physics.residuals import laplace_jump_residual

# Stations along the bubble the half-width profiles are compared on. Odd, so a
# station lands exactly mid-bubble where the measured neck sits.
PROFILE_STATIONS = 9

# Times the front diagnostics are evaluated at, spread over the run's window.
DIAGNOSTIC_TIMES = 16

# Front resolution for the diagnostics. Denser than training needs: this is
# measurement, and it runs once.
DIAGNOSTIC_BODY_SAMPLES = 128
DIAGNOSTIC_CAP_SAMPLES = 32

# Below this capillary pressure a relative error is meaningless (0/0), so the
# sample is not counted. Far under any resolved cap's 1/(r We).
CAPILLARY_FLOOR = 1e-6


class Neck(NamedTuple):
    """The waist of a half-width profile: how far it collapses, and where.

    ``depth`` is the fractional collapse below the SHALLOWER of the two shoulders
    that bracket it, so a deep neck beside a large head is not flattered by the
    head. A profile whose minimum has nothing above it on one side is tapering,
    not necking, and reads exactly zero -- which is what the R3 baseline does.
    """

    depth: float
    location: float  # normalised station of the waist, 0 at the root, 1 at the nose


@dataclass(frozen=True)
class InterfaceDiagnostics:
    """What the physics is doing, per run."""

    # |p_v - p_liq - kappa/We| relative to |kappa/We|: the Young-Laplace jump the
    # interface is supposed to satisfy. Reported at the nose cap (where curvature
    # is largest and best-conditioned) and over the whole front.
    laplace_error_nose: float
    laplace_error_front: float
    # RMS d/dx of the capillary pressure along the body. Zero means the shape
    # cannot drain: nothing pushes vapour from the mid-body toward the nose.
    axial_capillary_gradient: float
    # The model's own neck, and the measured one, on the last evaluated frame.
    neck_model: Neck
    neck_measured: Neck


def neck_of_profile(profile: np.ndarray) -> Neck:
    """The neck of a half-width profile sampled on a uniform station grid.

    The waist is the profile's minimum; it counts as a neck only if the profile
    rises above it on BOTH sides, which is what distinguishes a bubble pinching
    in the middle from one that simply tapers to an end.
    """
    if profile.ndim != 1 or profile.size < 3:
        raise ValueError(f"a profile needs at least 3 stations, got shape {profile.shape}")
    waist = int(np.argmin(profile))
    if waist == 0 or waist == profile.size - 1:
        return Neck(0.0, waist / (profile.size - 1))
    shoulder = min(profile[:waist].max(), profile[waist + 1 :].max())
    if shoulder <= 0:
        return Neck(0.0, waist / (profile.size - 1))
    return Neck(float(1.0 - profile[waist] / shoulder), waist / (profile.size - 1))


def measured_half_width_profile(
    data, row: int, n_stations: int = PROFILE_STATIONS
) -> np.ndarray:
    """The bubble's half-width at ``n_stations`` even stations along its own
    extent, read from the segmented mask of dataset row ``row``.

    Stations are placed *inside* the extent (the ends are the caps, where the
    half-width collapses to zero and would swamp any neck), on the same
    normalised grid :func:`model_half_width_profile` uses, so the two are
    comparable station by station.
    """
    vapour = data.alpha[row] > 0.5
    columns = np.nonzero(vapour.any(axis=0))[0]
    if columns.size == 0:
        raise ValueError(f"dataset row {row} has no vapour to profile")

    x_first, x_last = data.x[columns[0]], data.x[columns[-1]]
    profile = np.zeros(n_stations, dtype=float)
    for i, u in enumerate(_stations(n_stations)):
        column = int(np.argmin(np.abs(data.x - (x_first + u * (x_last - x_first)))))
        rows = np.nonzero(vapour[:, column])[0]
        if rows.size:
            profile[i] = 0.5 * float(data.y[rows[-1]] - data.y[rows[0]])
    return profile


def model_half_width_profile(
    geometry, t: float, n_stations: int = PROFILE_STATIONS
) -> np.ndarray:
    """The model's own half-width profile at time ``t``, on the same stations."""
    u = torch.tensor(_stations(n_stations), dtype=torch.float32).reshape(-1, 1)
    times = torch.full_like(u, float(t))
    with torch.no_grad():
        radius = geometry.half_width(u, times)
    return radius.squeeze(1).cpu().numpy().astype(float)


def _stations(n_stations: int) -> np.ndarray:
    """Interior stations: the open interval, so the caps are excluded."""
    return np.linspace(0.0, 1.0, n_stations + 2)[1:-1]


def interface_diagnostics(model, data, groups: dict[str, float] | None = None):
    """Measure the interface conditions and the shape the model actually holds.

    Requires the explicit front (``model.front_geometry``); the jump error also
    needs the sharp-interface unknowns, and reads ``nan`` without them rather
    than inventing a vapour pressure that the run never trained.
    """
    geometry = model.nets["phi"]
    if not hasattr(geometry, "front"):
        raise ValueError(
            "interface diagnostics need the explicit front: this run was trained "
            "without model.front_geometry, so there is no parameterized interface "
            "to measure the conditions on."
        )
    groups = groups if groups is not None else compute_groups(model.cfg)
    domain = data.domain
    times = torch.linspace(domain.t_min, domain.t_max, DIAGNOSTIC_TIMES).reshape(-1, 1)
    front = geometry.front(times, n_body=DIAGNOSTIC_BODY_SAMPLES, n_cap=DIAGNOSTIC_CAP_SAMPLES)

    nose_error, front_error = _laplace_errors(model, front, groups)
    last = len(data.t) - 1
    return InterfaceDiagnostics(
        laplace_error_nose=nose_error,
        laplace_error_front=front_error,
        axial_capillary_gradient=_axial_capillary_gradient(front, groups),
        neck_model=neck_of_profile(model_half_width_profile(geometry, float(data.t[last]))),
        neck_measured=neck_of_profile(measured_half_width_profile(data, last)),
    )


def _laplace_errors(model, front, groups: dict[str, float]) -> tuple[float, float]:
    """``(nose, whole-front)`` jump error. ``nan`` when the run has no vapour
    pressure to compare against.

    Two different normalisations, deliberately. At the nose the capillary
    pressure is O(1/r) and well away from zero, so a POINTWISE relative error is
    meaningful and is what the gate reads. Across the whole front it is not: a
    straight body section has ``kappa_par -> 0``, and dividing by it turns a
    perfectly small residual into an arbitrarily large ratio (measured: 1.8e3 on
    an untrained model). So the front-wide figure is the RMS residual against the
    RMS capillary pressure -- one global scale, finite wherever the caps are.
    """
    with torch.no_grad():
        vapour = _vapour_pressure(model, front)
        residual = (vapour - model.pressure(front.points) - front.kappa_par / groups["We"]).abs()
    capillary = (front.kappa_par / groups["We"]).abs().detach()

    # The nose cap: the far closure, where curvature is largest and the jump is
    # least ambiguous. `u == 1` marks it (see FrontSamples).
    nose = (front.u.squeeze(1) == 1.0) & (capillary.squeeze(1) > CAPILLARY_FLOOR)
    nose_error = _rms((residual / capillary.clamp(min=CAPILLARY_FLOOR)).squeeze(1)[nose])

    scale = _rms(capillary.squeeze(1))
    front_error = _rms(residual.squeeze(1)) / max(scale, CAPILLARY_FLOOR)
    return nose_error, front_error


def _vapour_pressure(model, front) -> torch.Tensor:
    """The vapour-interior pressure at each front sample's time.

    A sharp-interface run carries it as a trained unknown. Any OTHER
    front-geometry run does not -- but the jump error is exactly the number we
    want to compare such a run against, so it is estimated the same way the
    physics justifies: the mean liquid-pressure the model predicts along the
    bubble's own spine at that time. The vapour is near-isobaric
    (``mu_l/mu_v ~ 37``), so its interior pressure IS that mean, up to the drop
    the assumption neglects. Measuring a diffuse baseline this way is what makes
    a before/after honest -- both runs are scored by one definition.
    """
    times = front.points[:, 2:3]
    if getattr(model, "sharp_interface", False):
        return model.p_vapor(times)

    geometry = model.nets["phi"]
    unique = torch.unique(times)
    u = torch.linspace(0.0, 1.0, PROFILE_STATIONS + 2)[1:-1].reshape(-1, 1)
    interior = {}
    for t in unique:
        at_t = torch.full_like(u, float(t))
        frame = geometry.frame(at_t)
        spine = torch.cat(
            [frame.ax + u * (frame.bx - frame.ax), geometry.centerline(u, at_t), at_t], dim=1
        )
        interior[float(t)] = model.pressure(spine).mean()
    return torch.tensor(
        [[interior[float(t)]] for t in times.squeeze(1)], device=times.device
    )


# The body's interior, as a fraction of the spine: the stations the neck lives
# on. The remaining tenth at each end is the taper into the caps, where the
# radius turns over fast and would dominate any variation measured across it.
BODY_INTERIOR = (0.1, 0.9)


def _axial_capillary_gradient(front, groups: dict[str, float]) -> float:
    """Mean axial gradient of the capillary pressure across the body's interior:
    its range divided by the length it varies over, averaged across times.

    This is the drainage drive. A jump condition can only select a shape where
    the capillary pressure *varies* along the bubble -- if it is the same at
    every station, no station is distinguished and no neck can be selected. A
    straight-sided capsule has ``kappa_par == 0`` along its whole body, so this
    reads zero: the R3 baseline's necking was not merely unlearned, it was
    unreachable.

    The range, not an RMS of local derivatives: the latter is dominated by the
    taper into the caps and would report a healthy number for a body that is in
    fact perfectly straight where it matters.
    """
    lo, hi = BODY_INTERIOR
    u = front.u.squeeze(1)
    body = (
        (front.on_cap.squeeze(1) == 0) & (front.side.squeeze(1) > 0) & (u >= lo) & (u <= hi)
    )
    x = front.points[body, 0].detach()
    pressure = (front.kappa_par[body].squeeze(1) / groups["We"]).detach()
    times = front.points[body, 2].detach()

    gradients = []
    for t in torch.unique(times):
        at_t = times == t
        if int(at_t.sum()) < 2:
            continue
        span = x[at_t].max() - x[at_t].min()
        if span <= 0:
            continue
        spread = pressure[at_t].max() - pressure[at_t].min()
        gradients.append(spread / span)
    return float(torch.stack(gradients).mean()) if gradients else 0.0


def _rms(values: torch.Tensor) -> float:
    return float(torch.sqrt((values**2).mean())) if values.numel() else float("nan")
