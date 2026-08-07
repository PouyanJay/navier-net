"""Physics diagnostics: whether the solution obeys the physics, not just the pixels.

An overlap metric cannot see a violated force balance. The R3 baseline scored
IoU 0.929 on its trained frames and 0.866 on held-out ones while its momentum
residual never descended (flat ~4.7 over 1500 steps), its Young-Laplace jump at
the nose was 85% wrong, and its body carried essentially no in-plane curvature
variation -- so the localized necking that precedes detachment was not merely
unlearned but unreachable: measured neck depth 0.000 against the masks' 0.474.
These are the numbers that show that.

Every quantity here is non-dimensional and read from the model and the measured
masks; nothing is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from naviernet.physics.groups import compute_groups
from naviernet.physics.residuals import (
    gap_curvature,
    pressure_implied_curvature,
    total_curvature,
)

# Stations along the bubble the half-width profiles are compared on. Odd, so a
# station lands exactly mid-bubble where the measured neck sits.
PROFILE_STATIONS = 9

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
    # The capillary pressure's range across the body interior over the length it
    # varies on. Zero means no station along the bubble is distinguished from any
    # other, so no neck can be selected however long the run trains.
    axial_capillary_gradient: float
    # Where the shape and the pressure disagree, region by region: RMS of
    # (curvature the pressure demands - curvature the shape carries), over the
    # nose cap, the body, and the root cap. The errors above say HOW MUCH the
    # jump condition is violated; this says WHERE, which is what tells you which
    # part of the parameterization is binding.
    curvature_gap: dict[str, float]
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
    u = _station_tensor(n_stations, _device_of(geometry))
    times = torch.full_like(u, float(t))
    with torch.no_grad():
        radius = geometry.half_width(u, times)
    return radius.squeeze(1).cpu().numpy().astype(float)


def diagnostic_times(data, device=None) -> torch.Tensor:
    """The times the front is measured at: the dataset's own frame instants.

    Not an arbitrary grid -- every per-frame figure is then directly comparable
    to the mask it is scored against, and the aggregate covers exactly the window
    the run was supervised on.
    """
    return torch.tensor(np.asarray(data.t), dtype=torch.float32, device=device).reshape(-1, 1)


def _device_of(module) -> torch.device | None:
    """The device a model's parameters live on, so every tensor built here lands
    beside them instead of defaulting to CPU and crashing the first GPU run."""
    return next(module.parameters(), torch.empty(0)).device


def _station_tensor(n_stations: int, device) -> torch.Tensor:
    return torch.tensor(_stations(n_stations), dtype=torch.float32, device=device).reshape(
        -1, 1
    )


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
    front = geometry.front(
        diagnostic_times(data, _device_of(geometry)),
        n_body=DIAGNOSTIC_BODY_SAMPLES,
        n_cap=DIAGNOSTIC_CAP_SAMPLES,
    )

    nose_error, front_error = _laplace_errors(model, front, groups)
    last = len(data.t) - 1
    return InterfaceDiagnostics(
        laplace_error_nose=nose_error,
        laplace_error_front=front_error,
        axial_capillary_gradient=_axial_capillary_gradient(front, groups),
        curvature_gap=_curvature_gap(model, front, groups),
        neck_model=neck_of_profile(model_half_width_profile(geometry, float(data.t[last]))),
        neck_measured=neck_of_profile(measured_half_width_profile(data, last)),
    )


def _laplace_errors(model, front, groups: dict[str, float]) -> tuple[float, float]:
    """``(nose, whole-front)`` jump error. ``nan`` when the run has no vapour
    pressure to compare against.

    Two different normalisations, deliberately.

    At the nose the total capillary pressure is far from zero, so a POINTWISE
    relative error means something, and it is what the gate reads.

    Across the front it is normalised by the capillary pressure's VARIATION, not
    its magnitude. The gap term ``2/H*`` is a large constant (~4 against an
    in-plane O(1)), and a space-independent ``p_v`` absorbs any constant exactly
    -- so scoring against the magnitude would flatter every model, satisfied or
    not. What a free ``p_v`` cannot absorb is the variation along the front, and
    that is precisely the part that selects the shape.
    """
    if "p" not in getattr(model, "fields", ()):
        # A Stage-A run has no pressure field, so there is no jump to score. The
        # shape diagnostics below are pure geometry and stay measurable.
        return float("nan"), float("nan")
    with torch.no_grad():
        capillary = (total_curvature(front, groups) / groups["We"]).detach()
        liquid = model.pressure(front.points) + model.film_offset(front.on_cap)
        residual = (_vapour_pressure(model, front) - liquid - capillary).abs()

    # The nose cap: the far closure, where curvature is largest and the jump is
    # least ambiguous. `u == 1` marks it (see FrontSamples).
    magnitude = capillary.abs().squeeze(1)
    nose = (front.u.squeeze(1) == 1.0) & (magnitude > CAPILLARY_FLOOR)
    nose_error = _rms((residual.squeeze(1) / magnitude.clamp(min=CAPILLARY_FLOOR))[nose])

    variation = _rms(capillary.squeeze(1) - capillary.mean())
    front_error = _rms(residual.squeeze(1)) / max(variation, CAPILLARY_FLOOR)
    return nose_error, front_error


def _curvature_gap(model, front, groups: dict[str, float]) -> dict[str, float]:
    """Per region: RMS of (curvature the pressure demands - curvature the shape
    carries), in-plane.

    The Laplace errors above are one number for the whole front, so a shape that
    is right everywhere but wrong at one end scores the same as one that is
    uniformly slightly wrong. This splits it, because those two call for opposite
    fixes: the first says a REGION of the parameterization cannot express what
    the physics wants, the second says the physics is being outvoted everywhere.

    In-plane only: the gap term is a property of the channel, and no shape can
    change it, so including it would credit or blame the curve for a number it
    does not control.
    """
    if "p" not in getattr(model, "fields", ()):
        return {}  # a Stage-A run has no pressure to demand anything
    with torch.no_grad():
        # The same estimated vapour pressure the jump errors use, so a diffuse
        # front-geometry baseline is measurable by one definition too.
        demanded = (
            pressure_implied_curvature(
                model, front, groups, p_vapor=_vapour_pressure(model, front)
            )
            - gap_curvature(front.normal_speed, groups)
        ).squeeze(1)
        carried = front.kappa_par.squeeze(1).detach()
    on_cap, u = front.on_cap.squeeze(1) > 0, front.u.squeeze(1)
    regions = {
        "nose_cap": on_cap & (u == 1.0),
        "body": ~on_cap,
        "root_cap": on_cap & (u == 0.0),
    }
    return {
        name: _rms((demanded - carried)[sel])
        for name, sel in regions.items()
        if bool(sel.any())
    }


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
    u = _station_tensor(PROFILE_STATIONS, times.device)
    interior = {}
    for t in unique:
        at_t = torch.full_like(u, float(t))
        frame = geometry.frame(at_t)
        spine = torch.cat(
            [frame.ax + u * (frame.bx - frame.ax), geometry.centerline(u, at_t), at_t], dim=1
        )
        interior[float(t)] = model.pressure(spine).mean()
    return torch.tensor([[interior[float(t)]] for t in times.squeeze(1)], device=times.device)


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
    total = front.kappa_par + gap_curvature(front.normal_speed, groups)
    u = front.u.squeeze(1)
    body = (front.on_cap.squeeze(1) == 0) & (front.side.squeeze(1) > 0) & (u >= lo) & (u <= hi)
    x = front.points[body, 0].detach()
    pressure = (total[body].squeeze(1) / groups["We"]).detach()
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


def residual_convergence(history: list[dict], terms: tuple[str, ...], after_step: int) -> dict:
    """Whether each physics residual actually DESCENDED, per term.

    The value of a residual says little on its own -- the R3 baseline's momentum
    term sat at ~4.7, which is neither obviously large nor obviously small. What
    it never did was fall: 6.78 at the step it engaged, 4.68 at the end. So the
    figure reported is the ratio of the term's mean over the last fifth of its
    active window to its mean over the first fifth. Below 1 it is converging;
    at or above 1 the optimiser is not solving that equation at all.

    ``after_step`` excludes the Stage-B warm-up, where these terms are computed
    and logged but held at zero weight -- averaging across the gate would mix two
    different objectives.
    """
    report: dict[str, dict[str, float]] = {}
    for term in terms:
        values = [r[term] for r in history if term in r and r.get("step", 0) > after_step]
        if len(values) < 2:
            continue
        window = max(1, len(values) // 5)
        first = float(np.mean(values[:window]))
        last = float(np.mean(values[-window:]))
        report[term] = {
            "first": first,
            "last": last,
            "ratio": last / first if first > 0 else float("nan"),
        }
    return report


def physics_report(model, data, groups: dict[str, float] | None = None) -> dict:
    """The physics block of ``metrics.json``: the interface conditions, the
    drainage drive, and the neck the model holds against the measured one.

    Per frame as well as in aggregate, so the UI can show WHERE along the run the
    physics holds, not just whether it does on average.
    """
    groups = groups if groups is not None else compute_groups(model.cfg)
    summary = interface_diagnostics(model, data, groups)
    geometry = model.nets["phi"]

    per_frame = []
    for row, frame in enumerate(data.frame_numbers):
        model_profile = model_half_width_profile(geometry, float(data.t[row]))
        measured_profile = measured_half_width_profile(data, row)
        model_neck = neck_of_profile(model_profile)
        measured_neck = neck_of_profile(measured_profile)
        per_frame.append(
            {
                "frame": int(frame),
                "neck_depth_model": model_neck.depth,
                "neck_depth_measured": measured_neck.depth,
                "neck_location_model": model_neck.location,
                "neck_location_measured": measured_neck.location,
                "half_width_model": [float(v) for v in model_profile],
                "half_width_measured": [float(v) for v in measured_profile],
            }
        )

    return {
        "laplace_error_nose": summary.laplace_error_nose,
        "laplace_error_front": summary.laplace_error_front,
        "axial_capillary_gradient": summary.axial_capillary_gradient,
        "curvature_gap": summary.curvature_gap,
        "neck_depth_model": summary.neck_model.depth,
        "neck_depth_measured": summary.neck_measured.depth,
        "neck_location_model": summary.neck_model.location,
        "neck_location_measured": summary.neck_measured.location,
        "profile_stations": [float(u) for u in _stations(PROFILE_STATIONS)],
        "per_frame": per_frame,
    }
