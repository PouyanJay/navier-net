"""The measured outline: each frame's interface drawn the way a pen draws it.

Why this exists: the trained capsule builds the bubble from parts (a body
profile and two caps) and its curvature jumps where they meet. Young-Laplace
turns a curvature jump into a pressure step: 8 /mm at the nose seam is about
70 Pa that the images do not show. The faired contour in the tensors has the
opposite defect: band-limiting a long, flat-sided curve leaves ripples of about
2 /mm along its flanks.

The curve here is one stroke from the nose apex to the root apex, mirrored about
the channel axis. Its curvature is a cubic spline in arc length, so it changes
gradually everywhere, with no seams. The stroke is fitted to the centre of the
meniscus rim, and a penalty on how fast the curvature changes keeps it fair. The
penalty is scaled by the local curvature: a flat flank pays the full price for
any ripple, while a tight tip can still tighten quickly, as a pen does in a
sharp turn. On Series-1-w300 the stroke sits 0.9 um rms from the rim centre
(95 % of the rim within 2 um) and has no curvature wiggles.

Each frame is stored as stations at fixed parameter values, so the model can
interpolate the outline between frames (see
:mod:`naviernet.models.measured_outline`). The parameter is anchored in absolute
arc length near both apexes and stretches only along the body. A station near
the nose therefore sits at the same distance from the apex in every frame, and
interpolating between frames never blends a cap with a flank.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from naviernet.data.preprocess import fill_holes, medial_field, rim_band, seal_fov_cut
from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths

log = get_logger(__name__)

# The stroke: curvature coefficients (a clamped cubic B-spline over arc length)
# and the dense samples it is integrated on.
PEN_COEFFICIENTS = 30
PEN_SAMPLES = 4000

# The fairness penalty: integral of (dkappa/ds * g)^2 ds, in um^5, with
# g = K0^2 / (kappa^2 + K0^2) the curvature scaling (1 on a flat flank, small in
# a tight tip). Chosen on Series-1-w300 as the stiffest setting whose curvature
# has no wiggle in any frame (one minimum along the flank, rising to each apex)
# while the rim stays within 1 um rms.
PEN_STIFFNESS_UM5 = 3.0e7
PEN_CURVATURE_SCALE_PER_UM = 4.0e-3

# How hard the stroke is held to end on the axis with a vertical tangent and an
# even curvature (zero dkappa/ds) at both apexes: scales on y (um), the tangent
# angle (rad) and dkappa/ds (1/um^2), making each a near-constraint next to the
# rim residual.
END_WEIGHTS = (30.0, 3000.0, 3.0e4)

# Rim points further than this from the stroke (in pixels) count less and less:
# a notch where a channel trace grazes the rim cannot drag the curve.
ROBUST_SCALE_PX = 1.5

# Each rim point is tied to a station along the stroke, and the ties are redone
# from the fitted curve this many times.
REPROJECTIONS = 4

# Rim points this close to a field-of-view edge the bubble touches are dropped:
# the rim is sealed across the cut there, so its centre is not the interface.
FOV_MARGIN_PX = 20

# Stations per half outline (nose apex to root apex, both included).
STATIONS_PER_HALF = 321

# The parameter's anchoring: within END_ARC_UM of each apex, a station sits at a
# fixed arc length from that apex; the body between stretches. END_FRACTION is
# the share of the parameter each end takes, and RAMP_FRACTION how gradually the
# stretch takes over (a smooth ramp, so the stations' spacing is smooth).
END_ARC_UM = 200.0
END_FRACTION = 0.2
RAMP_FRACTION = 0.1

# The arcs labelled as caps on the sampled front, measured from each apex: about
# a quarter turn of each apex's radius on Series-1-w300 (115 um at the nose, 35 um
# at the root). Labels only: the outline is one curve either way.
NOSE_CAP_UM = 180.0
ROOT_CAP_UM = 60.0


def outline_path(paths: RunPaths):
    """Where a dataset's measured outline is stored, beside its tensors."""
    return paths.processed_dir / "outline.npz"


def rim_centre(band: np.ndarray, axis_row: float) -> np.ndarray:
    """The rim centre of a mirror-averaged frame as one ordered half outline.

    Returns ``(n, 2)`` points ``(column, height above the axis in pixels)`` from
    the nose apex to the root apex, both on the axis. The rim's medial field is
    averaged with its reflection about the axis first, so both halves of the
    bubble count and the half is symmetric by construction.
    """
    import contourpy

    sealed = seal_fov_cut(band) if band[:, [0, -1]].any() else band
    filled = fill_holes(sealed)
    hole = (filled & (1 - sealed)).astype(np.uint8)
    field = medial_field(filled, hole).astype(np.float64)
    rows, cols = np.mgrid[0 : field.shape[0], 0 : field.shape[1]].astype(np.float64)
    mirrored = map_coordinates(field, [2.0 * axis_row - rows, cols], order=1, mode="nearest")
    line = max(contourpy.contour_generator(z=0.5 * (field + mirrored)).lines(0.0), key=len)
    col, height = line[:, 0], axis_row - line[:, 1]

    # The longest run on one side of the axis is that half, apex to apex.
    crossings = np.nonzero(np.diff(np.sign(height)) != 0)[0] + 1
    runs = np.split(np.arange(len(col)), crossings)
    if len(runs) > 1 and height[runs[0][0]] * height[runs[-1][-1]] > 0:
        runs = [np.concatenate([runs[-1], runs[0]]), *runs[1:-1]]
    half = max((r for r in runs if height[r].mean() > 0), key=len)
    points = np.column_stack([col[half], height[half]])
    points = points if points[0, 0] < points[-1, 0] else points[::-1]

    width = band.shape[1]
    keep = np.ones(len(points), dtype=bool)
    if band[:, 0].any():
        keep &= points[:, 0] > FOV_MARGIN_PX
    if band[:, -1].any():
        keep &= points[:, 0] < width - 1 - FOV_MARGIN_PX
    return points[keep]


class PenStroke:
    """A half outline as a stroke of smoothly varying curvature.

    The stroke starts on the axis at ``x_apex`` heading straight up (away from
    the axis) and turns by the curvature ``kappa(s) = B(s/L) c`` as it goes.
    Coordinates are in um: ``x`` along the channel in the camera's direction,
    ``y`` the height above the axis.
    """

    def __init__(self, n_coefficients: int = PEN_COEFFICIENTS, n_samples: int = PEN_SAMPLES):
        knots = np.r_[[0.0] * 3, np.linspace(0.0, 1.0, n_coefficients - 2), [1.0] * 3]
        spline = BSpline(knots, np.eye(n_coefficients), 3)
        self.sigma = np.linspace(0.0, 1.0, n_samples)
        self.basis = spline(self.sigma)
        self.slope_basis = spline.derivative()(self.sigma)
        self.n_coefficients = n_coefficients

    def trace(self, params: np.ndarray):
        """``(x, y, angle, kappa)`` along the stroke for ``params = (x_apex,
        length, *coefficients)``."""
        x_apex, length, coefficients = params[0], params[1], params[2:]
        kappa = self.basis @ coefficients
        ds = length / (len(self.sigma) - 1)
        angle = 0.5 * np.pi - _cumulative(kappa, ds)
        x = x_apex + _cumulative(np.cos(angle), ds)
        y = _cumulative(np.sin(angle), ds)
        return x, y, angle, kappa

    def fit(
        self, rim: np.ndarray, um_per_px: float, start: np.ndarray | None = None
    ) -> np.ndarray:
        """The stroke through ``rim`` (``(n, 2)`` um, apex to apex), as params.

        ``start`` is where the fit begins. Without it the stroke starts as an
        even half turn over the rim's own length, which needs the rim to run
        apex to apex; a frame cut by the field of view starts from a nearby
        frame's stroke instead.
        """
        if start is None:
            arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(rim, axis=0).T))]
            tie = arc / arc[-1]
            params = np.r_[rim[0, 0], arc[-1], np.full(self.n_coefficients, np.pi / arc[-1])]
        else:
            params = start.copy()
            x, y, _, _ = self.trace(params)
            tie = self._nearest(rim, x, y)
        robust = ROBUST_SCALE_PX * um_per_px / np.sqrt(len(rim))
        for _ in range(REPROJECTIONS):
            params = least_squares(
                self._residuals,
                params,
                args=(rim, tie),
                loss="soft_l1",
                f_scale=robust,
                x_scale="jac",
            ).x
            x, y, _, _ = self.trace(params)
            tie = self._nearest(rim, x, y)
        return params

    def _residuals(self, params: np.ndarray, rim: np.ndarray, tie: np.ndarray) -> np.ndarray:
        x, y, angle, kappa = self.trace(params)
        length = params[1]
        miss = np.r_[
            np.interp(tie, self.sigma, x) - rim[:, 0], np.interp(tie, self.sigma, y) - rim[:, 1]
        ] / np.sqrt(len(rim))
        slope = (self.slope_basis @ params[2:]) / length
        scale = PEN_CURVATURE_SCALE_PER_UM**2 / (kappa**2 + PEN_CURVATURE_SCALE_PER_UM**2)
        fairness = np.sqrt(PEN_STIFFNESS_UM5 * length / len(self.sigma)) * slope * scale
        on_axis, upright, even = END_WEIGHTS
        ends = np.r_[
            on_axis * y[-1],
            upright * (angle[-1] + 0.5 * np.pi),
            even * slope[0],
            even * slope[-1],
        ]
        return np.r_[miss, fairness, ends]

    def _nearest(self, rim: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Each rim point's nearest station along the stroke (as ``s / L``)."""
        coarse = slice(None, None, 4)
        gap = (rim[:, None, 0] - x[None, coarse]) ** 2 + (
            rim[:, None, 1] - y[None, coarse]
        ) ** 2
        return self.sigma[coarse][np.argmin(gap, axis=1)]


def _cumulative(rate: np.ndarray, ds: float) -> np.ndarray:
    """Trapezoid running integral starting at zero."""
    return np.r_[0.0, np.cumsum(0.5 * (rate[1:] + rate[:-1]) * ds)]


def station_arc(length: float, n: int = STATIONS_PER_HALF) -> np.ndarray:
    """Arc length (um, from the nose apex) of each station on a half of
    ``length``: absolute within ``END_ARC_UM`` of each apex, stretched between.

    The spacing ``ds/dq`` is ``END_ARC_UM / END_FRACTION`` at the ends and a
    constant set by the length in the middle, joined by smooth ramps, so the
    stations are smooth functions of the parameter (which the interpolating
    spline needs) and the two ends correspond frame to frame.
    """
    q = np.linspace(0.0, 1.0, 8 * (n - 1) + 1)
    window = _ramp((q - END_FRACTION) / RAMP_FRACTION) * _ramp(
        (1.0 - END_FRACTION - q) / RAMP_FRACTION
    )
    end_rate = END_ARC_UM / END_FRACTION
    share = np.trapezoid(window, q)
    body_rate = (length - end_rate * (1.0 - share)) / share
    if body_rate <= 0:
        raise ValueError(
            f"a half outline of {length:.0f} um is too short for stations anchored "
            f"{END_ARC_UM:.0f} um into each end"
        )
    rate = end_rate * (1.0 - window) + body_rate * window
    arc = np.r_[0.0, np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(q))]
    return (arc * length / arc[-1])[::8]


def _ramp(x: np.ndarray) -> np.ndarray:
    """A smooth step: 0 for ``x <= 0``, 1 for ``x >= 1``, infinitely smooth."""

    def bump(z):
        return np.where(z > 0, np.exp(-1.0 / np.maximum(z, 1e-12)), 0.0)

    return bump(x) / (bump(x) + bump(1.0 - x))


def measured_outline(cfg, paths: RunPaths) -> dict:
    """Fit every usable frame and return the stations in the tensors' units.

    ``stations`` is ``(frames, 2 * (STATIONS_PER_HALF - 1), 2)`` ``(x*, y*)``,
    one closed curve per frame: index 0 is the nose apex, index
    ``STATIONS_PER_HALF - 1`` the root apex, the first half on the ``y*`` above
    the axis and the second its mirror, so the curve runs once round.
    """
    archive = np.load(paths.tensors)
    meta = json.loads(str(archive["meta"]))
    um_per_px, l_ref = float(meta["um_per_px"]), float(meta["L_ref_um"])
    roi = tuple(meta["y_roi"])
    axis_row = 0.5 * sum(meta["wall_rows"]) - roi[0]
    width = archive["alpha"].shape[2]
    stroke = PenStroke()

    halves, report, fits = [], [], []
    for n in meta["frame_numbers"]:
        band = rim_band(cfg, paths, n, roi)
        rim = rim_centre(band, axis_row) * um_per_px
        cut = bool(band[:, [0, -1]].any())
        params = stroke.fit(rim, um_per_px, _advanced(fits) if cut else None)
        fits.append(params)
        x, y, _, _ = stroke.trace(params)
        miss = np.sqrt(
            ((rim[:, None, 0] - x[None, ::2]) ** 2 + (rim[:, None, 1] - y[None, ::2]) ** 2).min(
                1
            )
        )
        arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        at = station_arc(arc[-1])
        halves.append(np.column_stack([np.interp(at, arc, x), np.interp(at, arc, y)]))
        report.append(
            {
                "frame": int(n),
                "rim_rms_um": float(np.sqrt(np.mean(miss**2))),
                "rim_p95_um": float(np.percentile(miss, 95)),
                "length_um": float(arc[-1]),
            }
        )
        log.info(
            "outline frame %d: rim %.2f um rms, 95%% within %.2f um, half length %.0f um",
            n,
            report[-1]["rim_rms_um"],
            report[-1]["rim_p95_um"],
            arc[-1],
        )

    axis_star = (axis_row + 0.5) * um_per_px / l_ref
    stations = np.stack([_closed_star(h, width, um_per_px, l_ref, axis_star) for h in halves])
    end_rate = END_ARC_UM / END_FRACTION
    return {
        "t_star": archive["t_star"].astype(np.float64),
        "stations": stations,
        "axis_star": axis_star,
        "nose_cap_q": NOSE_CAP_UM / end_rate,
        "root_cap_q": ROOT_CAP_UM / end_rate,
        "report": report,
    }


def _advanced(fits: list[np.ndarray]) -> np.ndarray:
    """A start for a frame the field of view cuts: the last stroke, with its apex
    and length moved on by as much as they moved over the frame before."""
    if len(fits) < 2:
        raise ValueError(
            "a frame cut by the field of view needs two earlier frames to start its outline from"
        )
    last, before = fits[-1], fits[-2]
    start = last.copy()
    start[:2] += last[:2] - before[:2]
    return start


def _closed_star(
    half: np.ndarray, width: int, um_per_px: float, l_ref: float, axis_star: float
) -> np.ndarray:
    """A half outline (camera um, height above the axis) as the closed curve in
    ``(x*, y*)``: flipped so ``x*`` runs downstream, as the tensors are."""
    x_star = (width * um_per_px - half[:, 0]) / l_ref - 0.5 * um_per_px / l_ref
    upper = np.column_stack([x_star, axis_star + np.abs(half[:, 1]) / l_ref])
    lower = upper[-2:0:-1] * np.array([1.0, -1.0]) + np.array([0.0, 2.0 * axis_star])
    return np.concatenate([upper, lower])


def write_measured_outline(cfg, paths: RunPaths) -> dict:
    """Fit the dataset's outlines and store them beside its tensors."""
    outline = measured_outline(cfg, paths)
    np.savez_compressed(
        outline_path(paths),
        t_star=outline["t_star"],
        stations=outline["stations"],
        axis_star=outline["axis_star"],
        nose_cap_q=outline["nose_cap_q"],
        root_cap_q=outline["root_cap_q"],
        report=json.dumps(outline["report"]),
    )
    log.info("wrote %s", outline_path(paths))
    return outline
