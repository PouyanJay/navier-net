"""Root-window measurement: what the masks say about the bubble's root end.

The diagnostics' half-width stations deliberately exclude the caps (the ends
are where the half-width collapses and would swamp any neck), so the root cap
is a blind spot: no existing gate can see whether the model's root matches the
measured one. It does not -- measured on Series-1, the masks disagree with a
circle more and more as the event runs (circle-fit residual 12% -> 32% of the
cap radius, taper exponent drifting from the circle's 1/2 toward 0.38), and
that deviation is the driven, time-varying fact the nucleation-root journey is
built around. This module is the measurement: a per-frame fit of the root
window (the first fraction of the bubble's extent) with three views of one
question -- how far from a circle, and in which direction.

- ``taper_exponent``: the p in ``W(d) ~ d^p`` near the root apex. A circle
  tapers as ``sqrt(d)``; a squared-off (blunter) root reads below 1/2.
- ``circle_rms``: RMS radial residual of the best-fit circle over the window,
  as a fraction of its radius -- how much shape the circle cannot carry.
- ``bluntness``: the half-width ratio between a quarter of the window depth
  and the full window depth. A plain functional of the same masks, monotone in
  the taper: the circle sits near ``4^{-1/2} = 0.5``, blunter reads higher.
  This is the per-frame SCALAR the root supervision regresses -- distance- and
  ratio-based, so it stays informative inside the interface blur band where
  pixel losses starve (the undriven-knob lesson).

Everything here is measurement, not model: the fit reads rasters and
coordinates only, so it runs identically on the data masks and (through
:func:`model_root_window_rms`) against any front-geometry run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# The root window: this fraction of the bubble's x-extent, measured inward from
# the root edge. Wide enough to span the whole cap with margin (the cap radius
# is ~20% of the extent late in the event), narrow enough to stay off the body.
ROOT_WINDOW_FRACTION = 0.25

# The bluntness scalar compares the half-width at this fraction of the window
# depth against the half-width at the full depth. A pure power law W ~ d^p makes
# the ratio exactly ``BLUNTNESS_DEPTH_FRACTION ** p`` -- 0.5 for a circle.
BLUNTNESS_DEPTH_FRACTION = 0.25

# A log-log slope through fewer points than this is a line through noise, and a
# circle fit through fewer is underdetermined; the fit reports NaN instead.
MIN_FIT_COLUMNS = 4

# The alpha level defining vapour, matching the dataset's own INTERFACE_ALPHA.
DEFAULT_ALPHA_LEVEL = 0.5


@dataclass(frozen=True)
class RootWindowFit:
    """One frame's root-window measurement. Distances in the dataset's own
    non-dimensional units; ``circle_rms`` and ``bluntness`` are ratios."""

    taper_exponent: float
    circle_rms: float
    bluntness: float
    updown_bias: float  # mean midline offset from the root anchor's y
    tilt: float  # midline slope over the window
    window_depth: float  # the window's x-extent, for provenance
    n_columns: int  # usable columns the fit saw


def fit_root_window(
    vapour: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    y_root: float,
    window: float = ROOT_WINDOW_FRACTION,
    root_at_left: bool = True,
) -> RootWindowFit:
    """Fit the root window of one vapour mask.

    ``vapour`` is ``(H, W)`` boolean over coordinates ``y`` (rows) and ``x``
    (columns); ``y_root`` anchors the up/down bias. ``root_at_left`` says which
    x-extent edge is the root -- the caller decides from the measured pin
    anchor, so a flipped series needs no special-casing here.
    """
    if root_at_left is False:
        # Mirror so the root is at the left edge; x distances are preserved
        # because only distances FROM the root edge enter any fit.
        vapour = vapour[:, ::-1]
        x = (x[-1] + x[0]) - x[::-1]

    columns = np.nonzero(vapour.any(axis=0))[0]
    if columns.size == 0:
        raise ValueError("the mask has no vapour to fit a root window on")

    extent = float(x[columns[-1]] - x[columns[0]])
    depth = window * extent
    x_root_edge = float(x[columns[0]])
    in_window = columns[(x[columns] - x_root_edge) <= depth]

    distance, half_width, midline = _window_profiles(vapour, x, y, in_window, x_root_edge)
    tapered = (distance > 0) & (half_width > 0)
    if int(tapered.sum()) < MIN_FIT_COLUMNS:
        return RootWindowFit(
            taper_exponent=float("nan"),
            circle_rms=float("nan"),
            bluntness=float("nan"),
            updown_bias=float("nan"),
            tilt=float("nan"),
            window_depth=depth,
            n_columns=int(tapered.sum()),
        )

    slope, _ = np.polyfit(np.log(distance[tapered]), np.log(half_width[tapered]), 1)
    tilt, _ = np.polyfit(distance, midline, 1)
    return RootWindowFit(
        taper_exponent=float(slope),
        circle_rms=_circle_rms(distance, half_width, midline),
        bluntness=_bluntness(distance[tapered], half_width[tapered], depth),
        updown_bias=float(np.mean(midline) - y_root),
        tilt=float(tilt),
        window_depth=depth,
        n_columns=int(tapered.sum()),
    )


def _window_profiles(
    vapour: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    columns: np.ndarray,
    x_root_edge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per window column: distance inward from the root edge, half-width, and
    midline height. Columns whose vapour vanished (a hole) are dropped."""
    distance, half_width, midline = [], [], []
    for column in columns:
        rows = np.nonzero(vapour[:, column])[0]
        if rows.size == 0:
            continue
        top, bottom = float(y[rows[-1]]), float(y[rows[0]])
        distance.append(float(x[column]) - x_root_edge)
        half_width.append(0.5 * (top - bottom))
        midline.append(0.5 * (top + bottom))
    return np.asarray(distance), np.asarray(half_width), np.asarray(midline)


def _circle_rms(distance: np.ndarray, half_width: np.ndarray, midline: np.ndarray) -> float:
    """RMS radial residual of the best-fit (Kasa) circle through the window's
    edge points, as a fraction of the fitted radius."""
    points_x = np.concatenate([distance, distance])
    points_y = np.concatenate([midline + half_width, midline - half_width])
    design = np.column_stack([2 * points_x, 2 * points_y, np.ones_like(points_x)])
    target = points_x**2 + points_y**2
    (a, b, c), *_ = np.linalg.lstsq(design, target, rcond=None)
    radius = float(np.sqrt(max(c + a**2 + b**2, 1e-30)))
    residuals = np.hypot(points_x - a, points_y - b) - radius
    return float(np.sqrt(np.mean(residuals**2)) / radius)


def _bluntness(distance: np.ndarray, half_width: np.ndarray, depth: float) -> float:
    """Half-width ratio at ``BLUNTNESS_DEPTH_FRACTION * depth`` vs at ``depth``,
    interpolated on the measured profile."""
    shallow = float(np.interp(BLUNTNESS_DEPTH_FRACTION * depth, distance, half_width))
    deep = float(np.interp(depth, distance, half_width))
    if deep <= 0:
        return float("nan")
    return shallow / deep


def measured_root_fit(
    data,
    row: int,
    window: float = ROOT_WINDOW_FRACTION,
    alpha_level: float = DEFAULT_ALPHA_LEVEL,
) -> RootWindowFit:
    """The root-window fit of one dataset row's measured mask.

    The root side is the x-extent edge nearer the measured pin anchor -- the
    same orientation rule ``pin_anchor`` itself uses, so a flipped series
    measures its actual root.
    """
    vapour = (data.alpha[row] > alpha_level) & (data.valid[row] > 0)
    columns = np.nonzero(vapour.any(axis=0))[0]
    if columns.size == 0:
        raise ValueError(
            f"dataset row {row} has no vapour (alpha > {alpha_level}) to fit a "
            f"root window on -- check the dataset's masks."
        )
    x_pin, y_root = data.pin_anchor
    root_at_left = abs(float(data.x[columns[0]]) - x_pin) <= abs(
        float(data.x[columns[-1]]) - x_pin
    )
    return fit_root_window(
        vapour, data.x, data.y, y_root=y_root, window=window, root_at_left=root_at_left
    )


# Front sampling for the model-vs-mask distance. Diagnostic-grade (this runs
# once per report, not per step), matching diagnostics' cap resolution.
MODEL_ROOT_CAP_SAMPLES = 32
MODEL_ROOT_BODY_SAMPLES = 8


def model_root_window_rms(model, data, row: int) -> float:
    """How far the model's root cap sits from the measured interface: RMS of the
    measured signed-distance field sampled at the model's root-cap points, as a
    fraction of the model's own root-cap radius.

    The same normalisation the circle fit uses (distance over cap radius), so
    the model's number is directly comparable to how far the masks themselves
    sit from a circle -- a model that keeps asserting a circle bottoms out at
    the mask's own circle residual and can go no lower.
    """
    geometry = model.nets["phi"]
    device = next(geometry.parameters(), torch.empty(0)).device
    t = torch.tensor([[float(data.t[row])]], dtype=torch.float32, device=device)
    # NOT under no_grad: the front's own curvature and speed are autograd
    # derivatives, so the graph must exist even though only detached positions
    # are read here.
    front = geometry.front(t, n_body=MODEL_ROOT_BODY_SAMPLES, n_cap=MODEL_ROOT_CAP_SAMPLES)
    with torch.no_grad():
        radius = float(geometry.frame(t).r_root.squeeze())
    on_root = (front.on_cap.squeeze(1) > 0) & (front.u.squeeze(1) < 0.5)
    points = front.points[on_root].detach().cpu().numpy()
    if points.size == 0 or radius <= 0:
        return float("nan")
    distances = _sample_raster(data.sdf[row], data.x, data.y, points[:, 0], points[:, 1])
    return float(np.sqrt(np.mean(distances**2)) / radius)


def _sample_raster(
    raster: np.ndarray, x: np.ndarray, y: np.ndarray, px: np.ndarray, py: np.ndarray
) -> np.ndarray:
    """Bilinear samples of a ``(H, W)`` raster at points ``(px, py)``, clamped to
    the grid -- a root cap sits well inside the imaged domain, so clamping is a
    guard, not a distortion."""
    col = np.clip((px - x[0]) / (x[1] - x[0]), 0, x.size - 1)
    row = np.clip((py - y[0]) / (y[1] - y[0]), 0, y.size - 1)
    col_lo = np.clip(np.floor(col).astype(int), 0, x.size - 2)
    row_lo = np.clip(np.floor(row).astype(int), 0, y.size - 2)
    fx, fy = col - col_lo, row - row_lo
    return (
        raster[row_lo, col_lo] * (1 - fx) * (1 - fy)
        + raster[row_lo, col_lo + 1] * fx * (1 - fy)
        + raster[row_lo + 1, col_lo] * (1 - fx) * fy
        + raster[row_lo + 1, col_lo + 1] * fx * fy
    )


def root_report(model, data) -> dict:
    """The ``physics.root`` block of ``metrics.json``: the measured root-window
    fit and the model-vs-mask root distance, per frame and summarised.

    Present for every front-geometry run -- the measurement needs no flag, and
    a baseline that cannot see the root cannot be compared against a run that
    reshapes it.
    """
    per_frame = []
    for row, frame in enumerate(data.frame_numbers):
        fit = measured_root_fit(data, row)
        per_frame.append(
            {
                "frame": int(frame),
                "taper_exponent": fit.taper_exponent,
                "circle_rms": fit.circle_rms,
                "bluntness": fit.bluntness,
                "updown_bias": fit.updown_bias,
                "tilt": fit.tilt,
                "model_root_rms": model_root_window_rms(model, data, row),
            }
        )
    return {
        "window_fraction": ROOT_WINDOW_FRACTION,
        "bluntness_depth_fraction": BLUNTNESS_DEPTH_FRACTION,
        "taper_exponent_first": per_frame[0]["taper_exponent"],
        "taper_exponent_last": per_frame[-1]["taper_exponent"],
        "circle_rms_last": per_frame[-1]["circle_rms"],
        "model_root_rms_last": per_frame[-1]["model_root_rms"],
        "per_frame": per_frame,
    }
