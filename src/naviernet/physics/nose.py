"""Nose-window measurement: what the masks say about the bubble's advancing head.

The root journey closed the root cap's blind spot; this module closes the
nose's -- the one remaining piece of the interface that is a hard-coded
perfect circle. Measured on Series-1, the head is genuinely non-circular and
only late: the nose-window bluntness drifts 0.61 -> 1.83 through the event,
CROSSING 1.0. Bluntness above 1 means the head OVERHANGS -- it is wider a
quarter-window in than at the window edge, a mushroomed head that a circle
cannot draw at any radius (the taper exponent goes negative with it). The
model's nose error appears exactly where that bulge does (RMS 0.02 early,
0.19 at the last frame), and the trained pressure already demands a
non-circular nose (curvature gap 0.94 at the nose cap vs 0.23 on the body) --
so unlike the root story, a physics drive exists before any freedom does.

Everything here is a thin mirror of :mod:`naviernet.physics.root` -- the
window fit is `fit_root_window` read from the OTHER extent edge, and the
model-vs-mask distance samples the NOSE cap normalised by its own radius.
Shared machinery, not copies: the root/nose difference is which edge anchors
the window, so the fit itself is imported, never re-derived.
"""

from __future__ import annotations

import numpy as np
import torch

from naviernet.physics.root import (
    BLUNTNESS_DEPTH_FRACTION,
    DEFAULT_ALPHA_LEVEL,
    MODEL_ROOT_BODY_SAMPLES,
    MODEL_ROOT_CAP_SAMPLES,
    ROOT_WINDOW_FRACTION,
    Raster,
    RootWindowFit,
    fit_root_window,
    frame_time,
)


def _vapour_columns(data, row: int, alpha_level: float) -> tuple[np.ndarray, np.ndarray]:
    """One dataset row's vapour mask and its occupied columns, shared by the
    fit wrapper and the edge reader so the two cannot disagree on the mask."""
    mask = (data.alpha[row] > alpha_level) & (data.valid[row] > 0)
    columns = np.nonzero(mask.any(axis=0))[0]
    if columns.size == 0:
        raise ValueError(
            f"dataset row {row} has no vapour (alpha > {alpha_level}) to fit a "
            f"nose window on -- check the dataset's masks."
        )
    return mask, columns


def _nose_at_left(data, columns: np.ndarray) -> bool:
    """Whether the NOSE is the left x-extent edge: the exact complement of the
    root's rule -- the nose is the edge FARTHER from the measured pin anchor,
    so a flipped series measures its actual nose with no special-casing."""
    x_pin, _ = data.pin_anchor
    return abs(float(data.x[columns[0]]) - x_pin) > abs(float(data.x[columns[-1]]) - x_pin)


def measured_nose_fit(
    data,
    row: int,
    window: float = ROOT_WINDOW_FRACTION,
    alpha_level: float = DEFAULT_ALPHA_LEVEL,
) -> RootWindowFit:
    """The nose-window fit of one dataset row's measured mask: the root fit
    read from the opposite edge. Bluntness legitimately exceeds 1 here (the
    overhang) -- nothing downstream may assume ratios <= 1."""
    mask, columns = _vapour_columns(data, row, alpha_level)
    _, y_root = data.pin_anchor
    return fit_root_window(
        Raster(mask, data.x, data.y),
        y_root=y_root,
        window=window,
        root_at_left=_nose_at_left(data, columns),
    )


def measured_nose_edge(data, row: int, alpha_level: float = DEFAULT_ALPHA_LEVEL) -> float:
    """The measured nose edge's x position -- the moving anchor the nose
    window is measured inward from. The root's counterpart is the fixed pin;
    the nose ADVANCES, so its window anchor is per frame."""
    _, columns = _vapour_columns(data, row, alpha_level)
    edge = columns[0] if _nose_at_left(data, columns) else columns[-1]
    return float(data.x[edge])


def model_nose_window_rms(model, data, row: int) -> float:
    """How far the model's nose cap sits from the measured interface: RMS of
    the measured signed-distance field sampled at the model's nose-cap points,
    as a fraction of the model's own NOSE-cap radius -- `model_root_window_rms`
    mirrored, so the number is directly comparable to how far the masks
    themselves sit from a circle at the same end."""
    geometry = model.nets["phi"]
    t = frame_time(geometry, data, row)
    # NOT under no_grad: the front's own curvature and speed are autograd
    # derivatives, so the graph must exist even though only detached positions
    # are read here.
    front = geometry.front(t, n_body=MODEL_ROOT_BODY_SAMPLES, n_cap=MODEL_ROOT_CAP_SAMPLES)
    with torch.no_grad():
        radius = float(geometry.frame(t).r_nose.squeeze())
    on_nose = (front.on_cap.squeeze(1) > 0) & (front.u.squeeze(1) > 0.5)
    points = front.points[on_nose].detach().cpu().numpy()
    if points.size == 0 or radius <= 0:
        return float("nan")
    distances = Raster(data.sdf[row], data.x, data.y).sample(points[:, 0], points[:, 1])
    return float(np.sqrt(np.mean(distances**2)) / radius)


def nose_report(model, data) -> dict:
    """The ``physics.nose`` block of ``metrics.json``: the measured nose-window
    fit and the model-vs-mask nose distance, per frame and summarised -- plus,
    for a nose-shape run, the trained DOF itself: w_nose(t) and its budget use
    (the mechanism gate's own number).

    Present for every front-geometry run, like ``physics.root``: the
    measurement needs no flag, and a baseline that cannot see the nose cannot
    be compared against a run that reshapes it. The summary leads with the
    bluntness endpoints because the DRIFT (0.61 -> 1.83, crossing 1.0) is the
    measured fact this journey exists for.
    """
    per_frame = [
        _nose_frame_entry(model, data, row, frame)
        for row, frame in enumerate(data.frame_numbers)
    ]
    report = {
        "window_fraction": ROOT_WINDOW_FRACTION,
        "bluntness_depth_fraction": BLUNTNESS_DEPTH_FRACTION,
        "bluntness_first": per_frame[0]["bluntness"],
        "bluntness_last": per_frame[-1]["bluntness"],
        "circle_rms_last": per_frame[-1]["circle_rms"],
        "model_nose_rms_last": per_frame[-1]["model_nose_rms"],
        "per_frame": per_frame,
    }
    report.update(_trained_nose_summary(model, per_frame))
    return report


def _nose_frame_entry(model, data, row: int, frame: int) -> dict:
    """One frame's measured fit plus, on a nose-shape run, the trained DOF at
    that frame's own time."""
    geometry = model.nets["phi"]
    fit = measured_nose_fit(data, row)
    entry = {
        "frame": int(frame),
        "taper_exponent": fit.taper_exponent,
        "circle_rms": fit.circle_rms,
        "bluntness": fit.bluntness,
        "updown_bias": fit.updown_bias,
        "tilt": fit.tilt,
        "model_nose_rms": model_nose_window_rms(model, data, row),
    }
    if getattr(geometry, "nose_shape", False):
        t = frame_time(geometry, data, row)
        with torch.no_grad():
            entry["w"] = float(geometry.nose_bluntness(t))
    return entry


def _trained_nose_summary(model, per_frame: list[dict]) -> dict:
    """The nose-shape run's summary fields: the bluntness budget and its use
    -- the §6 mechanism gate reads ``w_budget_use`` directly."""
    geometry = model.nets["phi"]
    if not getattr(geometry, "nose_shape", False):
        return {}
    budget = float(getattr(geometry, "nose_delta", 0.0))
    peak = max(abs(f["w"]) for f in per_frame)
    return {
        "nose_delta": budget,
        "w_budget_use": peak / budget if budget > 0 else 0.0,
    }
