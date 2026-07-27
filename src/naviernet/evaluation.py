"""Evaluation: agreement with the measured masks and kinematic consistency.

The headline number is the IoU on the holdout frame, which is never supervised
at any point in training -- it is the only frame whose agreement is not partly a
statement about how well the network memorised its targets.

The kinematic checks are independent of the segmentation entirely: nose speed
inferred from the continuous reconstruction can be compared against the speed
measured off the raw frames, and neither quantity was ever given to the model.
"""

from __future__ import annotations

import json
import math
from typing import NamedTuple

import numpy as np
import torch

from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths

log = get_logger(__name__)


def _context(c, points) -> torch.Tensor | None:
    """Broadcast a dataset's conditioning row to a batch of points (``None`` for
    an unconditioned model)."""
    return None if c is None else c.expand(points.shape[0], -1)


@torch.no_grad()
def predict_alpha(model, data, t_star: float, stride: int = 4, c=None) -> np.ndarray:
    """Volume fraction on a strided pixel grid at an arbitrary time.

    ``c`` is the dataset's conditioning row when the model is a joint,
    condition-aware one; ``None`` for a plain single-dataset model.
    """
    points, _, shape = data.frame_grid(0, stride)
    points = points.clone()
    points[:, 2] = float(t_star)  # any time, not just a camera instant
    return model.alpha(points, _context(c, points)).cpu().numpy().reshape(shape)


@torch.no_grad()
def predict_alpha_fullres(model, data, t_star: float, chunk: int = 45_000) -> np.ndarray:
    """Volume fraction at full pixel resolution, evaluated in chunks."""
    _, height, width = data.alpha.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    points = np.stack(
        [data.x[xx.ravel()], data.y[yy.ravel()], np.full(xx.size, float(t_star))],
        axis=1,
    ).astype(np.float32)

    predictions = [
        model.alpha(torch.tensor(part, device=data.device)).cpu().numpy()
        for part in np.array_split(points, max(1, len(points) // chunk))
    ]
    return np.concatenate(predictions).reshape(height, width)


def frame_iou(cfg, model, data, frame: int, c=None) -> float:
    """Intersection over union between predicted and measured masks."""
    stride = cfg.evaluation.stride
    threshold = cfg.evaluation.threshold
    predicted = predict_alpha(model, data, data.t[frame], stride, c) > threshold
    measured = data.alpha[frame, ::stride, ::stride] > threshold
    union = (predicted | measured).sum()
    return float((predicted & measured).sum() / max(union, 1))


def iou_report(cfg, model, data, c=None) -> dict:
    """Per-frame IoU, its mean, and the held-out frame's IoU for one dataset.

    Shared by single-dataset :func:`evaluate` (``c=None``) and the joint report
    (one call per dataset, each with its conditioning row), so the headline
    generalisation metric is computed identically either way.
    """
    n_event = data.n_event
    holdout_row = data.holdout_row
    holdout = data.frame_numbers[holdout_row] if holdout_row >= 0 else None
    ious = {data.frame_numbers[row]: frame_iou(cfg, model, data, row, c) for row in range(n_event)}
    return {
        "iou_per_frame": ious,
        "iou_mean": float(np.mean(list(ious.values()))),
        "iou_holdout": ious.get(holdout) if holdout is not None else None,
        "holdout_frame": holdout,
    }


class GrowthTrajectory(NamedTuple):
    """Nose position and projected vapour area over a set of times.

    ``times`` is t* for the predicted trajectory and milliseconds for the
    measured one (each producer documents its convention); ``nose``/``area``
    are dimensionless (x*, area*).
    """

    times: np.ndarray
    nose: np.ndarray
    area: np.ndarray


def nose_trajectory(cfg, model, data) -> GrowthTrajectory:
    """Continuous nose position and projected vapour area over time (t*).

    The final camera frame is excluded from the time span because its bubble
    is cut by the field of view. A timestep whose predicted mask is empty
    yields ``nan`` for the nose position.
    """
    stride = cfg.evaluation.stride
    threshold = cfg.evaluation.threshold
    xs = data.x[::stride]
    times = np.linspace(data.t[0], data.t[-2], cfg.evaluation.n_traj_points)

    nose, area = [], []
    for t in times:
        mask = predict_alpha(model, data, t, stride) > threshold
        columns = np.where(mask.any(axis=0))[0]
        nose.append(xs[columns.max()] if len(columns) else np.nan)
        area.append(mask.mean() * data.domain.area)
    return GrowthTrajectory(times, np.asarray(nose), np.asarray(area))


def measured_trajectory(cfg, data) -> GrowthTrajectory:
    """The same quantities read straight off the segmented camera frames (ms)."""
    n_event = data.n_event
    threshold = cfg.evaluation.threshold
    # Each row's own acquisition time, so an excluded frame leaves a gap on the
    # axis instead of shifting every later measurement earlier.
    times = np.asarray(data.t[:n_event]) * float(data.meta["t_ref_ms"])

    nose, area = [], []
    for i in range(n_event):
        mask = data.alpha[i] > threshold
        columns = np.where(mask.any(axis=0))[0]
        nose.append(data.x[columns.max()])
        area.append(mask.mean() * data.domain.area)
    return GrowthTrajectory(times, np.asarray(nose), np.asarray(area))


def evaluate(cfg, model, data, paths: RunPaths) -> dict:
    """Full evaluation report; also written to ``metrics.json`` in the run dir."""
    paths.ensure()  # artifacts below need the run directory to exist

    ious_report = iou_report(cfg, model, data)
    ious = ious_report["iou_per_frame"]

    predicted = nose_trajectory(cfg, model, data)
    _write_trajectory(cfg, data, paths, predicted)
    speed = np.gradient(predicted.nose, predicted.times)
    # Trim the ends, where one-sided differences and the pinned start distort
    # the estimate, and average over the steady middle of the growth.
    middle = slice(len(predicted.times) // 5, 4 * len(predicted.times) // 5)
    mean_speed_star = float(speed[middle].mean())

    report = {
        "run_name": cfg.run_name,
        "dataset": cfg.dataset,
        **ious_report,
        "nose_speed_star": mean_speed_star,
        "nose_speed_mm_s": mean_speed_star * cfg.scales.U_ref * 1e3,
    }

    log.info("IoU per frame: %s", {k: round(v, 3) for k, v in ious.items()})
    if report["iou_holdout"] is not None:
        log.info(
            "holdout frame %d IoU = %.3f (never supervised)",
            report["holdout_frame"],
            report["iou_holdout"],
        )
    log.info("inferred nose speed: %.0f mm/s", report["nose_speed_mm_s"])

    paths.metrics_json.write_text(json.dumps(report, indent=2))
    log.info("metrics written to %s", paths.metrics_json)
    return report


def evaluate_joint(cfg, model, contexts, paths: RunPaths) -> dict:
    """Evaluate a joint (transfer-learning) run: each dataset scored with its own
    conditioning row, into one metrics.json with per-dataset IoU and an aggregate.

    ``contexts`` are the run's datasets, each carrying its tensors (``.data``),
    name (``.name``), and conditioning row (``.c``) — as the joint trainer built
    them. The holdout IoU is still the headline number, now reported per dataset
    and averaged, so a joint model is judged on generalisation across conditions.
    """
    paths.ensure()

    per_dataset = {cx.name: iou_report(cfg, model, cx.data, cx.c) for cx in contexts}
    holdouts = [d["iou_holdout"] for d in per_dataset.values() if d["iou_holdout"] is not None]

    report = {
        "run_name": cfg.run_name,
        "datasets": [cx.name for cx in contexts],
        "per_dataset": per_dataset,
        "iou_mean": float(np.mean([d["iou_mean"] for d in per_dataset.values()])),
        "iou_holdout_mean": float(np.mean(holdouts)) if holdouts else None,
    }

    for name, d in per_dataset.items():
        log.info("dataset %s: IoU mean %.3f, holdout %s", name, d["iou_mean"], d["iou_holdout"])
    paths.metrics_json.write_text(json.dumps(report, indent=2))
    log.info("joint metrics for %d datasets written to %s", len(contexts), paths.metrics_json)
    return report


def _scaled(values, factor: float, digits: int) -> list[float | None]:
    """Scale an array into physical units; NaN becomes None (JSON has no NaN,
    and a bare ``NaN`` token would break every standards-compliant consumer)."""
    return [None if math.isnan(v) else round(float(v) * factor, digits) for v in values]


def _write_trajectory(cfg, data, paths: RunPaths, predicted: GrowthTrajectory) -> None:
    """Persist the continuous and measured growth kinematics as data.

    The same arrays the trajectory figure plots, in physical units, so the
    platform can chart them interactively instead of reading a rendered PNG.
    """
    l_ref = float(cfg.scales.L_ref_um)
    t_ref_ms = float(data.meta["t_ref_ms"])
    measured = measured_trajectory(cfg, data)
    payload = {
        "t_ms": _scaled(predicted.times, t_ref_ms, 4),
        "nose_um": _scaled(predicted.nose, l_ref, 2),
        "area_um2": _scaled(predicted.area, l_ref * l_ref, 1),
        "measured": {
            "t_ms": _scaled(measured.times, 1.0, 4),  # already in ms
            "nose_um": _scaled(measured.nose, l_ref, 2),
            "area_um2": _scaled(measured.area, l_ref * l_ref, 1),
        },
    }
    paths.trajectory_json.write_text(json.dumps(payload, allow_nan=False))
    log.info("trajectory written to %s", paths.trajectory_json)
