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

import numpy as np
import torch

from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths

log = get_logger(__name__)


@torch.no_grad()
def predict_alpha(model, data, t_star: float, stride: int = 4) -> np.ndarray:
    """Volume fraction on a strided pixel grid at an arbitrary time."""
    points, _, shape = data.frame_grid(0, stride)
    points = points.clone()
    points[:, 2] = float(t_star)  # any time, not just a camera instant
    return model.alpha(points).cpu().numpy().reshape(shape)


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


def frame_iou(cfg, model, data, frame: int) -> float:
    """Intersection over union between predicted and measured masks."""
    stride = cfg.evaluation.stride
    threshold = cfg.evaluation.threshold
    predicted = predict_alpha(model, data, data.t[frame], stride) > threshold
    measured = data.alpha[frame, ::stride, ::stride] > threshold
    union = (predicted | measured).sum()
    return float((predicted & measured).sum() / max(union, 1))


def nose_trajectory(cfg, model, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous nose position and projected vapour area over time.

    Returns ``(t_star, nose_x_star, area_star)``. The final camera frame is
    excluded from the time span because its bubble is cut by the field of view.
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
    return times, np.asarray(nose), np.asarray(area)


def measured_trajectory(cfg, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same quantities read straight off the segmented camera frames."""
    n_event = cfg.experiment.n_frames_event
    threshold = cfg.evaluation.threshold
    times = np.arange(n_event) * cfg.experiment.dt_frame_ms

    nose, area = [], []
    for i in range(n_event):
        mask = data.alpha[i] > threshold
        columns = np.where(mask.any(axis=0))[0]
        nose.append(data.x[columns.max()])
        area.append(mask.mean() * data.domain.area)
    return times, np.asarray(nose), np.asarray(area)


def evaluate(cfg, model, data, paths: RunPaths) -> dict:
    """Full evaluation report; also written to ``metrics.json`` in the run dir."""
    n_event = cfg.experiment.n_frames_event
    holdout = int(cfg.training.holdout_frame)

    # Keyed by 1-based physical frame number, matching the TIFF filenames.
    ious = {frame + 1: frame_iou(cfg, model, data, frame) for frame in range(n_event)}

    times, nose, _ = nose_trajectory(cfg, model, data)
    speed = np.gradient(nose, times)
    # Trim the ends, where one-sided differences and the pinned start distort
    # the estimate, and average over the steady middle of the growth.
    middle = slice(len(times) // 5, 4 * len(times) // 5)
    mean_speed_star = float(speed[middle].mean())

    report = {
        "run_name": cfg.run_name,
        "dataset": cfg.dataset,
        "iou_per_frame": ious,
        "iou_mean": float(np.mean(list(ious.values()))),
        "iou_holdout": ious[holdout + 1] if holdout >= 0 else None,
        "holdout_frame": holdout + 1 if holdout >= 0 else None,
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

    paths.ensure()
    paths.metrics_json.write_text(json.dumps(report, indent=2))
    log.info("metrics written to %s", paths.metrics_json)
    return report
