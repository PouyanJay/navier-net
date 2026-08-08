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
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from naviernet.data.dataset import edge_pair
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
    ious = {
        data.frame_numbers[row]: frame_iou(cfg, model, data, row, c) for row in range(n_event)
    }
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


def nose_trajectory(cfg, model, data, c=None) -> GrowthTrajectory:
    """Continuous nose position and projected vapour area over time (t*).

    The final camera frame is excluded from the time span because its bubble
    is cut by the field of view. A timestep whose predicted mask is empty
    yields ``nan`` for the nose position. ``c`` is the dataset's conditioning
    row for a joint, condition-aware model; ``None`` for a plain one.
    """
    stride = cfg.evaluation.stride
    threshold = cfg.evaluation.threshold
    xs = data.x[::stride]
    times = np.linspace(data.t[0], data.t[-2], cfg.evaluation.n_traj_points)

    nose, area = [], []
    for t in times:
        mask = predict_alpha(model, data, t, stride, c) > threshold
        columns = np.where(mask.any(axis=0))[0]
        nose.append(xs[columns.max()] if len(columns) else np.nan)
        area.append(mask.mean() * data.domain.area)
    return GrowthTrajectory(times, np.asarray(nose), np.asarray(area))


def root_position(mask: np.ndarray, xs: np.ndarray, x_anchor: float) -> float:
    """The bubble-root x* of one mask (``nan`` when empty). The root staying at
    the anchor on held-out frames is the hard pin's direct mechanistic check."""
    edges = edge_pair(mask, xs, x_anchor)
    return float("nan") if edges is None else edges[0]


def front_position(mask: np.ndarray, xs: np.ndarray, x_anchor: float) -> float:
    """The bubble-front x* of one mask (``nan`` when empty). The late-window
    front undershoot is the failure the kinematic growth constraints target."""
    edges = edge_pair(mask, xs, x_anchor)
    return float("nan") if edges is None else edges[1]


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


def _physics_block(cfg, model, data, paths: RunPaths) -> dict | None:
    """The physics diagnostics, or ``None`` for a run that has no explicit front.

    IoU alone hid a ~55% violated force balance on the R3 baseline, so these
    travel with every ``metrics.json`` a front-geometry run writes: the
    Young-Laplace jump error, the drainage drive, the neck against the measured
    masks, and whether each physics residual actually descended.

    The loss history comes from the checkpoint rather than being threaded through
    the pipeline: ``evaluate`` is also a standalone stage, so the run's own record
    on disk is the only source that is right in both cases.
    """
    if not getattr(model, "front_geometry", False):
        return None
    from naviernet.physics import diagnostics, registry

    block = diagnostics.physics_report(model, data)
    equations = registry.enabled_equations(
        cfg.model.fields,
        bool(getattr(cfg.model, "sharp_interface", False)),
        bool(getattr(cfg.model, "liquid_film", False)),
    )
    ckpt = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state", {})
    block["residual_convergence"] = diagnostics.residual_convergence(
        state.get("hist", []),
        registry.stage_b_terms(equations),
        # The warm-up the RUN used, not the one this invocation happens to
        # compose: a standalone evaluate need not carry the launch override, and
        # reading the wrong one averages the window where the term was inert.
        int(state.get("stage_b_warmup_steps", cfg.training.stage_b_warmup_steps)),
    )
    return block


def evaluate(cfg, model, data, paths: RunPaths) -> dict:
    """Full evaluation report; also written to ``metrics.json`` in the run dir."""
    paths.ensure()  # artifacts below need the run directory to exist

    ious_report = iou_report(cfg, model, data)
    ious = ious_report["iou_per_frame"]

    # In-distribution validation IoU over the frames held out of supervision (the
    # axis-A split and/or the legacy holdout frame), so a single-dataset run with a
    # validation split still surfaces the metric rather than silently dropping it.
    iou_val, validation_frames = _validation_iou(ious_report, data)

    predicted = nose_trajectory(cfg, model, data)
    _write_trajectory(cfg, data, paths.trajectory_json, predicted)
    _write_front_velocity(cfg, model, data, paths.front_velocity_json)
    speed = np.gradient(predicted.nose, predicted.times)
    # Trim the ends, where one-sided differences and the pinned start distort
    # the estimate, and average over the steady middle of the growth.
    middle = slice(len(predicted.times) // 5, 4 * len(predicted.times) // 5)
    mean_speed_star = float(speed[middle].mean())

    report = {
        "run_name": cfg.run_name,
        "dataset": cfg.dataset,
        **ious_report,
        "iou_val": iou_val,
        "validation_frames": validation_frames,
        "nose_speed_star": mean_speed_star,
        "nose_speed_mm_s": mean_speed_star * cfg.scales.U_ref * 1e3,
        "physics": _physics_block(cfg, model, data, paths),
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


def _validation_iou(report: dict, data) -> tuple[float | None, list[int]]:
    """In-distribution validation IoU for one training dataset: the mean IoU over
    the frames held out of supervision (the split plus the legacy holdout frame),
    and those frame numbers. ``None`` when the dataset held nothing out."""
    frames = data.validation_frames
    per_frame = report["iou_per_frame"]
    ious = [per_frame[f] for f in frames if f in per_frame]
    return (float(np.mean(ious)) if ious else None), frames


def evaluate_joint(cfg, model, contexts, paths: RunPaths, heldout_datasets=None) -> dict:
    """Evaluate a joint (transfer-learning) run into one metrics.json (v2), each
    dataset scored with its own conditioning row along two validation axes:

    - **In-distribution** (``iou_val``): the *training* datasets' held-out frames
      -- unseen time instants of a condition the model trained on.
    - **Transfer** (``transfer``): every frame of the *held-out* datasets (axis B)
      -- conditions the model never trained on, predicted from conditioning alone.

    ``contexts`` are every dataset the run spans (training + held-out).
    ``heldout_datasets`` is the split the model was *trained* with -- pass the value
    recorded in its checkpoint (via :func:`~naviernet.training.load_joint`) so a
    standalone ``stage=evaluate`` classifies each dataset by how it was actually
    trained, not by whatever ``cfg`` a re-run happens to compose. Falls back to
    ``cfg.heldout_datasets`` only when a checkpoint predates that record.
    """
    paths.ensure()

    heldout = set(cfg.heldout_datasets if heldout_datasets is None else heldout_datasets)
    # Each dataset scores through its bound view: its conditioning row and -- on a
    # hard-pin run -- its own root anchor (an unbound hard-pin call raises).
    reports = {
        cx.name: iou_report(cfg, model.bound(cx.c, pin=cx.pin, geometry=cx.geometry), cx.data)
        for cx in contexts
    }

    per_dataset: dict[str, dict] = {}
    transfer: dict[str, float] = {}
    for cx in contexts:
        rep = reports[cx.name]
        if cx.name in heldout:
            # Held out of training entirely: every frame is a transfer prediction.
            transfer[cx.name] = rep["iou_mean"]
        else:
            iou_val, validation_frames = _validation_iou(rep, cx.data)
            per_dataset[cx.name] = {
                "iou_mean": rep["iou_mean"],
                "iou_val": iou_val,
                "validation_frames": validation_frames,
                "iou_per_frame": rep["iou_per_frame"],
            }

    val_ious = [d["iou_val"] for d in per_dataset.values() if d["iou_val"] is not None]
    report = {
        "run_name": cfg.run_name,
        "datasets": [cx.name for cx in contexts],
        "training_datasets": [cx.name for cx in contexts if cx.name not in heldout],
        "heldout_datasets": [cx.name for cx in contexts if cx.name in heldout],
        "per_dataset": per_dataset,
        "iou_mean": float(np.mean([d["iou_mean"] for d in per_dataset.values()])),
        "val_iou_mean": float(np.mean(val_ious)) if val_ious else None,
    }
    if transfer:
        report["transfer"] = {
            "per_dataset": transfer,
            "mean": float(np.mean(list(transfer.values()))),
            # Per-frame transfer scores too: the platform charts how agreement
            # on a never-trained condition evolves over the event.
            "per_frame": {name: reports[name]["iou_per_frame"] for name in transfer},
        }

    _write_joint_trajectories(cfg, model, contexts, paths)

    for name, d in per_dataset.items():
        log.info("dataset %s: IoU mean %.3f, val %s", name, d["iou_mean"], d["iou_val"])
    for name, iou in transfer.items():
        log.info("held-out %s: transfer IoU %.3f (never trained)", name, iou)
    paths.metrics_json.write_text(json.dumps(report, indent=2))
    log.info("joint metrics for %d datasets written to %s", len(contexts), paths.metrics_json)
    return report


def _write_joint_trajectories(cfg, model, contexts, paths: RunPaths) -> None:
    """Per-dataset growth kinematics, held-out conditions included: transfer
    kinematics are evidence too."""
    for cx in contexts:
        bound = model.bound(cx.c, pin=cx.pin, geometry=cx.geometry)
        _write_trajectory(
            cfg,
            cx.data,
            paths.trajectory_json_for(cx.name),
            nose_trajectory(cfg, bound, cx.data),
        )
        _write_front_velocity(cfg, bound, cx.data, paths.front_velocity_json_for(cx.name))


def _write_front_velocity(cfg, model, data, path: Path) -> None:
    """The front's motion, beside its position.

    Imported here rather than at module scope: the report is built ON this
    module's trajectories and unit helpers, so a top-level import would close a
    cycle. Same deferral :func:`_physics_block` makes for the diagnostics.
    """
    from naviernet import front_kinematics

    front_kinematics.write_report(cfg, model, data, path)


def physical_series(values, factor: float, digits: int) -> list[float | None]:
    """Scale an array into physical units; NaN becomes None (JSON has no NaN,
    and a bare ``NaN`` token would break every standards-compliant consumer)."""
    return [None if math.isnan(v) else round(float(v) * factor, digits) for v in values]


def reference_length_um(cfg, data) -> float:
    """The dataset's own reference length; the composed cfg is the fallback for
    tensors that predate the ``L_ref_um`` record (exact for the primary dataset,
    approximate otherwise — logged so a re-preprocess can fix it)."""
    l_ref_um = data.meta.get("L_ref_um")
    if l_ref_um is None:
        l_ref_um = cfg.scales.L_ref_um
        log.warning(
            "tensors for %s record no L_ref_um; scaling with the composed cfg "
            "value (%.1f µm) — re-preprocess to fix",
            data.meta.get("dataset", "?"),
            l_ref_um,
        )
    return float(l_ref_um)


def _write_trajectory(cfg, data, path: Path, predicted: GrowthTrajectory) -> None:
    """Persist the continuous and measured growth kinematics as data.

    The same arrays the trajectory figure plots, in physical units (each
    dataset scaled by its own reference length), so the platform can chart
    them interactively instead of reading a rendered PNG.
    """
    l_ref_um = reference_length_um(cfg, data)
    t_ref_ms = float(data.meta["t_ref_ms"])
    measured = measured_trajectory(cfg, data)
    payload = {
        "t_ms": physical_series(predicted.times, t_ref_ms, 4),
        "nose_um": physical_series(predicted.nose, l_ref_um, 2),
        "area_um2": physical_series(predicted.area, l_ref_um * l_ref_um, 1),
        "measured": {
            "t_ms": physical_series(measured.times, 1.0, 4),  # already in ms
            "nose_um": physical_series(measured.nose, l_ref_um, 2),
            "area_um2": physical_series(measured.area, l_ref_um * l_ref_um, 1),
        },
    }
    path.write_text(json.dumps(payload, allow_nan=False))
    log.info("trajectory written to %s", path)
