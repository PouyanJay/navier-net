"""CSV exports of a run's evaluation artifacts.

Flat, analysis-ready tables built from what the pipeline already wrote
(metrics.json, trajectory*.json, the checkpoint's loss history) — no new
numbers are computed here, so a spreadsheet and the Results page can never
disagree.
"""

from __future__ import annotations

import csv
import io

from naviernet_api.services import runs as runs_service
from naviernet_api.settings import Settings


def _csv(header: list[str], rows: list[list]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue()


def iou_csv(settings: Settings, run_id: str) -> str | None:
    """Per-frame IoU in long format: one row per (dataset, frame)."""
    found = runs_service.read_dataset_and_metrics(settings, run_id)
    if found is None:
        return None
    dataset, metrics = found
    if not metrics:
        return None

    if isinstance(metrics.get("per_dataset"), dict):
        rows = _iou_rows_joint(metrics)
    else:
        rows = _iou_rows_single(dataset or run_id, metrics)
    if not rows:
        return None
    return _csv(["dataset", "camera_frame", "iou", "role"], rows)


def _iou_rows_joint(metrics: dict) -> list[list]:
    """v2 metrics: every trained dataset's frames plus the transfer frames."""
    rows: list[list] = []
    for name, block in metrics["per_dataset"].items():
        validation = set(block.get("validation_frames") or [])
        for frame, iou in sorted(
            (int(f), v) for f, v in (block.get("iou_per_frame") or {}).items()
        ):
            role = "validation" if frame in validation else "supervised"
            rows.append([name, frame, iou, role])
    transfer = (metrics.get("transfer") or {}).get("per_frame") or {}
    for name, per_frame in transfer.items():
        for frame, iou in sorted((int(f), v) for f, v in per_frame.items()):
            rows.append([name, frame, iou, "transfer"])
    return rows


def _iou_rows_single(dataset: str, metrics: dict) -> list[list]:
    """v1 metrics: one dataset's frames with holdout/validation roles."""
    holdout = metrics.get("holdout_frame")
    validation = set(metrics.get("validation_frames") or [])
    rows: list[list] = []
    for frame, iou in sorted(
        (int(f), v) for f, v in (metrics.get("iou_per_frame") or {}).items()
    ):
        role = (
            "holdout"
            if frame == holdout
            else "validation"
            if frame in validation
            else "supervised"
        )
        rows.append([dataset, frame, iou, role])
    return rows


def trajectory_csv(settings: Settings, run_id: str, dataset: str | None) -> str | None:
    """Kinematics in long format: one row per (series, instant)."""
    trajectory = runs_service.read_trajectory(settings, run_id, dataset)
    if trajectory is None:
        return None

    rows: list[list] = []
    for i, t in enumerate(trajectory.get("t_ms") or []):
        rows.append(["pinn", t, trajectory["nose_um"][i], trajectory["area_um2"][i]])
    measured = trajectory.get("measured") or {}
    for i, t in enumerate(measured.get("t_ms") or []):
        rows.append(["measured", t, measured["nose_um"][i], measured["area_um2"][i]])
    return _csv(["series", "t_ms", "nose_um", "area_um2"], rows)


def loss_csv(settings: Settings, run_id: str) -> str | None:
    """The checkpoint's loss history, one row per logged step."""
    history = runs_service.read_loss_history(settings, run_id)
    if not history:
        return None
    keys = ["step", "lr"] + sorted(
        {key for record in history for key in record} - {"step", "lr"}
    )
    rows = [[record.get(key) for key in keys] for record in history]
    return _csv(keys, rows)
