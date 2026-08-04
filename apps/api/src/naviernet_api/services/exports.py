"""CSV exports of a run's evaluation artifacts.

Flat, analysis-ready tables built from what the pipeline already wrote
(metrics.json, trajectory*.json, the checkpoint's loss history) — no new
numbers are computed here, so a spreadsheet and the Results page can never
disagree.
"""

from __future__ import annotations

import csv
import io
from typing import NamedTuple

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


# µm/ms is the reporting unit (it makes each speed the slope of the position
# artifact beside it); this is the SI column every export carries alongside.
_M_PER_S_PER_UM_PER_MS = 1e-3


def _si(value) -> float | None:
    """Converted from the artifact's own already-rounded value, not recomputed
    from a raw float. That costs a little precision and buys the guarantee this
    module exists for: the CSV and the Results page can never disagree, because
    both read the same number."""
    return None if value is None else round(value * _M_PER_S_PER_UM_PER_MS, 9)


def front_velocity_csv(settings: Settings, run_id: str, dataset: str | None) -> str | None:
    """The front's motion in long format: one row per (series, instant, position).

    `s` is the position around the closed front for the profile series and empty
    for the whole-front quantities (nose, apex), which have no position. Both
    units on every row, so the file needs no conversion to read.
    """
    report = runs_service.read_front_velocity(settings, run_id, dataset)
    if report is None:
        return None

    rows: list[list] = []
    _append_speed_rows(rows, report)
    _append_profile_rows(rows, report.get("profile"))
    return _csv(["series", "t_ms", "s", "v_um_per_ms", "v_m_per_s", "heldout"], rows)


class _Series(NamedTuple):
    """One speed series as the report stores it: parallel arrays of instants and
    values, plus (for a measured series) whether each interval spanned a
    held-out frame. Bundled because the three only mean anything together."""

    times: list | None
    values: list | None
    heldout: list | None = None

    @classmethod
    def of(cls, block: dict | None, key: str = "v_um_per_ms") -> _Series:
        block = block or {}
        return cls(block.get("t_ms"), block.get(key), block.get("heldout"))


def _append_speed_rows(rows: list[list], report: dict) -> None:
    """The whole-front speeds: the nose, and each apex component."""
    nose = report.get("nose_speed") or {}
    _append_series(rows, "nose_speed", _Series(nose.get("t_ms"), nose.get("v_um_per_ms")))
    _append_series(rows, "nose_speed_measured", _Series.of(nose.get("measured")))

    apex = report.get("apex")
    if not apex:
        return
    for axis in ("vx", "vy"):
        key = f"{axis}_um_per_ms"
        _append_series(rows, f"apex_{axis}", _Series(apex.get("t_ms"), apex.get(key)))
        _append_series(rows, f"apex_{axis}_measured", _Series.of(apex.get("measured"), key))


def _append_profile_rows(rows: list[list], profile: dict | None) -> None:
    """The normal-speed profile: one row per (frame pair, position along s).

    A suppressed bin is written with an empty value rather than dropped: the row
    records that the position WAS looked at and nothing trustworthy was found
    there, which is not the same as the position not existing.
    """
    if not profile:
        return
    positions = profile.get("s") or []
    for frame in profile.get("times") or []:
        for name in ("model", "measured"):
            for i, value in enumerate(frame.get(name) or []):
                rows.append(
                    [
                        f"profile_{name}",
                        frame.get("t_ms"),
                        positions[i] if i < len(positions) else None,
                        value,
                        _si(value),
                        frame.get("heldout"),
                    ]
                )


def _append_series(rows: list[list], name: str, series: _Series) -> None:
    """One (t, v) series as long-format rows, with no position."""
    for i, t in enumerate(series.times or []):
        value = (series.values or [])[i]
        held = None if series.heldout is None else series.heldout[i]
        rows.append([name, t, None, value, _si(value), held])


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
