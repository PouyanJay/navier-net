"""CSV and JSON exports that carry their provenance with them."""

from __future__ import annotations

import csv
import io
import math
from datetime import datetime, timezone

from naviernet_studio import catalogue
from naviernet_studio.scene import Scene


def provenance(scene: Scene, quantity_ids: list[str], **extra) -> dict:
    """What the numbers are and where they came from."""
    return {
        "run": scene.run_id,
        "dataset": str(scene.cfg.dataset),
        "checkpoint": str(scene.paths.checkpoint),
        "checkpoint_mtime": scene.checkpoint_mtime,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quantities": {
            i: {
                "label": catalogue.BY_ID[i].label,
                "unit": catalogue.BY_ID[i].unit,
                "tex": catalogue.BY_ID[i].tex,
                "how": catalogue.BY_ID[i].how,
            }
            for i in quantity_ids
            if i in catalogue.BY_ID
        },
        **extra,
    }


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else repr(value)
    return str(value)


def csv_text(columns: list[str], rows: list[list], meta: dict) -> str:
    """A CSV with provenance as leading comment lines."""
    buffer = io.StringIO()
    for key, value in meta.items():
        if isinstance(value, dict):
            continue
        buffer.write(f"# {key}: {value}\n")
    for i, q in meta.get("quantities", {}).items():
        buffer.write(f"# {i}: {q['label']} [{q['unit']}]  {q['how']}\n")
    writer = csv.writer(buffer)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(v) for v in row])
    return buffer.getvalue()


def without_nan(value):
    """NaN and infinities as null, recursively: JSON carries neither."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: without_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [without_nan(v) for v in value]
    return value
