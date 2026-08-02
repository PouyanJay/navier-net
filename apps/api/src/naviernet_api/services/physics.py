"""Physics-validation summary.

Composes the numbers the Results view compares against the physics; inferred vs
measured nose speed, Bretherton film, and the key dimensionless groups; out of
what the pipeline already wrote. The API performs no physics itself.
"""

from __future__ import annotations

import math

from naviernet_api.models import DatasetAgreement, PhysicsDiagnostics, PhysicsValidation

# Measured nose speed (mm/s) reported for each dataset. An experimental datum from
# the project README ("inferred 177 mm/s vs 180 mm/s measured"); it is not
# derivable from the artifacts, so it is recorded here (cited) until it lives in
# the experiment config alongside q_wall, flow_rate, etc.
MEASURED_NOSE_SPEED_MM_S: dict[str, float] = {"highest_t": 180.0}


def build_validation(
    dataset: str | None, metrics: dict | None, groups: dict | None
) -> PhysicsValidation:
    """Assemble the validation summary; every field is None when unavailable."""
    metrics = metrics or {}
    groups = groups or {}

    inferred = metrics.get("nose_speed_mm_s")
    measured = MEASURED_NOSE_SPEED_MM_S.get(dataset or "")
    error_pct = None
    if inferred is not None and measured:  # measured is a positive speed or absent
        error_pct = abs(inferred - measured) / measured * 100.0

    transfer = metrics.get("transfer") or {}
    per_dataset = _parse_per_dataset(metrics)

    return PhysicsValidation(
        nose_speed_inferred_mm_s=inferred,
        nose_speed_measured_mm_s=measured,
        nose_speed_error_pct=error_pct,
        bretherton_film_um=groups.get("bretherton_film_um"),
        hele_shaw=groups.get("hele_shaw"),
        reynolds=groups.get("Re"),
        weber=groups.get("We"),
        capillary=groups.get("Ca"),
        prandtl=groups.get("Pr"),
        iou_mean=metrics.get("iou_mean"),
        iou_holdout=metrics.get("iou_holdout"),
        holdout_frame=metrics.get("holdout_frame"),
        val_iou_mean=metrics.get("val_iou_mean"),
        iou_val=metrics.get("iou_val"),
        validation_frames=metrics.get("validation_frames") or [],
        transfer_iou_mean=transfer.get("mean"),
        transfer_per_dataset=transfer.get("per_dataset"),
        transfer_per_frame=transfer.get("per_frame"),
        per_dataset=per_dataset,
        training_datasets=metrics.get("training_datasets"),
        heldout_datasets=metrics.get("heldout_datasets"),
        physics=_parse_physics(metrics),
    )


def _parse_physics(metrics: dict) -> PhysicsDiagnostics | None:
    """The run's physics block, validated. ``None`` for a run without an explicit
    front, and NaNs dropped: JSON has no NaN, and a metric that could not be
    measured must read as absent rather than as a number the UI would plot.
    """
    block = metrics.get("physics")
    if not isinstance(block, dict):
        return None
    finite = {
        key: value
        for key, value in block.items()
        if not (isinstance(value, float) and not math.isfinite(value))
    }
    return PhysicsDiagnostics.model_validate(finite)


def _parse_per_dataset(metrics: dict) -> dict[str, DatasetAgreement] | None:
    """A joint run's per-dataset agreement blocks, validated; None on v1 runs."""
    raw = metrics.get("per_dataset")
    if not isinstance(raw, dict):
        return None
    return {
        name: DatasetAgreement.model_validate(block)
        for name, block in raw.items()
        if isinstance(block, dict)
    }
