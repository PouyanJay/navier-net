"""Physics-validation summary.

Composes the numbers the Results view compares against the physics; inferred vs
measured nose speed, Bretherton film, and the key dimensionless groups; out of
what the pipeline already wrote. The API performs no physics itself.
"""

from __future__ import annotations

from naviernet_api.models import PhysicsValidation

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
    )
