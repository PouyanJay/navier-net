"""Quantities along the bubble edge at one instant."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from naviernet_studio import catalogue, edge, exports
from naviernet_studio.routes.common import (
    format_query,
    instant_or_422,
    quantities_or_4xx,
    respond,
    scene_or_404,
)
from naviernet_studio.scene import Scene

router = APIRouter(prefix="/api/runs", tags=["edge"])

DEFAULT_BODY_SAMPLES = 160
DEFAULT_CAP_SAMPLES = 40
MAX_BODY_SAMPLES = 1000
MAX_CAP_SAMPLES = 300
ALWAYS = ("edge_x", "edge_y", "arclength", "segment")


@router.get("/{run}/edge")
def get_edge(
    q: str = Query(..., description="comma-separated edge quantity ids"),
    frame: int | None = Query(default=None),
    t_ms: float | None = Query(default=None),
    n_body: int = Query(default=DEFAULT_BODY_SAMPLES, ge=2, le=MAX_BODY_SAMPLES),
    n_cap: int = Query(default=DEFAULT_CAP_SAMPLES, ge=2, le=MAX_CAP_SAMPLES),
    fmt: str = Depends(format_query),
    scene: Scene = Depends(scene_or_404),
):
    """Per-sample edge quantities; the position, arclength and segment always
    come along so every value can be placed."""
    if not scene.has_front:
        raise HTTPException(status_code=404, detail="this run has no explicit front to sample")
    ids = quantities_or_4xx(scene, q, catalogue.EDGE_IDS)
    ids = [*ALWAYS, *(i for i in ids if i not in ALWAYS)]
    t_star = instant_or_422(scene, frame, t_ms)
    result = edge.evaluate(scene, t_star, ids, n_body, n_cap)
    values = {i: v.tolist() for i, v in result["values"].items()}
    payload = {
        "frame": scene.frame_of(t_star),
        "t_ms": t_star * scene.scales.time_ms,
        "t_star": t_star,
        "n_samples": result["n_samples"],
        "segments": result["segments"],
        "units": {i: catalogue.BY_ID[i].unit for i in ids},
        "values": values,
        "provenance": exports.provenance(scene, ids, n_body=n_body, n_cap=n_cap),
    }
    rows = [[k, *(values[i][k] for i in ids)] for k in range(result["n_samples"])]
    return respond(fmt, payload, (["sample", *ids], rows), f"{scene.run_id}_edge")
