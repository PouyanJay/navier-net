"""Integral budgets at one instant."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from naviernet_studio import budgets, exports
from naviernet_studio.routes.common import format_query, instant_or_422, respond, scene_or_404
from naviernet_studio.scene import Scene

router = APIRouter(prefix="/api/runs", tags=["budgets"])


@router.get("/{run}/budgets/gap")
def get_gap_budget(
    frame: int | None = Query(default=None),
    t_ms: float | None = Query(default=None),
    side: str = Query(default="upper", pattern="^(upper|lower)$"),
    bins: str = Query(default=",".join(str(int(b)) for b in budgets.DEFAULT_GAP_BINS_UM)),
    fmt: str = Depends(format_query),
    scene: Scene = Depends(scene_or_404),
):
    """The x-momentum budget along the middle of the gap beside the head."""
    if "p" not in scene.fields:
        raise HTTPException(status_code=404, detail="this run has no pressure field")
    try:
        edges = tuple(float(b) for b in bins.split(","))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="bins must be numbers") from exc
    if len(edges) < 2 or any(a <= b for a, b in zip(edges[:-1], edges[1:], strict=True)):
        raise HTTPException(
            status_code=422, detail="bins must decrease, e.g. 450,300,200,150,100,40"
        )
    t_star = instant_or_422(scene, frame, t_ms)
    try:
        payload = budgets.gap_budget(scene, t_star, side == "upper", edges)
    except LookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    payload.update(frame=scene.frame_of(t_star), t_ms=t_star * scene.scales.time_ms)
    payload["provenance"] = exports.provenance(
        scene, list(budgets.GAP_TERMS), bins=edges, side=side
    )
    columns = ["from_um", "to_um", *(k for k in payload["bins"][0] if k.endswith("_Pa"))]
    rows = [[b[c] for c in columns] for b in payload["bins"]]
    return respond(fmt, payload, (columns, rows), f"{scene.run_id}_gap_budget")


@router.get("/{run}/budgets/flux")
def get_flux_budget(
    frame: int | None = Query(default=None),
    t_ms: float | None = Query(default=None),
    fmt: str = Depends(format_query),
    scene: Scene = Depends(scene_or_404),
):
    """Liquid flux through every section against mass conservation."""
    t_star = instant_or_422(scene, frame, t_ms)
    payload = budgets.flux_budget(scene, t_star)
    payload.update(frame=scene.frame_of(t_star), t_ms=t_star * scene.scales.time_ms)
    payload["provenance"] = exports.provenance(scene, ["u"])
    columns = ["x_um", "q_total_mL_h", "q_liquid_mL_h", "q_required_mL_h", "defect"]
    rows = [[payload[c][k] for c in columns] for k in range(len(payload["x_um"]))]
    return respond(fmt, payload, (columns, rows), f"{scene.run_id}_flux_budget")


@router.get("/{run}/budgets/heat")
def get_heat_budget(
    frame: int = Query(...),
    fmt: str = Depends(format_query),
    scene: Scene = Depends(scene_or_404),
):
    """The latent-power estimates at a camera frame, in mW."""
    payload = budgets.heat_budget(scene, frame)
    if payload is None:
        raise HTTPException(
            status_code=422, detail=f"frame {frame} is not an event frame of this run"
        )
    payload["provenance"] = exports.provenance(scene, ["s", "s_footprint"])
    numeric = [k for k, v in payload.items() if isinstance(v, (int, float))]
    return respond(
        fmt, payload, (numeric, [[payload[k] for k in numeric]]), f"{scene.run_id}_heat_budget"
    )
