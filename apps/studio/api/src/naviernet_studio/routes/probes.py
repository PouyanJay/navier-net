"""Probes: lines, cross-sections and points, evaluated at several instants."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from naviernet_studio import catalogue, exports, probes, sampler
from naviernet_studio.routes.common import format_query, respond, scene_or_404
from naviernet_studio.scene import Scene

router = APIRouter(prefix="/api/runs", tags=["probes"])


@router.post("/{run}/probe")
def post_probe(
    request: probes.ProbeRequest,
    fmt: str = Depends(format_query),
    scene: Scene = Depends(scene_or_404),
):
    """Sample point quantities along a probe at each named frame or instant.

    The stations are recomputed per instant for sections tied to the bubble
    (the gap cross-section follows the nose); a free line or point set keeps
    the same stations throughout.
    """
    ids = _ids_or_4xx(scene, request.quantities)
    try:
        instants = request.instants(scene)
    except LookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    frames_out = []
    rows: list[list] = []
    for frame, t_star in instants:
        try:
            stations = probes.stations_for(scene, request, t_star)
        except (ValueError, LookupError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        xs, ys = stations.star(scene)
        values = sampler.evaluate(scene, sampler.points_at(xs, ys, t_star), ids)
        t_ms = t_star * scene.scales.time_ms
        frames_out.append(
            {
                "frame": frame,
                "t_ms": t_ms,
                "t_star": t_star,
                "stations": {
                    "label": stations.label,
                    "s_um": stations.s_um.tolist(),
                    "x_um": stations.x_um.tolist(),
                    "y_um": stations.y_um.tolist(),
                },
                "values": {i: values[i].tolist() for i in ids},
            }
        )
        for k in range(len(stations.s_um)):
            rows.append(
                [frame, t_ms, k, stations.s_um[k], stations.x_um[k], stations.y_um[k]]
                + [float(values[i][k]) for i in ids]
            )
    payload = {
        "kind": request.kind,
        "section": request.section,
        "units": {i: catalogue.BY_ID[i].unit for i in ids},
        "frames": frames_out,
        "provenance": exports.provenance(scene, ids, probe=request.model_dump()),
    }
    columns = ["frame", "t_ms", "station", "s_um", "x_um", "y_um", *ids]
    return respond(fmt, payload, (columns, rows), f"{scene.run_id}_probe")


def _ids_or_4xx(scene: Scene, ids: list[str]) -> list[str]:
    unknown = catalogue.unknown(ids)
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown quantities: {', '.join(unknown)}")
    wrong = [i for i in ids if i not in catalogue.POINT_IDS]
    if wrong:
        raise HTTPException(status_code=422, detail=f"not point quantities: {', '.join(wrong)}")
    missing = catalogue.unavailable(ids, scene)
    if missing:
        raise HTTPException(
            status_code=404, detail=f"this run cannot produce: {', '.join(missing)}"
        )
    return ids
