"""Shared route helpers: resolving a run to its scene, naming instants, and
answering in JSON or CSV."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from naviernet_studio import catalogue, exports
from naviernet_studio.scene import Scene, get_scene
from naviernet_studio.settings import Settings, get_settings

FORMAT_JSON = "json"
FORMAT_CSV = "csv"


def scene_or_404(run: str, settings: Settings = Depends(get_settings)) -> Scene:
    scene = get_scene(settings, run)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"run {run!r} has no loadable checkpoint")
    return scene


def instant_or_422(scene: Scene, frame: int | None, t_ms: float | None) -> float:
    try:
        return scene.t_star_of(frame, t_ms)
    except LookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def quantities_or_4xx(scene: Scene, raw: str, allowed: frozenset[str]) -> list[str]:
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    if not ids:
        raise HTTPException(status_code=422, detail="name at least one quantity with q=")
    unknown = [i for i in ids if i not in catalogue.BY_ID]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown quantities: {', '.join(unknown)}")
    wrong_kind = [i for i in ids if i not in allowed]
    if wrong_kind:
        raise HTTPException(
            status_code=422, detail=f"not valid for this endpoint: {', '.join(wrong_kind)}"
        )
    missing = catalogue.unavailable(ids, scene)
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"this run cannot produce: {', '.join(missing)} (see the catalogue's availability)",
        )
    return ids


def format_query(fmt: str = Query(default=FORMAT_JSON, pattern="^(json|csv)$")) -> str:
    return fmt


def respond(
    fmt: str, payload: dict, table: tuple[list[str], list[list]], filename: str
) -> Response:
    """JSON (NaN as null) or CSV with the payload's provenance as comments."""
    if fmt == FORMAT_CSV:
        text = exports.csv_text(table[0], table[1], payload.get("provenance", {}))
        return Response(
            content=text,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}.csv"'},
        )
    return JSONResponse(content=exports.without_nan(payload))
