"""Field grids: one or more point quantities on a lattice at one instant."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from naviernet_studio import catalogue, exports, grids
from naviernet_studio.routes.common import (
    FORMAT_CSV,
    format_query,
    instant_or_422,
    quantities_or_4xx,
    respond,
    scene_or_404,
)
from naviernet_studio.scene import Scene

router = APIRouter(prefix="/api/runs", tags=["fields"])


@router.get("/{run}/grid")
def get_grid(
    q: str = Query(..., description="comma-separated quantity ids"),
    frame: int | None = Query(default=None),
    t_ms: float | None = Query(default=None),
    nx: int = Query(default=grids.DEFAULT_NX, ge=8, le=grids.MAX_NX),
    ny: int = Query(default=grids.DEFAULT_NY, ge=8, le=grids.MAX_NY),
    fmt: str = Depends(format_query),
    scene: Scene = Depends(scene_or_404),
) -> Response:
    """The quantities on an ``nx`` by ``ny`` lattice spanning the channel wall
    to wall at a camera frame or an instant in ms. Binary by default (see
    :mod:`naviernet_studio.grids` for the layout); ``fmt=csv`` for a long table."""
    ids = quantities_or_4xx(scene, q, catalogue.POINT_IDS)
    t_star = instant_or_422(scene, frame, t_ms)
    header, arrays = grids.grid(scene, ids, t_star, nx, ny)
    if fmt == FORMAT_CSV:
        header["provenance"] = exports.provenance(
            scene, ids, frame=header["frame"], t_ms=header["t_ms"]
        )
        return respond(fmt, header, grids.as_table(header, arrays), f"{scene.run_id}_grid")
    return Response(
        content=grids.pack(exports.without_nan(header), arrays),
        media_type="application/octet-stream",
    )


@router.get("/{run}/interface")
def get_interface(
    frame: int | None = Query(default=None),
    t_ms: float | None = Query(default=None),
    scene: Scene = Depends(scene_or_404),
) -> dict:
    """The interface contour(s) in µm at an instant, with the nose and root."""
    from naviernet_studio import geometry

    return exports.without_nan(geometry.interface(scene, instant_or_422(scene, frame, t_ms)))
