"""Geometry against the images: the pixel transform, raw frames, agreement."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from naviernet_studio import exports, geometry
from naviernet_studio.routes.common import scene_or_404
from naviernet_studio.scene import Scene

router = APIRouter(prefix="/api/runs", tags=["geometry"])


@router.get("/{run}/geometry/transform")
def get_transform(scene: Scene = Depends(scene_or_404)) -> dict:
    """How model coordinates (µm) map onto the raw camera frame's pixels."""
    return geometry.transform(scene)


@router.get("/{run}/geometry/frame/{n}.png")
def get_raw_frame(
    n: int,
    crop: bool = Query(default=True),
    clean: bool = Query(default=False),
    scene: Scene = Depends(scene_or_404),
) -> Response:
    """The camera frame at full resolution as PNG, cropped to the channel band
    (plus a margin) unless ``crop=false``. ``clean=true`` removes the set-up's
    heaters and walls, keeping the bubble (see :mod:`naviernet_studio.cleaning`)."""
    png = (
        geometry.cleaned_frame_png(scene, n, crop)
        if clean
        else geometry.raw_frame_png(scene, n, crop)
    )
    if png is None:
        raise HTTPException(status_code=404, detail=f"no raw frame {n} for {scene.run_id}")
    return Response(
        content=png, media_type="image/png", headers={"Cache-Control": "max-age=3600"}
    )


@router.get("/{run}/geometry/agreement")
def get_agreement(frame: int = Query(...), scene: Scene = Depends(scene_or_404)) -> dict:
    """IoU, nose, areas and both contours at a camera frame."""
    payload = geometry.agreement(scene, frame)
    if payload is None:
        raise HTTPException(
            status_code=422, detail=f"frame {frame} is not in this run's tensors"
        )
    return exports.without_nan(payload)
