"""Geometry against the images: the pixel transform, raw frames, agreement."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from naviernet_studio import exports, geometry
from naviernet_studio.routes.common import scene_or_404
from naviernet_studio.scene import Scene
from naviernet_studio.settings import Settings, get_settings

router = APIRouter(prefix="/api/runs", tags=["geometry"])


@router.get("/{run}/geometry/transform")
def get_transform(scene: Scene = Depends(scene_or_404)) -> dict:
    """How model coordinates (µm) map onto the raw camera frame's pixels."""
    return geometry.transform(scene)


@router.get("/{run}/geometry/frame/{n}.png")
def get_raw_frame(
    n: int,
    crop: bool = Query(default=True),
    mode: Literal["actual", "cleaned", "outline"] = Query(default="actual"),
    scene: Scene = Depends(scene_or_404),
    settings: Settings = Depends(get_settings),
) -> Response:
    """The camera frame as PNG, cropped to the channel band (plus a margin)
    unless ``crop=false``. ``mode``: ``actual`` is the recording; ``cleaned`` is the
    generated cleaned frame stored in the repository; ``outline`` keeps the bubble
    alone, edged by the measured outline (see :mod:`naviernet_studio.cleaning`)."""
    if mode == "cleaned":
        png = geometry.cleaned_frame_png(settings.repo_root, scene.run_id, n, crop)
        if png is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no cleaned frame {n} for {scene.run_id}: generate the cleaned frames with "
                    "apps/studio/tools/generate_cleaned.py"
                ),
            )
    elif mode == "outline":
        png = geometry.rendered_frame_png(scene, n, crop)
    else:
        png = geometry.raw_frame_png(scene, n, crop)
    if png is None:
        raise HTTPException(status_code=404, detail=f"no frame {n} for {scene.run_id}")
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
