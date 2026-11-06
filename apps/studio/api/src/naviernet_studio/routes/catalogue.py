"""The quantity catalogue, with availability for a run when one is named."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query

from naviernet_studio import catalogue
from naviernet_studio.scene import get_scene
from naviernet_studio.settings import Settings, get_settings

router = APIRouter(prefix="/api/catalogue", tags=["catalogue"])


@router.get("")
def get_catalogue(
    run: str | None = Query(default=None), settings: Settings = Depends(get_settings)
) -> list[dict]:
    """Every quantity the studio knows. With ``run=`` each entry also says
    whether that run can produce it and the numeric factor from model units."""
    if run is None:
        return [asdict(q) for q in catalogue.ALL]
    scene = get_scene(settings, run)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"run {run!r} has no loadable checkpoint")
    return catalogue.catalogue_for(scene)
