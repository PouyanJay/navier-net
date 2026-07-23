"""Run endpoints: list runs and read one run's detail."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from naviernet_api.models import RunDetail, RunSummary
from naviernet_api.services import runs as runs_service
from naviernet_api.settings import Settings, get_settings

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("", response_model=list[RunSummary])
def list_runs(settings: Settings = Depends(get_settings)) -> list[RunSummary]:
    """Every run under `outputs/`."""
    return runs_service.list_runs(settings)


@router.get("/{run_id}", response_model=RunDetail)
def get_run(run_id: str, settings: Settings = Depends(get_settings)) -> RunDetail:
    """Full detail for one run."""
    detail = runs_service.get_run(settings, run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return detail
