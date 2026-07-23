"""Run endpoints: list runs and read one run's detail."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from naviernet_api.models import PhysicsValidation, RunDetail, RunSummary
from naviernet_api.services import physics as physics_service
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


@router.get("/{run_id}/groups")
def get_groups(run_id: str, settings: Settings = Depends(get_settings)) -> dict:
    """The run's derived dimensionless groups."""
    groups = runs_service.read_groups(settings, run_id)
    if groups is None:
        raise HTTPException(status_code=404, detail=f"no groups for run {run_id!r}")
    return groups


@router.get("/{run_id}/validation", response_model=PhysicsValidation)
def get_validation(
    run_id: str, settings: Settings = Depends(get_settings)
) -> PhysicsValidation:
    """Physics-validation summary (nose speed, Bretherton, key groups, IoU)."""
    # Read only dataset + metrics (no checkpoint load — validation never uses it).
    result = runs_service.read_dataset_and_metrics(settings, run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    dataset, metrics = result
    groups = runs_service.read_groups(settings, run_id)
    return physics_service.build_validation(dataset, metrics, groups)


@router.get("/{run_id}/loss-history")
def get_loss_history(run_id: str, settings: Settings = Depends(get_settings)) -> list[dict]:
    """Per-log-step training loss records from the checkpoint."""
    history = runs_service.read_loss_history(settings, run_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"no loss history for run {run_id!r}")
    return history


@router.get("/{run_id}/figures/{name}")
def get_figure(
    run_id: str, name: str, settings: Settings = Depends(get_settings)
) -> FileResponse:
    """Serve one figure PNG from the run."""
    path = runs_service.figure_path(settings, run_id, name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"figure {name!r} not found")
    return FileResponse(path, media_type="image/png")


@router.get("/{run_id}/video")
def get_video(run_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    """Serve the run's rendered MP4."""
    path = runs_service.video_path(settings, run_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no video for run {run_id!r}")
    return FileResponse(path, media_type="video/mp4")


@router.get("/{run_id}/checkpoint")
def get_checkpoint(run_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    """Download the run's checkpoint."""
    path = runs_service.checkpoint_path(settings, run_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no checkpoint for run {run_id!r}")
    return FileResponse(path, media_type="application/octet-stream", filename="ckpt.pt")


@router.get("/{run_id}/tensors")
def get_tensors(run_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    """Download the preprocessed tensors for the run's dataset."""
    path = runs_service.tensors_path(settings, run_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no tensors for run {run_id!r}")
    return FileResponse(
        path, media_type="application/octet-stream", filename="training_data.npz"
    )
