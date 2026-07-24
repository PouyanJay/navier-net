"""Run endpoints: list, read, launch, and stream runs."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from naviernet_api.models import (
    PhysicsValidation,
    RunDetail,
    RunJobStatus,
    RunLaunchRequest,
    RunSummary,
)
from naviernet_api.services import physics as physics_service
from naviernet_api.services import reconstruction, run_manager
from naviernet_api.services import runs as runs_service
from naviernet_api.settings import Settings, get_settings

router = APIRouter(prefix="/api/runs", tags=["runs"])

# How often the SSE stream checks the run's event buffer for news.
_STREAM_POLL_SECONDS = 0.25


@router.get("", response_model=list[RunSummary])
def list_runs(settings: Settings = Depends(get_settings)) -> list[RunSummary]:
    """Every run under `outputs/`."""
    return runs_service.list_runs(settings)


@router.post("", response_model=RunJobStatus, status_code=202)
def launch_run(
    request: RunLaunchRequest, settings: Settings = Depends(get_settings)
) -> RunJobStatus:
    """Start (or resume) a background training run."""
    try:
        return run_manager.launch(settings, request)
    except run_manager.LaunchRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.get("/active", response_model=RunJobStatus | None)
def get_active_run() -> RunJobStatus | None:
    """The training run currently in progress, if any."""
    return run_manager.active_run()


@router.get("/{run_id}/status", response_model=RunJobStatus)
def get_run_status(run_id: str) -> RunJobStatus:
    """Live status of a run launched by this server."""
    result = run_manager.status(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"no launched run {run_id!r}")
    return result


async def _run_events(run_id: str) -> AsyncIterator[dict]:
    """Replay the run's buffered events, then follow until it finishes."""
    cursor = 0
    while True:
        batch = run_manager.events_since(run_id, cursor)
        if batch is None:
            return
        events, cursor, terminal = batch
        for event in events:
            yield {"event": event["event"], "data": json.dumps(event["data"])}
        if terminal:
            return
        await asyncio.sleep(_STREAM_POLL_SECONDS)


@router.get("/{run_id}/stream")
def stream_run(run_id: str) -> EventSourceResponse:
    """SSE stream of a launched run: `hist`, `log`, and `status` events."""
    if run_manager.status(run_id) is None:
        raise HTTPException(status_code=404, detail=f"no launched run {run_id!r}")
    return EventSourceResponse(_run_events(run_id))


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


@router.get("/{run_id}/trajectory")
def get_trajectory(run_id: str, settings: Settings = Depends(get_settings)) -> dict:
    """Continuous + measured growth kinematics (written by the evaluate stage)."""
    trajectory = runs_service.read_trajectory(settings, run_id)
    if trajectory is None:
        raise HTTPException(status_code=404, detail=f"no trajectory for run {run_id!r}")
    return trajectory


@router.get("/{run_id}/interface")
def get_interface(
    run_id: str,
    frames: int = Query(default=40, ge=8, le=120),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Interface contours over continuous time, for the reconstruction viewport."""
    payload = reconstruction.interface_frames(settings, run_id, frames)
    if payload is None:
        raise HTTPException(
            status_code=404, detail=f"run {run_id!r} has no trained model to reconstruct from"
        )
    return payload


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
