"""Sweep endpoints: launch a seed sweep and read its status.

Children stream individually over the existing `/api/runs/{id}/stream`; this
router only owns the sweep-level lifecycle.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from naviernet_api.models import SweepLaunchRequest, SweepStatus
from naviernet_api.services import run_manager, sweep_manager
from naviernet_api.settings import Settings, get_settings

router = APIRouter(prefix="/api/sweeps", tags=["sweeps"])


@router.post("", response_model=SweepStatus, status_code=202)
def launch_sweep(
    request: SweepLaunchRequest, settings: Settings = Depends(get_settings)
) -> SweepStatus:
    """Start a background seed sweep (sequential children)."""
    try:
        return sweep_manager.launch(settings, request)
    except run_manager.LaunchRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.get("/active", response_model=SweepStatus | None)
def get_active_sweep() -> SweepStatus | None:
    """The sweep currently in progress, if any."""
    return sweep_manager.active_sweep()


@router.get("/{sweep_id}", response_model=SweepStatus)
def get_sweep(sweep_id: str) -> SweepStatus:
    """Live status of a sweep launched by this server."""
    result = sweep_manager.status(sweep_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"no sweep {sweep_id!r}")
    return result
