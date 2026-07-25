"""Fluids endpoint: the catalogue of selectable working fluids."""

from __future__ import annotations

from fastapi import APIRouter

from naviernet_api.models import Fluid
from naviernet_api.services import fluids as fluids_service

router = APIRouter(prefix="/api/fluids", tags=["fluids"])


@router.get("", response_model=list[Fluid])
def list_fluids() -> list[Fluid]:
    """Every characterised fluid with its atmospheric T_sat and saturated
    properties, for the new-series form's fluid selector."""
    return fluids_service.list_fluids()
