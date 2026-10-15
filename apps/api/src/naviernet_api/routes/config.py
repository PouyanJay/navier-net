"""Config endpoint: the field table every user-facing input is declared in."""

from __future__ import annotations

from fastapi import APIRouter

from naviernet.config.fields import export
from naviernet_api.models import FieldSpecOut

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("/fields", response_model=list[FieldSpecOut])
def list_fields() -> list[FieldSpecOut]:
    """Every user-facing input (`naviernet.config.fields`): its Hydra key, label,
    unit, kind, options, bounds, owner and default, in table order. The web's
    forms render from a committed copy of this; the endpoint is how any other
    client reads the same declaration."""
    return [FieldSpecOut(**entry) for entry in export()]
