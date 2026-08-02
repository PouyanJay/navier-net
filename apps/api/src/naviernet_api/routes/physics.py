"""Physics endpoint: a dataset's governing equations and editing which train.

The equation set is read from the pipeline's physics registry, so the API never
hardcodes physics. Enabling an equation writes the series' model config, which
every compose site applies -- so a launched run trains exactly what is shown.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from naviernet.physics.registry import REGISTRY, enabled_equations
from naviernet_api.models import EquationState, PhysicsState, PhysicsUpdate
from naviernet_api.services import datasets as datasets_service
from naviernet_api.services.config_service import compose_cfg, compute_groups_for
from naviernet_api.services.datasets import ModelConfigError
from naviernet_api.settings import Settings, get_settings

router = APIRouter(prefix="/api/physics", tags=["physics"])


def _physics_state(settings: Settings, dataset: str) -> PhysicsState:
    overrides = datasets_service.series_overrides(settings, dataset)
    cfg = compose_cfg(dataset, overrides=overrides)
    fields = list(cfg.model.fields)
    sharp = bool(getattr(cfg.model, "sharp_interface", False))
    active = {e.id for e in enabled_equations(fields, sharp_interface=sharp)}
    weights = cfg.training.weights
    equations = [
        EquationState(
            id=e.id,
            name=e.name,
            stage=e.stage,
            tex=e.tex,
            weight_key=e.weight_key,
            fields_required=list(e.fields_required),
            fields_added=list(e.fields_added),
            groups=list(e.groups),
            core=e.core,
            enabled=e.id in active,
            weight=float(getattr(weights, e.weight_key)),
            mode=e.mode,
        )
        for e in REGISTRY
    ]
    return PhysicsState(
        dataset=dataset,
        equations=equations,
        fields=fields,
        groups=compute_groups_for(dataset, overrides=overrides),
        sharp_interface=sharp,
    )


@router.get("/{dataset}", response_model=PhysicsState)
def get_physics(dataset: str, settings: Settings = Depends(get_settings)) -> PhysicsState:
    """The governing equations, their enabled state and weights, and live groups."""
    if datasets_service.get_dataset(settings, dataset) is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset!r} not found")
    return _physics_state(settings, dataset)


@router.put("/{dataset}", response_model=PhysicsState)
def update_physics(
    dataset: str, payload: PhysicsUpdate, settings: Settings = Depends(get_settings)
) -> PhysicsState:
    """Save which Stage-B equations train and their loss weights."""
    if datasets_service.get_dataset(settings, dataset) is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset!r} not found")
    try:
        datasets_service.save_physics(settings, dataset, payload.model_dump())
    except ModelConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _physics_state(settings, dataset)
