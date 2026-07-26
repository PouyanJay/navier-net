"""Model endpoint: the PINN field-ensemble architecture for a dataset."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from naviernet_api.models import ModelArchitecture, ModelUpdate, PerFieldArch
from naviernet_api.services import datasets as datasets_service
from naviernet_api.services.config_service import compose_cfg
from naviernet_api.services.datasets import ModelConfigError
from naviernet_api.settings import Settings, get_settings

router = APIRouter(prefix="/api/model", tags=["model"])


def _architecture(settings: Settings, dataset: str) -> ModelArchitecture:
    net = compose_cfg(
        dataset, overrides=datasets_service.series_overrides(settings, dataset)
    ).model
    per_field = {
        name: PerFieldArch(hidden=sub.get("hidden"), layers=sub.get("layers"))
        for name, sub in (net.get("per_field") or {}).items()
    }
    return ModelArchitecture(
        fields=list(net.fields),
        hidden=net.hidden,
        layers=net.layers,
        fourier_feats=net.fourier_feats,
        fourier_scale=net.fourier_scale,
        alpha_eps=net.alpha_eps,
        nodewise_activation=net.nodewise_activation,
        per_field=per_field,
    )


@router.get("/{dataset}", response_model=ModelArchitecture)
def get_model(dataset: str, settings: Settings = Depends(get_settings)) -> ModelArchitecture:
    """The field-ensemble architecture, read live from the dataset's config."""
    if datasets_service.get_dataset(settings, dataset) is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset!r} not found")
    return _architecture(settings, dataset)


@router.put("/{dataset}", response_model=ModelArchitecture)
def update_model(
    dataset: str, payload: ModelUpdate, settings: Settings = Depends(get_settings)
) -> ModelArchitecture:
    """Save per-series model-architecture overrides (globals + per-field)."""
    if datasets_service.get_dataset(settings, dataset) is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset!r} not found")
    arch = payload.model_dump(exclude_unset=True, exclude_none=True)
    try:
        datasets_service.save_model_arch(settings, dataset, arch)
    except ModelConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _architecture(settings, dataset)
