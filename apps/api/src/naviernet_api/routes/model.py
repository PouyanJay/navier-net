"""Model endpoint: the PINN field-ensemble architecture for a dataset."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from naviernet_api.models import ModelArchitecture
from naviernet_api.services import datasets as datasets_service
from naviernet_api.services.config_service import compose_cfg
from naviernet_api.settings import Settings, get_settings

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/{dataset}", response_model=ModelArchitecture)
def get_model(dataset: str, settings: Settings = Depends(get_settings)) -> ModelArchitecture:
    """The field-ensemble architecture, read live from the dataset's config."""
    if datasets_service.get_dataset(settings, dataset) is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset!r} not found")
    net = compose_cfg(dataset).model
    return ModelArchitecture(
        fields=list(net.fields),
        hidden=net.hidden,
        layers=net.layers,
        fourier_feats=net.fourier_feats,
        fourier_scale=net.fourier_scale,
        alpha_eps=net.alpha_eps,
        nodewise_activation=net.nodewise_activation,
    )
