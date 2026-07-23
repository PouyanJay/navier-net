"""Dataset endpoints: list, detail, live groups, QC + frame previews, upload,
and driving preprocess."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from naviernet_api.models import (
    DatasetDetail,
    DatasetSummary,
    PreprocessStatus,
)
from naviernet_api.services import datasets as datasets_service
from naviernet_api.services import jobs as jobs_service
from naviernet_api.services.config_service import compute_groups_for
from naviernet_api.services.datasets import UploadError
from naviernet_api.settings import Settings, get_settings

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetSummary])
def list_datasets(settings: Settings = Depends(get_settings)) -> list[DatasetSummary]:
    """Every dataset under `data/raw/`."""
    return datasets_service.list_datasets(settings)


@router.get("/{dataset}", response_model=DatasetDetail)
def get_dataset(dataset: str, settings: Settings = Depends(get_settings)) -> DatasetDetail:
    """Operating conditions and status for one dataset."""
    detail = datasets_service.get_dataset(settings, dataset)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset!r} not found")
    return detail


@router.get("/{dataset}/groups")
def get_dataset_groups(dataset: str, settings: Settings = Depends(get_settings)) -> dict:
    """Live dimensionless groups, computed from the dataset's config."""
    if not datasets_service.is_valid_dataset_id(dataset):
        raise HTTPException(status_code=404, detail=f"dataset {dataset!r} not found")
    return compute_groups_for(dataset)


@router.get("/{dataset}/qc")
def get_dataset_qc(dataset: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    """The preprocessing QC figure."""
    path = datasets_service.qc_path(settings, dataset)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no QC figure for {dataset!r}")
    return FileResponse(path, media_type="image/png")


@router.get("/{dataset}/frames/{n}")
def get_frame_preview(
    dataset: str, n: int, settings: Settings = Depends(get_settings)
) -> Response:
    """A raw frame rendered to a downscaled PNG for the browser."""
    png = datasets_service.frame_preview_png(settings, dataset, n)
    if png is None:
        raise HTTPException(status_code=404, detail=f"frame {n} not found in {dataset!r}")
    return Response(content=png, media_type="image/png")


@router.post("/{dataset}/upload", response_model=DatasetSummary)
async def upload_frames(
    dataset: str,
    files: list[UploadFile],
    settings: Settings = Depends(get_settings),
) -> DatasetSummary:
    """Upload an image sequence (validated TIFFs) into the dataset."""
    frames = [await f.read() for f in files]
    try:
        datasets_service.save_frames(settings, dataset, frames)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return next(d for d in datasets_service.list_datasets(settings) if d.id == dataset)


@router.post("/{dataset}/preprocess", response_model=PreprocessStatus)
def start_preprocess(
    dataset: str, settings: Settings = Depends(get_settings)
) -> PreprocessStatus:
    """Drive the preprocess stage for the dataset in the background."""
    if not datasets_service.is_valid_dataset_id(dataset):
        raise HTTPException(status_code=404, detail=f"dataset {dataset!r} not found")
    return jobs_service.start_preprocess(settings, dataset)


@router.get("/{dataset}/preprocess", response_model=PreprocessStatus)
def preprocess_status(
    dataset: str, settings: Settings = Depends(get_settings)
) -> PreprocessStatus:
    """Poll the preprocessing job state."""
    return jobs_service.status(settings, dataset)
