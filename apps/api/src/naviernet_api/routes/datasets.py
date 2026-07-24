"""Dataset endpoints: list, detail, live groups, QC + frame previews, upload,
and driving preprocess."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from naviernet_api.models import (
    ConditionsResponse,
    ConditionsUpdate,
    DatasetDetail,
    DatasetSummary,
    ExclusionsUpdate,
    PreprocessStatus,
    QcData,
)
from naviernet_api.services import datasets as datasets_service
from naviernet_api.services import jobs as jobs_service
from naviernet_api.services import qc as qc_service
from naviernet_api.services.config_service import compose_cfg, compute_groups_for
from naviernet_api.services.datasets import ConditionsError, ExclusionError, UploadError
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
def get_dataset_groups(
    dataset: str, settings: Settings = Depends(get_settings)
) -> dict[str, float]:
    """Live dimensionless groups, computed from the dataset's config plus its
    saved per-series conditions."""
    if datasets_service.get_dataset(settings, dataset) is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset!r} not found")
    return compute_groups_for(
        dataset, overrides=datasets_service.series_overrides(settings, dataset)
    )


@router.patch("/{dataset}/conditions", response_model=ConditionsResponse)
def update_conditions(
    dataset: str,
    payload: ConditionsUpdate,
    settings: Settings = Depends(get_settings),
) -> ConditionsResponse:
    """Save per-series operating conditions; groups recompute immediately."""
    if datasets_service.get_dataset_summary(settings, dataset) is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset!r} not found")
    updates = payload.model_dump(exclude_unset=True, exclude_none=True)
    try:
        datasets_service.save_conditions(settings, dataset, updates)
    except ConditionsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    overrides = datasets_service.series_overrides(settings, dataset)
    cfg = compose_cfg(dataset, overrides=overrides)
    return ConditionsResponse(
        conditions=datasets_service.conditions_from_cfg(cfg),
        groups=compute_groups_for(dataset, overrides=overrides),
    )


@router.put("/{dataset}/excluded-frames", response_model=DatasetDetail)
def set_excluded_frames(
    dataset: str,
    payload: ExclusionsUpdate,
    settings: Settings = Depends(get_settings),
) -> DatasetDetail:
    """Replace the frames held out of the tensors. Takes effect on the next
    preprocessing run — the returned detail says whether one is still pending."""
    if datasets_service.get_dataset_summary(settings, dataset) is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset!r} not found")
    try:
        datasets_service.save_excluded_frames(settings, dataset, payload.excluded_frames)
    except ExclusionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    detail = datasets_service.get_dataset(settings, dataset)
    if detail is None:  # saved but not found — should not happen
        raise HTTPException(status_code=500, detail="exclusions saved but dataset not found")
    return detail


@router.get("/{dataset}/qc-data", response_model=QcData)
def get_qc_data(dataset: str, settings: Settings = Depends(get_settings)) -> QcData:
    """The preprocessing QC checks as chart data (kinematics, interface, SDF)."""
    data = qc_service.qc_data(settings, dataset)
    if data is None:
        raise HTTPException(
            status_code=404, detail=f"dataset {dataset!r} has not been preprocessed"
        )
    return data


@router.get("/{dataset}/qc")
def get_dataset_qc(dataset: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    """The pipeline's rendered QC figure, as a downloadable artifact.

    The web app draws its own interactive QC from /qc-data; this stays as the
    raw matplotlib artifact the preprocess stage writes to disk."""
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
    # Reject the count before buffering any payload (SECURITY.md §4), and read
    # each part bounded so an oversized frame can't exhaust memory first.
    if len(files) > datasets_service.MAX_FRAMES:
        raise HTTPException(status_code=400, detail=f"too many frames ({len(files)})")
    limit = datasets_service.MAX_FRAME_BYTES
    frames = [await f.read(limit + 1) for f in files]

    try:
        datasets_service.save_frames(settings, dataset, frames)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary = datasets_service.get_dataset_summary(settings, dataset)
    if summary is None:  # saved but not found — should not happen
        raise HTTPException(status_code=500, detail="upload saved but dataset not found")
    return summary


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
    if not datasets_service.is_valid_dataset_id(dataset):
        raise HTTPException(status_code=404, detail=f"dataset {dataset!r} not found")
    return jobs_service.status(settings, dataset)
