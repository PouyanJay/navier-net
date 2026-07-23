"""API response models.

Deliberately thin: these mirror what the pipeline already writes to disk
(`metrics.json`, the `.hydra` config snapshot, the artifact layout) rather than
introducing a parallel data model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class RunSummary(BaseModel):
    """One row in the runs list."""

    id: str
    dataset: str | None = None
    status: str  # "trained" (has a checkpoint) | "empty"
    steps: int | None = None  # completed training steps, if known
    iou_holdout: float | None = None  # headline generalization metric, if evaluated


class ArtifactFlags(BaseModel):
    """Which deliverables exist for a run."""

    checkpoint: bool = False
    metrics: bool = False
    groups: bool = False
    video: bool = False
    figures: list[str] = []


class RunDetail(BaseModel):
    """Full detail for a single run."""

    id: str
    dataset: str | None = None
    status: str
    steps: int | None = None
    metrics: dict | None = None  # verbatim metrics.json
    config: dict | None = None  # resolved Hydra config snapshot (.hydra/config.yaml)
    artifacts: ArtifactFlags


class OperatingConditions(BaseModel):
    """The experiment's operating conditions (from the experiment config)."""

    fluid: str
    T_sat_C: float
    q_wall_W_cm2: float
    flow_rate_mL_hr: float
    channel_width_um: float
    channel_height_um: float
    dt_frame_ms: float
    flow_direction: str
    n_frames_raw: int
    n_frames_usable: int
    n_frames_event: int


class DatasetSummary(BaseModel):
    """One row in the datasets list."""

    id: str
    n_frames: int  # raw TIFFs present on disk
    processed: bool  # preprocessed tensors exist


class DatasetDetail(BaseModel):
    """Full detail for one dataset."""

    id: str
    n_frames: int
    processed: bool
    has_qc: bool  # a preprocessing QC figure exists
    conditions: OperatingConditions


class PreprocessStatus(BaseModel):
    """State of a dataset's preprocessing job."""

    dataset: str
    state: Literal["idle", "running", "done", "error"]
    message: str | None = None
    has_qc: bool = False


class ModelArchitecture(BaseModel):
    """The PINN field-ensemble architecture (from the model config)."""

    fields: list[str]
    hidden: int
    layers: int
    fourier_feats: int
    fourier_scale: float
    alpha_eps: float
    nodewise_activation: bool


class PhysicsValidation(BaseModel):
    """The physics-validation summary the Results view shows.

    Composed from the run's `metrics.json` and `dimensionless_groups.json` plus a
    documented measured nose speed; the API does no physics of its own.
    """

    nose_speed_inferred_mm_s: float | None = None
    nose_speed_measured_mm_s: float | None = None
    nose_speed_error_pct: float | None = None
    bretherton_film_um: float | None = None
    hele_shaw: float | None = None
    reynolds: float | None = None
    weber: float | None = None
    capillary: float | None = None
    prandtl: float | None = None
    iou_mean: float | None = None
    iou_holdout: float | None = None
    holdout_frame: int | None = None
