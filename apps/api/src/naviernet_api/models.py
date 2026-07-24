"""API response models.

Deliberately thin: these mirror what the pipeline already writes to disk
(`metrics.json`, the `.hydra` config snapshot, the artifact layout) rather than
introducing a parallel data model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


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
    U_ref_m_s: float | None = None  # reference velocity (nondimensionalisation)


class ProjectSummary(BaseModel):
    """A project: a uuid identity with editable metadata and the series
    (datasets under data/raw/) uploaded into it."""

    id: str
    name: str
    description: str = ""
    datasets: list[str] = []  # series ids, in upload order
    created_at: str  # ISO-8601 UTC

    @model_validator(mode="before")
    @classmethod
    def _migrate_single_dataset(cls, data: dict) -> dict:
        """Files written before multi-series support carried `dataset: str|null`."""
        if isinstance(data, dict) and "datasets" not in data and "dataset" in data:
            data = dict(data)
            legacy = data.pop("dataset")
            data["datasets"] = [legacy] if legacy else []
        return data


class ProjectCreate(BaseModel):
    """Payload for creating a project (an empty environment, no data yet)."""

    name: str
    description: str = ""


class ProjectUpdate(BaseModel):
    """Editable project fields; omitted fields are left unchanged."""

    name: str | None = None
    description: str | None = None
    datasets: list[str] | None = None  # full replacement list; null clears


class DatasetSummary(BaseModel):
    """One row in the datasets list."""

    id: str
    n_frames: int  # raw TIFFs present on disk
    processed: bool  # preprocessed tensors exist
    conditions_set: bool = False  # per-series conditions.json saved
    frame_px: tuple[int, int] | None = None  # (width, height) of the raw frames
    dt_frame_ms: float | None = None  # frame interval, from the series' config


class ConditionsUpdate(BaseModel):
    """Editable per-series operating conditions; omitted fields keep their
    current (config-default or previously saved) value."""

    T_sat_C: float | None = None
    dt_frame_ms: float | None = None
    channel_width_um: float | None = None
    channel_height_um: float | None = None
    flow_rate_mL_hr: float | None = None
    q_wall_W_cm2: float | None = None
    U_ref: float | None = None


class QcKinematics(BaseModel):
    """Bubble length per frame plus the linear growth fit."""

    t_ms: list[float]
    length_um: list[float]
    fit_slope_mm_s: float
    fit_intercept_um: float


class QcInterfaceFrame(BaseModel):
    """One frame's interface contours as [x*, y*] polylines."""

    index: int
    t_ms: float
    contours: list[list[list[float]]]


class QcInterface(BaseModel):
    x_pin_star: float
    x_range: list[float]
    y_range: list[float]
    frames: list[QcInterfaceFrame]


class QcSdf(BaseModel):
    """The mid-frame signed distance field, decimated for the browser."""

    frame_index: int
    t_ms: float
    x_range: list[float]
    y_range: list[float]
    values: list[list[float]]


class QcData(BaseModel):
    """The three preprocessing checks as chart data."""

    dataset: str
    n_frames_event: int
    kinematics: QcKinematics
    interface: QcInterface
    sdf: QcSdf


class ConditionsResponse(BaseModel):
    """A conditions edit round-trip: the saved values + recomputed groups."""

    conditions: OperatingConditions
    groups: dict[str, float]


class DatasetDetail(BaseModel):
    """Full detail for one dataset."""

    id: str
    n_frames: int
    processed: bool
    has_qc: bool  # a preprocessing QC figure exists
    conditions_set: bool = False
    frame_px: tuple[int, int] | None = None
    holdout_frame: int | None = None  # 1-based camera frame that is never supervised
    um_per_px: float | None = None  # calibration, once preprocessed
    notes: str | None = None  # the experiment's frame-usage story
    conditions: OperatingConditions
    excluded_frames: list[int] = []  # 1-based camera frames kept out of the tensors
    # Whether the preprocessed tensors were built with the exclusions above; false
    # means an edit is pending a preprocessing re-run.
    exclusions_applied: bool = False


class ExclusionsUpdate(BaseModel):
    """Full replacement of a series' excluded camera frames (1-based)."""

    excluded_frames: list[int]


class PreprocessStatus(BaseModel):
    """State of a dataset's preprocessing job."""

    dataset: str
    state: Literal["idle", "running", "done", "error"]
    message: str | None = None
    has_qc: bool = False


class LossWeightsInput(BaseModel):
    """Initial loss-term weights for a run (`cfg.training.weights`)."""

    data: float = Field(default=10.0, ge=0, le=1e4)
    vof: float = Field(default=1.0, ge=0, le=1e4)
    div: float = Field(default=1.0, ge=0, le=1e4)
    src: float = Field(default=0.1, ge=0, le=1e4)
    bc: float = Field(default=5.0, ge=0, le=1e4)


class RunLaunchRequest(BaseModel):
    """A request to start (or resume) a training run.

    Every numeric field maps 1:1 onto `cfg.training`; the bounds exist because
    the Hydra schema types but does not range-check its values, and these come
    from the network (SECURITY.md §4). On resume only `steps` and `render`
    apply — the rest of the configuration is fixed by the original run's own
    config snapshot, and any other values sent here are ignored.
    """

    dataset: str | None = None  # required for a new run
    resume: bool = False
    run_id: str | None = None  # required when resuming
    steps: int = Field(default=1500, ge=1, le=20_000)
    lr: float = Field(default=2e-3, gt=0, le=1.0)
    lr_halflife: int = Field(default=800, ge=1, le=100_000)
    n_data: int = Field(default=3072, ge=16, le=16_384)
    n_coll: int = Field(default=3072, ge=16, le=16_384)
    n_bc: int = Field(default=512, ge=8, le=8192)
    holdout_frame: int = Field(default=5, ge=-1, le=64)  # -1 = train on all frames
    rebalance_every: int = Field(default=500, ge=10, le=100_000)
    log_every: int = Field(default=200, ge=10, le=5000)  # ≥10 bounds the event stream
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    weights: LossWeightsInput = Field(default_factory=LossWeightsInput)
    render: bool = True  # render figures + video after evaluation

    @model_validator(mode="after")
    def _check_target(self) -> RunLaunchRequest:
        if self.resume and not self.run_id:
            raise ValueError("resume requires run_id")
        if not self.resume and not self.dataset:
            raise ValueError("a new run requires dataset")
        return self


class RunJobStatus(BaseModel):
    """Live state of a launched training run."""

    run_id: str
    dataset: str | None = None
    state: Literal["queued", "running", "done", "error"]
    stage: str | None = None  # pipeline stage currently executing
    message: str | None = None
    steps_done: int = 0
    steps_total: int = 0


class SweepLaunchRequest(RunLaunchRequest):
    """A request to run the same configuration across several seeds.

    Children are ordinary runs (train + evaluate; rendering defaults off — a
    sweep is for comparison, not deliverables). `seed` is ignored; `seeds`
    drives the children. Sweeps never resume.
    """

    seeds: list[int] = Field(min_length=1, max_length=6)
    render: bool = False

    @model_validator(mode="after")
    def _check_sweep(self) -> SweepLaunchRequest:
        if self.resume or self.run_id:
            raise ValueError("a sweep cannot resume an existing run")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if any(seed < 0 or seed > 2**31 - 1 for seed in self.seeds):
            raise ValueError("seeds must be non-negative 32-bit integers")
        return self


class SweepStatus(BaseModel):
    """Live state of a seed sweep and its child runs."""

    sweep_id: str
    dataset: str
    state: Literal["running", "done", "error"]
    message: str | None = None
    seeds: list[int]
    children: list[RunJobStatus]


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
