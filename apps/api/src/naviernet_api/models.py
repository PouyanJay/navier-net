"""API response models.

Deliberately thin: these mirror what the pipeline already writes to disk
(`metrics.json`, the `.hydra` config snapshot, the artifact layout) rather than
introducing a parallel data model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Fluid(BaseModel):
    """A characterised working fluid: its label, atmospheric saturation
    temperature, and the saturated two-phase properties the pipeline needs.
    Mirrors the fluid config group (`configs/fluid/<id>.yaml`)."""

    id: str  # the config group stem, e.g. "fc72" (also the fluid= override value)
    name: str  # display label, e.g. "FC-72"
    T_sat_C: float  # saturation temperature at 1 atm (deg C)
    rho_l: float  # kg/m^3
    rho_v: float
    mu_l: float  # Pa.s
    mu_v: float
    k_l: float  # W/m/K
    k_v: float
    cp_l: float  # J/kg/K
    cp_v: float
    sigma: float  # N/m
    h_lv: float  # J/kg


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

    id: str  # immutable filesystem key (data/raw/<id>)
    n_frames: int  # raw TIFFs present on disk
    processed: bool  # preprocessed tensors exist
    conditions_set: bool = False  # per-series conditions.json saved
    label: str | None = None  # editable display name; None = show the id
    frame_px: tuple[int, int] | None = None  # (width, height) of the raw frames
    dt_frame_ms: float | None = None  # frame interval, from the series' config


class ConditionsUpdate(BaseModel):
    """Editable per-series operating conditions; omitted fields keep their
    current (config-default or previously saved) value. Unknown fields are
    rejected (T_sat is not here: it is derived from the selected fluid)."""

    model_config = {"extra": "forbid"}

    fluid: str | None = None  # fluid config-group id, e.g. "water" (allow-listed)
    dt_frame_ms: float | None = None
    channel_width_um: float | None = None
    channel_height_um: float | None = None
    flow_rate_mL_hr: float | None = None
    q_wall_W_cm2: float | None = None
    U_ref: float | None = None  # reference velocity (scales.U_ref)


class QcKinematics(BaseModel):
    """Bubble length per frame plus the linear growth fit."""

    t_ms: list[float]
    length_um: list[float]
    fit_slope_mm_s: float
    fit_intercept_um: float


class QcInterfaceFrame(BaseModel):
    """One frame's bubble silhouette as closed [x*, y*] rings.

    Closed, not open polylines: the bubble spans the channel, so its α = 0.5
    contour runs into the top and bottom of the imaged band. Rings carry the
    boundary segments too, giving a complete outline instead of two loose arcs.
    """

    index: int
    # The 1-based camera frame this silhouette belongs to. Not index + 1 once
    # frames are excluded: it maps a ring straight onto its raw frame image.
    camera_frame: int
    t_ms: float
    rings: list[list[list[float]]]


class QcInterface(BaseModel):
    x_pin_star: float
    x_range: list[float]
    y_range: list[float]
    l_ref_um: float  # x* · l_ref_um = µm, for physical axis labels
    # Top row of the imaged band in raw-frame pixels: rings are cut to this ROI,
    # so overlaying them on the full frame means shifting y down by this much.
    y_roi_top: int
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

    id: str  # immutable filesystem key (data/raw/<id>)
    n_frames: int
    processed: bool
    has_qc: bool  # a preprocessing QC figure exists
    conditions_set: bool = False
    label: str | None = None  # editable display name; None = show the id
    frame_px: tuple[int, int] | None = None
    holdout_frame: int | None = None  # 1-based camera frame that is never supervised
    um_per_px: float | None = None  # calibration, once preprocessed
    notes: str | None = None  # the experiment's frame-usage story
    conditions: OperatingConditions
    excluded_frames: list[int] = []  # 1-based camera frames kept out of the tensors
    # Whether the preprocessed tensors were built with the exclusions above; false
    # means an edit is pending a preprocessing re-run.
    exclusions_applied: bool = False
    # Whether the tensors were built with the current tensor-baked conditions
    # (frame interval, channel width, reference velocity). False means a baked
    # condition edit is pending a re-preprocess. True for an unprocessed series.
    conditions_applied: bool = True


class ExclusionsUpdate(BaseModel):
    """Full replacement of a series' excluded camera frames (1-based)."""

    excluded_frames: list[int]


# The single source of truth for the series display-name length cap, shared by
# the request model (transport gate) and the service (defence for direct callers).
MAX_LABEL_LEN = 80


class LabelUpdate(BaseModel):
    """A series' editable display name. Blank clears it back to the id."""

    label: str = Field(default="", max_length=MAX_LABEL_LEN)


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
    apply; the rest of the configuration is fixed by the original run's own
    config snapshot, and any other values sent here are ignored.
    """

    dataset: str | None = None  # a single-dataset run (or the primary of a joint one)
    # Joint (transfer-learning) training: train ONE model across these datasets.
    # Empty/absent means the single `dataset`. Use `resolved_datasets()` to read.
    datasets: list[str] | None = None
    resume: bool = False
    run_id: str | None = None  # required when resuming
    steps: int = Field(default=1500, ge=1, le=20_000)
    lr: float = Field(default=2e-3, gt=0, le=1.0)
    lr_halflife: int = Field(default=800, ge=1, le=100_000)
    n_data: int = Field(default=3072, ge=16, le=16_384)
    n_coll: int = Field(default=3072, ge=16, le=16_384)
    n_bc: int = Field(default=512, ge=8, le=8192)
    holdout_frame: int = Field(default=5, ge=-1, le=64)  # -1 = train on all frames
    # Validation axis A: a per-dataset fraction of frames held out as an
    # in-distribution validation set (0 = off, unchanged). `tail` extrapolates
    # (the honest test); `scatter` interpolates. Composes with `holdout_frame`.
    val_fraction: float = Field(default=0.0, ge=0.0, le=0.9)
    val_strategy: Literal["tail", "scatter"] = "tail"
    # Validation axis B: whole datasets kept OUT of training and scored on every
    # frame as a transfer test. A subset of `datasets`, never all of them.
    heldout_datasets: list[str] | None = None
    rebalance_every: int = Field(default=500, ge=10, le=100_000)
    log_every: int = Field(default=200, ge=10, le=5000)  # ≥10 bounds the event stream
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    weights: LossWeightsInput = Field(default_factory=LossWeightsInput)
    render: bool = True  # render figures + video after evaluation

    def resolved_datasets(self) -> list[str]:
        """The datasets a new run trains on: `datasets` if given, else `[dataset]`.
        Order-preserving and deduplicated; empty only when neither was sent."""
        names = self.datasets if self.datasets else ([self.dataset] if self.dataset else [])
        return list(dict.fromkeys(names))

    @model_validator(mode="after")
    def _check_target(self) -> RunLaunchRequest:
        if self.resume and not self.run_id:
            raise ValueError("resume requires run_id")
        if not self.resume and not self.resolved_datasets():
            raise ValueError("a new run requires dataset or datasets")
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

    Children are ordinary runs (train + evaluate; rendering defaults off; a
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


class PerFieldArch(BaseModel):
    """A per-field architecture override; unset keys fall back to the globals."""

    model_config = {"extra": "forbid"}  # a stray/typo'd key is a client bug, not silence

    hidden: int | None = Field(default=None, ge=8, le=1024)
    layers: int | None = Field(default=None, ge=1, le=32)


class ModelArchitecture(BaseModel):
    """The PINN field-ensemble architecture (from the model config)."""

    fields: list[str]
    hidden: int
    layers: int
    fourier_feats: int
    fourier_scale: float
    alpha_eps: float
    nodewise_activation: bool
    per_field: dict[str, PerFieldArch] = Field(default_factory=dict)


class ModelUpdate(BaseModel):
    """A model-architecture edit: any global left unset keeps its current value;
    ``per_field`` maps a field name to its width/depth override.

    Bounds mirror ``datasets.MODEL_ARCH_FIELDS`` so field-level 422s and the
    service's 400s agree on the same limits.
    """

    model_config = {"extra": "forbid"}

    hidden: int | None = Field(default=None, ge=8, le=1024)
    layers: int | None = Field(default=None, ge=1, le=32)
    fourier_feats: int | None = Field(default=None, ge=4, le=512)
    fourier_scale: float | None = Field(default=None, ge=0.1, le=100.0)
    alpha_eps: float | None = Field(default=None, ge=1e-4, le=1.0)
    per_field: dict[str, PerFieldArch] | None = None


class EquationState(BaseModel):
    """One governing equation as the Physics view shows it: its metadata from the
    registry plus whether it is currently active and its loss weight."""

    id: str
    name: str
    stage: Literal["A", "B"]
    tex: str
    weight_key: str
    fields_required: list[str]
    fields_added: list[str]
    groups: list[str]
    core: bool
    enabled: bool
    weight: float


class PhysicsState(BaseModel):
    """The equation set for a dataset, its resulting fields, and the live groups."""

    dataset: str
    equations: list[EquationState]
    fields: list[str]
    groups: dict[str, float]


class PhysicsUpdate(BaseModel):
    """A physics edit: which toggleable Stage-B equations are on, and any
    Stage-B per-equation loss-weight overrides."""

    model_config = {"extra": "forbid"}

    enabled: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)


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
