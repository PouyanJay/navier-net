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
    # Editable display name; None = show the id. The id stays the immutable key.
    label: str | None = None
    dataset: str | None = None
    status: str  # "running" | "trained" | "failed" | "empty"
    steps: int | None = None  # completed training steps, if known
    iou_holdout: float | None = None  # single-dataset holdout-frame IoU, if evaluated
    datasets: list[str] = Field(default_factory=list)  # every dataset the run spans
    heldout_datasets: list[str] = Field(default_factory=list)  # axis-B conditions
    val_iou_mean: float | None = None  # joint runs' in-distribution (axis-A) IoU
    # The two metrics every evaluated run writes, whatever its vintage. Without
    # them the list has no number to lead with at all: `iou_holdout` belongs to
    # the retired single-frame holdout and `val_iou_mean` only to joint runs, so
    # on a single-series run both are null and the row led with nothing.
    iou_val: float | None = None  # in-distribution validation IoU
    iou_mean: float | None = None  # mean IoU over the evaluated frames
    # How many frames that mean covers. Two runs evaluated over different frame
    # counts are different measurements, not a better and a worse one, so the
    # list needs this to say which of its rows may honestly be ranked together.
    n_frames: int | None = None
    # The run's own seed and the recipe it ran, read from its config snapshot.
    # `recipe` is None when the run wrote no snapshot (a CLI-era or bench run) --
    # distinct from an empty list, which means "the recommended recipe, exactly".
    seed: int | None = None
    recipe: list[str] | None = None
    date: str | None = None  # ISO timestamp of the run directory
    steps_done: int | None = None  # live progress while the server trains it
    steps_total: int | None = None


class ArtifactFlags(BaseModel):
    """Which deliverables exist for a run."""

    checkpoint: bool = False
    metrics: bool = False
    groups: bool = False
    video: bool = False
    figures: list[str] = []
    # Whether the evaluate stage wrote a front-velocity report. Without it the
    # UI could only find out by asking for the report and reading a 404, so the
    # tab that holds it looked identical to one with six panels of content.
    front_velocity: bool = False
    # Whether the run kept the config snapshot its architecture is rebuilt from.
    # A run without one keeps every measurement it made and can never be
    # replayed, which is a different thing from a run that failed.
    config: bool = False


class RunDetail(BaseModel):
    """Full detail for a single run."""

    id: str
    label: str | None = None  # editable display name; None = show the id
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
    """An editable display name for a series or a run. Blank clears it back to
    the id, which stays the immutable key in both cases."""

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
    # Accuracy techniques (opt-in; every default is a no-op, so an unchanged request
    # trains exactly as before). The fine sub-parameters (rba_gamma/eta, causal chunks/
    # eps schedule, resample cadence) keep their schema defaults -- CLI-tunable only.
    # See the README "Accuracy techniques" note.
    weighting: Literal["gradnorm", "rba"] = "gradnorm"
    causal_weighting: bool = False
    causal_mode: Literal["weight", "march"] = "weight"
    adaptive_collocation: bool = False
    # Hard root pin: anchor the interface at the dataset's measured nucleation
    # site for all t (the anchor is data-derived; only the gate scale is a knob).
    hard_pin: bool = False
    pin_d_ref: float = Field(default=0.1, gt=0.0, le=2.0)
    # Front geometry (R3): the interface as a geometric capsule -- exact root,
    # monotone nose, single connected shape at every t. Mutually exclusive with
    # hard_pin (the geometry pins exactly by construction; the trainer rejects
    # the combination, mirrored below as a 422).
    # ON by default: the recommended physics recipe. The capsule interface is what
    # makes the interface conditions below expressible at all, and it is the
    # recipe every measured gain in this line of work was built on.
    front_geometry: bool = True
    # Sharp-interface physics (R4): drive the shape with the conditions that hold
    # AT the interface -- the Young-Laplace jump and the kinematic condition on
    # the explicit front -- with depth-averaged Darcy in place of the 2-D momentum
    # residual. Requires front_geometry (there is no front to sample without it),
    # mirrored below as a 422.
    # `None` means "apply the recommended default where the series can support it"
    # -- these need the Stage-B field set, so a Stage-A series simply does not get
    # them. Asking for one EXPLICITLY on a series that cannot support it is an
    # error, not a silent downgrade.
    sharp_interface: bool | None = None
    # Film pressure: at the bubble's sides the capillary pressure is balanced by
    # the Bretherton film, not the bulk the depth-averaged `p` represents. One
    # inferred scalar offset on the body. Requires sharp_interface.
    film_pressure: bool | None = None
    # Liquid film delta(x, t): the layer the bubble leaves on the gap walls --
    # deposited by the advancing meniscus (Aussillous-Quere on the local speed),
    # thinned by evaporation through the kinetic resistance, drying out at the
    # roughness scale; its evaporation becomes the mass source and its surface
    # pressure enters the jump. Opt-in until benched (the merged-default-off
    # discipline every physics feature ships under). Requires sharp_interface
    # and the 'T' field.
    liquid_film: bool = False
    # Let the superheat deplete below the inlet, so evaporation can throttle
    # itself as the bubble blankets the wall. Needs the 'T' field.
    depletable_superheat: bool | None = None
    # Let the evaporation mass closure raise the temperature, not only lower the
    # source. Inert without the energy equation, so it needs no gating.
    evap_closure_two_way: bool = True
    # Evolving width: the profile deforms along a learned direction at a rate tied
    # to the bubble's elongation, instead of being a raw function of time that
    # relaxes back toward flat past the data. Requires front_geometry.
    evolving_width: bool = False
    # Let the bubble detach: the radius becomes signed and the nose may retreat,
    # so pinch-off is expressible. Requires front_geometry -- it relaxes that
    # construction's own guarantees.
    allow_pinch: bool = False
    # Cap freedom: the end caps stop being circles. Without it the nose is a
    # constant-curvature arc, which the data cannot fit (a circle cannot comply)
    # and the jump condition cannot reshape (it is handed 1/r rather than reading
    # curvature off the curve). The gate vanishes at the apex and the seam, so the
    # root pin and the monotone nose stay exact. Requires front_geometry.
    cap_freedom: bool = False
    # How far a cap may depart from its circle, as a fraction of the local radius.
    # Bounded on purpose: at 1 the radius reaches zero and the cap self-intersects.
    cap_delta: float = Field(default=0.2, ge=0.0, lt=1.0)
    # Interface sharpening: anneal alpha_eps down to `alpha_eps_final` over this
    # many steps, so a neck is not the same width as the interface blur.
    alpha_eps_anneal_steps: int = Field(default=0, ge=0, le=1_000_000)
    alpha_eps_final: float = Field(default=0.0, ge=0.0, le=1.0)
    # Kinematic growth constraints. The evap-floor weight defaults to 0 here
    # (diverging from the trainer's 1.0 on purpose): the bench showed the floor
    # destabilizes the front, so the platform hands it out only on explicit
    # opt-in. The other fine sub-parameters (kin_time_frac/times/grid/ramp,
    # kin_evap_floor) keep their schema defaults -- CLI-tunable only. The kin
    # weights operate near O(1), so le=100 is already a generous sanity cap
    # (unlike the loss weights' 1e4, which the rebalancer can legitimately reach).
    kinematics: bool = False
    kin_margin_frac: float = Field(default=0.3, ge=0.0, le=2.0)
    kin_weight_mono: float = Field(default=1.0, ge=0.0, le=100.0)
    kin_weight_balance: float = Field(default=1.0, ge=0.0, le=100.0)
    kin_weight_evap: float = Field(default=0.0, ge=0.0, le=100.0)
    # Measured front velocity: supervise the front's own normal speed against the
    # velocity measured from consecutive masks, and the nose apex against its
    # measured 2-D displacement. The shape is already supervised at every training
    # frame; its RATE is not, and the rate is what carries the shape past the data.
    # Requires front_geometry -- there is no explicit front, and no normal speed,
    # without it -- mirrored below as a 422. The measurement's own knobs
    # (fv_samples_per_bin, fv_smooth_px) keep their schema defaults, CLI-tunable
    # only, like the fine kinematics sub-parameters. Both weights sit near O(1).
    front_velocity: bool = False
    # Both default to 10, matching the trainer: these terms sit outside the
    # rebalancer, and at 1.0 they are measurably inert (benched, a weight-1 run
    # reproduced the baseline exactly).
    fv_weight: float = Field(default=10.0, ge=0.0, le=100.0)
    fv_apex_weight: float = Field(default=10.0, ge=0.0, le=100.0)

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

    @model_validator(mode="after")
    def _check_accuracy_combo(self) -> RunLaunchRequest:
        # Mirror the trainer's own guards so an invalid combination fails at the API
        # boundary (422) with an actionable message, not deep in the worker.
        if self.weighting == "rba" and self.causal_weighting:
            raise ValueError("weighting='rba' is not compatible with causal_weighting yet")
        if self.adaptive_collocation and self.weighting != "rba":
            raise ValueError("adaptive_collocation requires weighting='rba'")
        if self.front_geometry and self.hard_pin:
            raise ValueError(
                "front_geometry already pins the root exactly by construction; "
                "it is mutually exclusive with hard_pin -- disable one."
            )
        for flag in (
            "sharp_interface",
            "allow_pinch",
            "evolving_width",
            "front_velocity",
            "cap_freedom",
        ):
            if getattr(self, flag) and not self.front_geometry:
                raise ValueError(
                    f"{flag} requires front_geometry -- enable it, or turn {flag} off"
                )
        if self.front_velocity and self.fv_weight == 0 and self.fv_apex_weight == 0:
            raise ValueError(
                "front_velocity with both fv_weight and fv_apex_weight at 0 measures "
                "the front and then ignores it -- give one a weight, or turn "
                "front_velocity off"
            )
        if self.film_pressure and self.sharp_interface is False:
            raise ValueError(
                "film_pressure corrects the Young-Laplace jump, so it requires "
                "sharp_interface -- enable it, or turn film_pressure off"
            )
        if self.liquid_film and self.sharp_interface is False:
            raise ValueError(
                "liquid_film rides on the explicit front the sharp-interface "
                "conditions sample, so it requires sharp_interface -- enable it, "
                "or turn liquid_film off"
            )
        if self.alpha_eps_anneal_steps and not self.alpha_eps_final:
            raise ValueError(
                "alpha_eps_anneal_steps needs alpha_eps_final: a target of 0 would "
                "divide phi by zero rather than sharpen the interface"
            )
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
    # Which interface treatment the equation belongs to: "any", "diffuse" (bulk
    # residual over a smeared alpha) or "sharp" (imposed on the explicit front).
    mode: str


class PhysicsState(BaseModel):
    """The equation set for a dataset, its resulting fields, and the live groups."""

    dataset: str
    equations: list[EquationState]
    fields: list[str]
    groups: dict[str, float]
    # Whether this series composes the sharp-interface treatment, which decides
    # which of the mode-specific equations above are active.
    sharp_interface: bool


class PhysicsUpdate(BaseModel):
    """A physics edit: which toggleable Stage-B equations are on, and any
    Stage-B per-equation loss-weight overrides."""

    model_config = {"extra": "forbid"}

    enabled: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)


class DatasetAgreement(BaseModel):
    """One dataset's agreement block inside a joint run's v2 metrics."""

    iou_mean: float | None = None
    iou_val: float | None = None  # axis A: this dataset's validation-frame IoU
    validation_frames: list[int] = Field(default_factory=list)
    iou_per_frame: dict[str, float] = Field(default_factory=dict)


class PhysicsFrame(BaseModel):
    """One frame's shape diagnostics, model against the measured mask."""

    frame: int
    neck_depth_model: float
    neck_depth_measured: float
    neck_location_model: float
    neck_location_measured: float
    half_width_model: list[float]
    half_width_measured: list[float]


class ResidualConvergence(BaseModel):
    """Whether one physics residual descended over its active window.

    ``ratio`` is absent when it could not be formed -- a term whose first window
    averaged exactly zero has nothing to be a ratio of.
    """

    first: float
    last: float
    ratio: float | None = None


class PhysicsDiagnostics(BaseModel):
    """The measurements IoU cannot make: whether the solution obeys the physics.

    Present only for runs with an explicit front (`model.front_geometry`); the
    jump figures are NaN-free-or-absent when the run has no pressure field.
    """

    laplace_error_nose: float | None = None
    laplace_error_front: float | None = None
    axial_capillary_gradient: float | None = None
    neck_depth_model: float | None = None
    neck_depth_measured: float | None = None
    neck_location_model: float | None = None
    neck_location_measured: float | None = None
    profile_stations: list[float] = Field(default_factory=list)
    per_frame: list[PhysicsFrame] = Field(default_factory=list)
    residual_convergence: dict[str, ResidualConvergence] = Field(default_factory=dict)


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
    # Two-axis validation (metrics.json v2); None/empty on legacy runs.
    val_iou_mean: float | None = None  # axis A across training datasets
    iou_val: float | None = None  # axis A of a single-series run
    validation_frames: list[int] = Field(default_factory=list)
    transfer_iou_mean: float | None = None  # axis B across held-out datasets
    transfer_per_dataset: dict[str, float] | None = None
    transfer_per_frame: dict[str, dict[str, float]] | None = None
    per_dataset: dict[str, DatasetAgreement] | None = None
    training_datasets: list[str] | None = None
    heldout_datasets: list[str] | None = None
    # The physics diagnostics, when the run has an explicit front. None for a run
    # trained without one -- there is no interface to measure the conditions on.
    physics: PhysicsDiagnostics | None = None
