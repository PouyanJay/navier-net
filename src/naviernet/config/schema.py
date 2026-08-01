"""Typed configuration schema.

These dataclasses are the *single source of truth* for every tunable quantity in
the platform: experimental conditions, fluid properties, imaging parameters,
non-dimensionalisation, network architecture, and optimisation settings.

They are registered with Hydra's :class:`~hydra.core.config_store.ConfigStore`,
so the YAML files under ``configs/`` are type-checked at composition time: a
typo in a key name or a string where a float belongs fails immediately with a
clear error rather than silently producing wrong physics.

Derived quantities (reference time, dimensionless groups) deliberately do *not*
live here -- they are computed from these inputs in :mod:`naviernet.physics`,
so they can never drift out of sync with the values they depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# Stages of the pipeline, in dependency order. ``all`` runs each in turn.
STAGES = ("preprocess", "train", "evaluate", "figures", "video")


@dataclass
class ExperimentConfig:
    """Operating conditions of one imaged ebullition dataset."""

    name: str = MISSING
    # Both derived from the selected fluid group (``${fluid.name}`` /
    # ``${fluid.T_sat_C}``): the working fluid fixes its own label and its
    # atmospheric saturation temperature. Kept on the experiment so every
    # existing consumer (groups, residuals, API) reads one stable path.
    fluid: str = MISSING
    T_sat_C: float = MISSING  # saturation temperature at operating pressure (from fluid)
    q_wall_W_cm2: float = MISSING  # bottom-wall heat flux setpoint
    flow_rate_mL_hr: float = MISSING
    channel_width_um: float = MISSING  # imaged (in-plane) direction
    channel_height_um: float = MISSING  # depth; bottom wall is heated
    dt_frame_ms: float = MISSING  # camera inter-frame time
    flow_direction: str = MISSING  # direction of flow in the raw camera frames

    n_frames_raw: int = MISSING  # TIFFs present on disk
    n_frames_usable: int = MISSING  # end of the usable window (1..n)
    n_frames_event: int = MISSING  # frames of one continuous growth event

    # 1-based camera frames dropped from the usable window (a mis-segmented
    # frame, a stray reflection). Absolute frame times are preserved, so the
    # frames that remain keep their true t* -- only tensor rows disappear.
    excluded_frames: list[int] = field(default_factory=list)

    notes: str = ""


@dataclass
class FluidConfig:
    """Saturated two-phase properties at ``T_sat`` (SI units)."""

    name: str = MISSING
    # Saturation temperature at the operating pressure (atmospheric). A property
    # of the fluid, not a free operating input: the experiment derives its
    # ``T_sat_C`` from this, so selecting a fluid sets the saturation point.
    T_sat_C: float = MISSING  # deg C
    rho_l: float = MISSING  # kg/m^3
    rho_v: float = MISSING
    mu_l: float = MISSING  # Pa.s
    mu_v: float = MISSING
    k_l: float = MISSING  # W/m/K
    k_v: float = MISSING
    cp_l: float = MISSING  # J/kg/K
    cp_v: float = MISSING
    sigma: float = MISSING  # N/m
    h_lv: float = MISSING  # J/kg


@dataclass
class ImagingConfig:
    """Segmentation and calibration parameters for the raw TIFF frames."""

    # Row band searched for the two channel walls (gives the um/px calibration).
    wall_search_rows: list[int] = MISSING
    wall_margin_top: int = 3  # rows trimmed inside the upper wall
    wall_margin_bottom: int = 2  # rows trimmed inside the lower wall
    dark_thresh: int = MISSING  # intensity below which a pixel is "dark"
    open_kernel: int = MISSING  # morphological opening; removes heater traces
    close_kernel: int = MISSING  # morphological closing; seals the bubble ring
    # The imaged bubble edge is a thick dark rim, not a line; the interface is
    # its centreline. Below this interior-hole fraction the rim has no enclosed
    # interior to centre in, so the filled outline is used instead.
    min_rim_hole_fraction: float = 0.05
    # Pixel scale that seeds the active contour (blur on the medial field it
    # starts from) and smooths the solid-nucleus fallback. The interface's own
    # smoothness comes from the contour's rigidity, not this. 0 disables the blur.
    contour_smooth_px: float = 5.0
    # Columns masked out on the last usable frame, where the bubble leaves the
    # field of view (expressed in flipped, downstream-positive coordinates).
    truncated_cols: int = 0


@dataclass
class ScalesConfig:
    """Non-dimensionalisation. ``x*`` runs downstream with the inlet at zero."""

    L_ref_um: float = MISSING
    U_ref: float = MISSING  # m/s; measured nose-speed scale


@dataclass
class FieldArch:
    """Per-field architecture override. Any field left ``None`` falls back to the
    global ``ModelConfig`` value, so a per-field entry stores only its deltas."""

    hidden: Optional[int] = None  # noqa: UP045 -- OmegaConf needs Optional, not X | None
    layers: Optional[int] = None  # noqa: UP045
    fourier_feats: Optional[int] = None  # noqa: UP045
    fourier_scale: Optional[float] = None  # noqa: UP045


@dataclass
class ModelConfig:
    """Fourier-feature MLP ensemble -- one network per physical field."""

    hidden: int = MISSING
    layers: int = MISSING
    fourier_feats: int = MISSING
    fourier_scale: float = MISSING
    # Interface half-thickness in alpha = sigmoid(phi / alpha_eps). Annealing
    # this downwards is the interface-sharpening curriculum.
    alpha_eps: float = MISSING
    nodewise_activation: bool = True  # per-neuron adaptive tanh slopes
    fields: list[str] = MISSING
    # Optional per-field architecture overrides, keyed by field name. Absent =
    # every field uses the global architecture above (Stage-A behaviour).
    per_field: dict[str, FieldArch] = field(default_factory=dict)
    # Localized nucleation heat pulse (Stage B; requires the `T` field). Off by default
    # -> the energy residual is unchanged. When on, a brief localized pulse at the fixed
    # cavity seeds the bubble (see residuals.NucleationPulse): its location (x_pin, from
    # the data) and width/timing are fixed priors; only its magnitude is learnable.
    nucleation_pulse: bool = False
    pulse_t0: float = 0.1  # pulse active for the first this-fraction of the time window
    pulse_width: float = 0.05  # streamwise Gaussian sigma, as a fraction of the domain width
    # Hard root pin: transform the level set so the interface (alpha = 0.5) passes
    # through the dataset's measured bubble-root anchor at EVERY time, including
    # times beyond the training window -- phi = tanh(dist/pin_d_ref) * N. Off by
    # default -> phi is the raw network output, byte-for-byte. The anchor is
    # derived from the data at load time, never configured by hand.
    hard_pin: bool = False
    pin_d_ref: float = 0.1  # saturation scale of the pin, in non-dim x* units
    # Front geometry (R3): replace the free level-set net with a geometric
    # construction -- monotone nose, bounded width envelope, bounded centerline --
    # so a single connected, root-anchored capsule is the ONLY expressible shape.
    # Fixes the extrapolation shape collapse structurally (the free level set kept
    # volume and the pinned point but rounded/fragmented the SHAPE off-data).
    # Mutually exclusive with hard_pin (the geometry pins exactly by construction).
    front_geometry: bool = False


@dataclass
class LossWeights:
    """Initial multipliers; ``vof``/``div``/``bc`` (and Stage-B ``mom``/``energy``)
    are rebalanced during training. Stage-B terms default to 1.0 so Stage-A
    configs compose unchanged -- they are inert while their equations are off."""

    data: float = MISSING
    vof: float = MISSING
    div: float = MISSING
    src: float = MISSING
    bc: float = MISSING
    mom: float = 1.0  # Stage B: momentum
    energy: float = 1.0  # Stage B: energy
    evap: float = 1.0  # Stage B: evaporation mass-closure penalty


@dataclass
class TrainingConfig:
    steps: int = MISSING
    lr: float = MISSING
    lr_halflife: int = MISSING  # steps between learning-rate halvings

    n_data: int = MISSING  # supervised points per step
    n_coll: int = MISSING  # PDE collocation points per step
    n_bc: int = MISSING  # boundary points per step

    # Frame withheld from supervision entirely (0-based) and used as the
    # honest generalisation test. Set to -1 to train on every frame. This is the
    # legacy single-frame knob; ``val_fraction`` below is the general axis and the
    # two compose (both frames are held out).
    holdout_frame: int = MISSING

    # Validation axis A: a per-dataset *fraction* of event frames held out of
    # supervision as an in-distribution validation set. 0.0 (the default) keeps
    # every existing run byte-for-byte unchanged; the Solver dials this to 0.2 for
    # new joint runs. ``val_strategy`` picks which frames: "tail" holds the last
    # frames (extrapolation, the honest test for a growth series); "scatter" holds
    # evenly spaced frames (interpolation, easier). Held-out frames are the union
    # of this split, the ``holdout_frame``, and (already-dropped) excluded frames.
    # ``val_strategy`` is a plain ``str`` (validated at use in ``BubbleDataset``),
    # not a ``Literal["tail", "scatter"]``: OmegaConf rejects ``Literal`` annotations
    # in structured configs, so ``str`` + a runtime check is the supported pattern.
    # NB the *aggregate* val IoU is surfaced only for joint runs (``evaluate_joint``);
    # a single-dataset CLI run honours ``val_fraction`` in training but ``evaluate``
    # does not report a separate ``iou_val`` for it.
    val_fraction: float = 0.0
    val_strategy: str = "tail"  # "tail" (extrapolation) | "scatter" (interpolation)

    weights: LossWeights = MISSING
    rebalance_every: int = MISSING  # gradient-norm loss rebalancing period
    # Stage B: steps over which the evaporation mass-closure ramps in while the
    # off-interface source penalty decays (soft -> hard curriculum). 0 disables
    # the schedule (the penalty and closure keep their static weights).
    curriculum_steps: int = 0
    # Stage B: steps to train the Stage-A objective alone before the Stage-B
    # physics (momentum, energy, evaporation) engages -- an in-run warm start, so
    # those terms act on an interface that has already converged instead of on a
    # random field (which collapses it). 0 disables the gate (Stage-B on from
    # step 1); a resumed run whose step count is past the gate is unaffected.
    stage_b_warmup_steps: int = 0

    # --- Accuracy techniques (opt-in; every default is a no-op, so an unchanged
    # config trains byte-for-byte as before). See
    # .claude/plans/pinn-accuracy-causality-adaptive-weighting-plan.md ---
    # Problem 1 -- temporal causal weighting (arXiv:2203.07404): weight the PDE
    # residual by time so the model learns causally (early times first), which
    # fixes the late-frame extrapolation bias. False -> uniform-in-time (today).
    causal_weighting: bool = False
    causal_time_chunks: int = 16  # number of ordered time bins the residual is split into
    # ε annealing schedule -- steeper causal enforcement as training proceeds.
    causal_eps_schedule: list[float] = field(
        default_factory=lambda: [1e-2, 1e-1, 1.0, 10.0, 100.0]
    )
    # Causal variant. "weight": soft per-time exp weights (uses causal_eps_schedule).
    # "march": a time-marching curriculum -- enforce the collocation residual only up
    # to a horizon that expands from the earliest bin to the whole domain, so late
    # times are trained at FULL weight once the front reaches them (causal_eps_schedule
    # is unused in this mode). See arXiv:2203.07404 §time-marching.
    causal_mode: str = "weight"  # "weight" | "march"
    # Fraction of the run over which the marching horizon expands to the full time
    # domain; it stays at the full domain afterwards (so late frames get real
    # full-weight training). Only used when causal_mode="march".
    causal_march_full_frac: float = 0.5
    # Problem 2 -- loss-weighting scheme. "gradnorm" is the gradient-norm rebalancer
    # (unbounded; ratchets to the 1e3 cap on long runs); "rba" is bounded residual-
    # based attention (arXiv:2307.00379). Default keeps existing behaviour until
    # Phase 2 makes "rba" the default.
    weighting: str = "gradnorm"  # "gradnorm" | "rba"
    rba_gamma: float = 0.999  # RBA EMA decay on per-point attention
    rba_eta: float = 0.01  # RBA update rate
    # Problem 3 -- residual-adaptive collocation (RAR): periodically resample
    # collocation points toward high-residual regions (interface, late times).
    # False -> uniform sampling (today).
    adaptive_collocation: bool = False
    resample_every: int = 500  # steps between collocation resamples
    resample_fraction: float = 0.5  # fraction of the batch drawn from high-residual regions

    # Kinematic growth constraints (R2): physics-only scalar functionals on a fixed
    # quadrature grid over the LATE time window, targeting the diagnosed
    # extrapolation failure (the dilatation source collapses at the supervision
    # boundary and predicted dV/dt goes negative while the true bubble keeps
    # growing). Off by default -> the objective is unchanged. The terms sit outside
    # causal weighting, RBA, and the rebalancer (they are functionals, not
    # pointwise residuals), normalized by the dataset's supervised-tail growth rate.
    kinematics: bool = False
    kin_weight_mono: float = 1.0  # monotone volume: ReLU(margin - dQ/dt)^2
    kin_weight_balance: float = 1.0  # volume-vs-source balance: (dQ/dt - ∫α·s)^2
    kin_weight_evap: float = 1.0  # evap floor: ReLU(floor·E - ∫α·s)^2 (needs T)
    kin_margin_frac: float = 0.3  # mono margin, as a fraction of the reference rate
    kin_evap_floor: float = 0.5  # require the source to carry ≥ this × the evap drive
    kin_time_frac: float = 0.4  # quadrature window = last this-fraction of [t_min, t_max]
    kin_times: int = 12  # fixed time slices in the window
    kin_grid: int = 48  # fixed spatial quadrature grid, per side
    kin_ramp: int = 300  # steps to ramp balance/evap in (the source field is noise early)

    seed: int = 0
    device: str = "cpu"
    log_every: int = 200


@dataclass
class EvaluationConfig:
    stride: int = 4  # pixel stride for gridded prediction
    n_traj_points: int = 61  # timesteps on the continuous nose trajectory
    threshold: float = 0.5  # alpha level defining the interface


@dataclass
class VideoConfig:
    n_timesteps: int = 100
    fps: int = 22
    width: int = 1600
    hold_frames: int = 14  # duplicated tail frames so the video ends on a beat
    background_dark_thresh: int = 110  # for the inpainted background plate


@dataclass
class PathsConfig:
    """All paths derive from ``root`` plus the dataset and run names.

    ``root`` stays relative by default and Hydra is configured with
    ``job.chdir=false``, so the working directory never changes and commands
    behave identically whether run via the CLI or from a notebook.
    """

    root: str = "."
    raw_dir: str = "${paths.root}/data/raw/${dataset}"
    processed_dir: str = "${paths.root}/data/processed/${dataset}"
    output_dir: str = "${paths.root}/outputs/${run_name}"


@dataclass
class Config:
    """Root configuration object handed to every stage."""

    # Which raw dataset to read, and the name of the output directory. Keeping
    # them separate lets several runs (architectures, seeds, weightings) share
    # one preprocessed dataset without recomputing or overwriting it.
    dataset: str = "highest_t"
    run_name: str = "${dataset}"

    # Joint (transfer-learning) training reads several datasets at once. Empty
    # means "just `dataset`", so every single-dataset run is the 1-element case
    # and existing configs are unchanged. Use `resolved_datasets(cfg)` to read it.
    datasets: list[str] = field(default_factory=list)

    # Validation axis B: whole datasets kept OUT of training and scored on every
    # frame as a transfer test ("can it predict a condition it never trained on?").
    # A subset of ``datasets`` -- never all of them, or nothing is left to train.
    # Possible only because the model is conditioned on dimensionless groups, so a
    # held-out dataset still has a valid conditioning vector. Empty = train on all.
    heldout_datasets: list[str] = field(default_factory=list)

    stage: str = "all"

    experiment: ExperimentConfig = MISSING
    fluid: FluidConfig = MISSING
    imaging: ImagingConfig = MISSING
    scales: ScalesConfig = MISSING
    model: ModelConfig = MISSING
    training: TrainingConfig = MISSING

    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"experiment": "highest_t"},
            {"fluid": "fc72"},
            {"imaging": "default"},
            {"scales": "default"},
            {"model": "stage_a"},
            {"training": "stage_a"},
        ]
    )


def resolved_datasets(cfg) -> list[str]:
    """The datasets a run trains on: ``cfg.datasets`` if set, else ``[cfg.dataset]``.

    Order-preserving and deduplicated, so callers get a clean list whether the run
    is single- or multi-dataset. This is the one place the two config fields are
    reconciled, so every stage sees the same resolved list.
    """
    names = list(cfg.datasets) if cfg.datasets else [cfg.dataset]
    return list(dict.fromkeys(names))


def validate_split(names: list[str], heldout: list[str]) -> list[str]:
    """The training datasets for a run: ``names`` minus the held-out conditions
    (validation axis B), validated.

    Every held-out name must be one of ``names``, and at least one dataset must
    remain to train on. Both failures raise ``ValueError`` -- an invalid split is a
    config error, not a silent no-op. Order-preserving. This is the single source of
    truth for the split rules, shared by the CLI (:func:`training_datasets`) and the
    API's launch validation so the two can never drift.
    """
    heldout = list(dict.fromkeys(heldout))
    unknown = [name for name in heldout if name not in names]
    if unknown:
        raise ValueError(f"held-out datasets are not in the run: {unknown}")

    training = [name for name in names if name not in set(heldout)]
    if not training:
        raise ValueError("cannot hold out every dataset; no training datasets remain")
    return training


def training_datasets(cfg) -> list[str]:
    """The datasets actually supervised: ``resolved_datasets(cfg)`` minus the
    held-out conditions. Thin wrapper over :func:`validate_split`."""
    return validate_split(resolved_datasets(cfg), list(cfg.heldout_datasets))


def register_configs() -> None:
    """Register the schema so ``configs/*.yaml`` is validated against it."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="experiment", name="base_experiment", node=ExperimentConfig)
    cs.store(group="fluid", name="base_fluid", node=FluidConfig)
    cs.store(group="imaging", name="base_imaging", node=ImagingConfig)
    cs.store(group="scales", name="base_scales", node=ScalesConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)
