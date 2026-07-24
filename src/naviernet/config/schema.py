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
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# Stages of the pipeline, in dependency order. ``all`` runs each in turn.
STAGES = ("preprocess", "train", "evaluate", "figures", "video")


@dataclass
class ExperimentConfig:
    """Operating conditions of one imaged ebullition dataset."""

    name: str = MISSING
    fluid: str = MISSING
    T_sat_C: float = MISSING  # saturation temperature at operating pressure
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
    # Columns masked out on the last usable frame, where the bubble leaves the
    # field of view (expressed in flipped, downstream-positive coordinates).
    truncated_cols: int = 0


@dataclass
class ScalesConfig:
    """Non-dimensionalisation. ``x*`` runs downstream with the inlet at zero."""

    L_ref_um: float = MISSING
    U_ref: float = MISSING  # m/s; measured nose-speed scale


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


@dataclass
class LossWeights:
    """Initial multipliers; ``vof``/``div``/``bc`` are rebalanced during training."""

    data: float = MISSING
    vof: float = MISSING
    div: float = MISSING
    src: float = MISSING
    bc: float = MISSING


@dataclass
class TrainingConfig:
    steps: int = MISSING
    lr: float = MISSING
    lr_halflife: int = MISSING  # steps between learning-rate halvings

    n_data: int = MISSING  # supervised points per step
    n_coll: int = MISSING  # PDE collocation points per step
    n_bc: int = MISSING  # boundary points per step

    # Frame withheld from supervision entirely (0-based) and used as the
    # honest generalisation test. Set to -1 to train on every frame.
    holdout_frame: int = MISSING

    weights: LossWeights = MISSING
    rebalance_every: int = MISSING  # gradient-norm loss rebalancing period

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
