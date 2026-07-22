"""Configuration schema and Hydra registration."""

import os
from pathlib import Path

from naviernet.config.schema import (
    STAGES,
    Config,
    EvaluationConfig,
    ExperimentConfig,
    FluidConfig,
    ImagingConfig,
    LossWeights,
    ModelConfig,
    PathsConfig,
    ScalesConfig,
    TrainingConfig,
    VideoConfig,
    register_configs,
)


def config_dir() -> Path:
    """Locate the ``configs/`` directory holding the YAML config groups.

    Checked in order: the ``NAVIERNET_CONFIG_DIR`` environment variable, the
    repository root relative to this file (the usual case -- an editable
    install from a checkout), then the current working directory.
    """
    candidates = []
    if env := os.environ.get("NAVIERNET_CONFIG_DIR"):
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parents[3] / "configs")
    candidates.append(Path.cwd() / "configs")

    for candidate in candidates:
        if (candidate / "config.yaml").is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "could not locate the configs/ directory. Run naviernet from a "
        "repository checkout, or set NAVIERNET_CONFIG_DIR. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


__all__ = [
    "STAGES",
    "Config",
    "config_dir",
    "EvaluationConfig",
    "ExperimentConfig",
    "FluidConfig",
    "ImagingConfig",
    "LossWeights",
    "ModelConfig",
    "PathsConfig",
    "ScalesConfig",
    "TrainingConfig",
    "VideoConfig",
    "register_configs",
]
