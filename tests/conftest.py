"""Shared fixtures.

The fast tests compose the real config and exercise the real physics and
networks, but never touch the raw dataset or a trained checkpoint. Tests that
do need those are marked ``needs_data`` and skip cleanly when it is absent.
"""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir

from naviernet.config import config_dir, register_configs

register_configs()


def make_config(overrides: list[str] | None = None):
    """Compose the real config, as the CLI would, with optional overrides."""
    with initialize_config_dir(config_dir=str(config_dir()), version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides or []))


@pytest.fixture
def cfg():
    """Default composed config."""
    return make_config()


@pytest.fixture
def tiny_cfg(tmp_path):
    """A deliberately small model and run, rooted in a temporary directory."""
    return make_config(
        [
            f"paths.root={tmp_path}",
            "model.hidden=8",
            "model.layers=2",
            "model.fourier_feats=4",
            "training.steps=2",
            "training.n_data=16",
            "training.n_coll=16",
            "training.n_bc=8",
        ]
    )


@pytest.fixture
def paths(cfg):
    from naviernet.utils.paths import RunPaths

    return RunPaths.from_config(cfg)


@pytest.fixture
def trained(cfg, paths):
    """A real trained model and dataset, or a skip if the run has not been made."""
    if not paths.tensors.exists() or not paths.checkpoint.exists():
        pytest.skip("no preprocessed tensors / checkpoint; run `make all` first")
    from naviernet.training import load_model

    model, data, _ = load_model(cfg, paths)
    return model, data
