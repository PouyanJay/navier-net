"""The config composes, validates, and derives its paths correctly."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError, ValidationError

from naviernet.config.schema import STAGES, resolved_datasets, training_datasets
from naviernet.utils.paths import RunPaths

from .conftest import make_config


def test_composes_with_expected_values(cfg):
    assert cfg.experiment.fluid == "FC-72"
    assert cfg.fluid.rho_l == pytest.approx(1601.6)
    assert cfg.scales.L_ref_um == pytest.approx(300.0)
    assert list(cfg.model.fields) == ["phi", "u", "v", "s"]


def test_frame_counts_are_self_consistent(cfg):
    exp = cfg.experiment
    assert exp.n_frames_event <= exp.n_frames_usable <= exp.n_frames_raw


def test_holdout_frame_is_inside_the_event(cfg):
    assert -1 <= cfg.training.holdout_frame < cfg.experiment.n_frames_event


def test_paths_interpolate_from_dataset_and_run_name():
    cfg = make_config(["dataset=demo", "run_name=trial", "paths.root=/tmp/root"])
    paths = RunPaths.from_config(cfg)

    assert str(paths.raw_dir) == "/tmp/root/data/raw/demo"
    assert str(paths.processed_dir) == "/tmp/root/data/processed/demo"
    assert str(paths.output_dir) == "/tmp/root/outputs/trial"
    assert paths.checkpoint.name == "ckpt.pt"
    assert paths.raw_frame(3).name == "3.tif"


def test_run_name_defaults_to_dataset():
    cfg = make_config(["dataset=other"])
    assert cfg.run_name == "other"


def test_datasets_defaults_to_the_single_dataset():
    # Back-compat: with no `datasets`, a run trains on its one `dataset`, so every
    # existing single-dataset run is just the 1-element case of the list form.
    cfg = make_config(["dataset=solo"])
    assert resolved_datasets(cfg) == ["solo"]


def test_datasets_list_overrides_the_single_dataset():
    cfg = make_config(["datasets=[first,second]"])
    assert resolved_datasets(cfg) == ["first", "second"]


def test_datasets_list_is_order_preserving_and_deduplicated():
    cfg = make_config(["datasets=[b,a,b]"])
    assert resolved_datasets(cfg) == ["b", "a"]


def test_validation_axes_default_to_off():
    # Back-compat: the new axes are inert unless dialled up, so every existing
    # run composes byte-for-byte as before (val_fraction=0, no held-out datasets).
    cfg = make_config(["dataset=solo"])
    assert cfg.training.val_fraction == pytest.approx(0.0)
    assert cfg.training.val_strategy == "tail"
    assert list(cfg.heldout_datasets) == []


def test_validation_split_composes_with_holdout_frame():
    # Axis A (a per-dataset frame fraction) and the legacy single holdout frame
    # are independent, additive hold-outs -- both can be set at once.
    cfg = make_config(["training.val_fraction=0.2", "training.val_strategy=scatter"])
    assert cfg.training.val_fraction == pytest.approx(0.2)
    assert cfg.training.val_strategy == "scatter"
    assert cfg.training.holdout_frame >= -1  # legacy knob still present


def test_training_datasets_excludes_held_out_conditions():
    # Axis B: whole datasets kept OUT of training. training_datasets = resolved − held-out.
    cfg = make_config(["datasets=[a,b,c]", "heldout_datasets=[c]"])
    assert resolved_datasets(cfg) == ["a", "b", "c"]
    assert training_datasets(cfg) == ["a", "b"]


def test_training_datasets_is_all_datasets_when_none_held_out():
    cfg = make_config(["datasets=[a,b]"])
    assert training_datasets(cfg) == ["a", "b"]


def test_all_datasets_held_out_is_rejected():
    # Nothing left to train on -> fail loudly rather than silently no-op.
    cfg = make_config(["datasets=[a,b]", "heldout_datasets=[a,b]"])
    with pytest.raises(ValueError, match="training"):
        training_datasets(cfg)


def test_held_out_dataset_must_be_one_of_the_datasets():
    cfg = make_config(["datasets=[a,b]", "heldout_datasets=[z]"])
    with pytest.raises(ValueError, match="not in"):
        training_datasets(cfg)


def test_ensure_creates_every_writable_directory(tmp_path):
    cfg = make_config([f"paths.root={tmp_path}"])
    paths = RunPaths.from_config(cfg).ensure()

    for directory in (paths.processed_dir, paths.figures_dir, paths.checkpoints_dir):
        assert directory.is_dir()


def test_unknown_key_is_rejected(cfg):
    """The schema is a struct: typos fail loudly rather than being ignored."""
    with pytest.raises(ConfigAttributeError):
        _ = cfg.experiment.no_such_key


def test_wrong_type_is_rejected(cfg):
    with pytest.raises(ValidationError):
        cfg.training.steps = "not a number"


def test_stage_list_matches_pipeline_methods():
    from naviernet.pipeline import Pipeline

    for stage in STAGES:
        assert callable(getattr(Pipeline, stage))


def test_config_is_serialisable(cfg):
    """Every value resolves, so Hydra can snapshot the run."""
    dumped = OmegaConf.to_container(cfg, resolve=True)
    assert dumped["run_name"] == "highest_t"
