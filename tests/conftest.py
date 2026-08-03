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


# --- Synthetic growth-event fixtures (shared by the hard-pin and kinematics
# --- suites): a bubble whose root edge stays fixed while the front advances.

GROWING_H, GROWING_W, GROWING_FRAMES = 6, 12, 4
ROOT_COL = 2  # the stationary (nucleation-side) bubble edge, all frames
VAPOR_ROWS = slice(2, 4)  # the bubble's y-extent at the root

# Joint-run datasets: distinct regimes (q_wall), distinct roots, and distinct
# growth rates (columns per frame), so per-dataset anchors AND per-dataset
# growth-rate references provably differ.
JOINT_SPECS = (("ds_a", 2.0, 2, 1), ("ds_b", 5.0, 3, 2))


def write_growing_bubble(
    path, n_frames: int = GROWING_FRAMES, root_col: int = ROOT_COL, growth: int = 1, groups=None
) -> None:
    """A synthetic growth event: the root edge fixed at ``root_col``, the front
    advancing ``growth`` columns per frame. ``groups`` (when given) is recorded
    as preprocess would, so the dataset can join a conditioned run."""
    import json

    import numpy as np

    alpha = np.zeros((n_frames, GROWING_H, GROWING_W), dtype=np.float32)
    for k in range(n_frames):
        alpha[k, VAPOR_ROWS, root_col : root_col + 3 + k * growth] = 1.0
    meta = {
        "x_pin_star": 0.2,
        "t_ref_ms": 1.5,
        "n_frames_usable": n_frames,
        "n_frames_event": n_frames,
        "frame_numbers": list(range(1, n_frames + 1)),
    }
    if groups is not None:
        meta["groups"] = groups
    np.savez_compressed(
        path,
        alpha=alpha,
        sdf=((0.5 - alpha) * 0.1).astype(np.float32),
        valid=np.ones_like(alpha),
        masks_camera=(alpha > 0.5).astype(np.uint8),
        x_star=np.linspace(0, 1.1, GROWING_W, dtype=np.float32),
        y_star=np.linspace(0, 0.5, GROWING_H, dtype=np.float32),
        t_star=(np.arange(n_frames) * 0.1).astype(np.float32),
        meta=json.dumps(meta),
    )


_TINY_RUN = [
    "model.hidden=8",
    "model.layers=2",
    "model.fourier_feats=4",
    "training.steps=2",
    "training.n_data=16",
    "training.n_coll=16",
    "training.n_bc=8",
    "training.holdout_frame=-1",
]


def staged_run(tmp_path, overrides=None, write=None):
    """A tiny single-dataset run over the synthetic growth event, staged as the
    CLI would: composed config + tensors written under a tmp root.

    ``write`` swaps in a different tensor writer for a suite that needs different
    geometry -- the front-velocity tests need a TRUE signed distance field, which
    :func:`write_growing_bubble`'s placeholder is not. Only the writer varies; the
    tiny-run overrides and the staging steps stay in one place, so a new required
    override reaches every suite.
    """
    from naviernet.utils.paths import RunPaths

    cfg = make_config([f"paths.root={tmp_path}", *_TINY_RUN, *(overrides or [])])
    paths = RunPaths.from_config(cfg)
    paths.ensure()
    paths.tensors.parent.mkdir(parents=True, exist_ok=True)
    (write or write_growing_bubble)(paths.tensors)
    return cfg, paths


def staged_joint_run(tmp_path, overrides=None):
    """Two synthetic datasets with distinct regimes, roots, AND growth rates,
    staged for a conditioned joint run, as preprocess would have left them."""
    from naviernet.physics.groups import compute_groups
    from naviernet.utils.paths import RunPaths

    names = ",".join(name for name, _, _, _ in JOINT_SPECS)
    cfg = make_config(
        [f"paths.root={tmp_path}", f"datasets=[{names}]", *_TINY_RUN, *(overrides or [])]
    )
    paths = RunPaths.from_config(cfg)
    paths.ensure()
    for name, q_wall, root_col, growth in JOINT_SPECS:
        ds_paths = paths.for_dataset(name)
        ds_paths.processed_dir.mkdir(parents=True, exist_ok=True)
        groups = compute_groups(make_config([f"experiment.q_wall_W_cm2={q_wall}"]))
        write_growing_bubble(ds_paths.tensors, root_col=root_col, growth=growth, groups=groups)
    return cfg, paths
