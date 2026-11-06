"""Studio service test fixtures.

Hermetic: a fake repository root under ``tmp_path`` holds synthetic tensors and
one tiny run trained for real (two steps) through the pipeline, so the studio
evaluates an actual checkpoint without ever touching the developer's outputs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from naviernet_studio.app import create_app
from naviernet_studio.settings import Settings, get_settings

DATASET = "highest_t"
RUN_ID = "tiny-run"
# Two steps over small batches: about a second, enough to write a checkpoint.
TINY_OVERRIDES = (
    "training.steps=2",
    "training.n_data=16",
    "training.n_coll=16",
    "training.n_bc=8",
    "training.log_every=10",
)


def write_synthetic_tensors(path: Path) -> None:
    """A tiny but schema-complete ``tensors.npz``: an 11-frame growing bubble."""
    n_t, height, width = 11, 12, 16
    alpha = np.zeros((n_t, height, width), dtype=np.float32)
    for i in range(n_t):
        alpha[i, 3:9, : 3 + i] = 1.0
    sdf = ((0.5 - alpha) * 0.1).astype(np.float32)
    valid = np.ones_like(alpha)
    x_star = ((np.arange(width) + 0.5) / width).astype(np.float32)
    y_star = ((np.arange(height) + 0.5) / height).astype(np.float32)
    t_star = (np.arange(n_t) * 0.1).astype(np.float32)
    interface_star = np.stack(
        [
            np.array(
                [
                    [x_star[0], y_star[3]],
                    [x_star[min(width - 1, 2 + i)], y_star[3]],
                    [x_star[min(width - 1, 2 + i)], y_star[8]],
                    [x_star[0], y_star[8]],
                ],
                dtype=np.float32,
            )
            for i in range(n_t)
        ]
    )
    meta = {
        "dataset": DATASET,
        "um_per_px": 4.3,
        "L_ref_um": 300.0,
        "U_ref": 0.2,
        "t_ref_ms": 1.5,
        "x_pin_star": float(x_star[1]),
        "n_frames_usable": n_t,
        "n_frames_event": 10,
        "y_roi": [64, 76],
        "wall_rows": [63, 77],
        "y_wall_lo_star": 0.0,
        "y_wall_hi_star": 1.0,
        "y_center_star": 0.5,
        "frame_numbers": list(range(1, n_t + 1)),
        "excluded_frames": [12],
        "baked_conditions": {"dt_frame_ms": 0.5, "channel_width_um": 300.0, "U_ref": 0.2},
        "groups": {
            "u_inlet_star": 0.77,
            "Re": 215.5,
            "We": 2.302,
            "Ca": 0.01068,
            "Bond": 0.0125,
            "Pr": 9.411,
            "Ja": 0.043,
            "rho_ratio": 178.0,
            "mu_ratio": 43.0,
            "hele_shaw": 0.2228,
            "q_wall_star": 0.31,
            "t_star_per_frame": 0.33,
        },
    }
    np.savez_compressed(
        path,
        alpha=alpha,
        sdf=sdf,
        valid=valid,
        x_star=x_star,
        y_star=y_star,
        t_star=t_star,
        masks_camera=(alpha > 0.5).astype(np.uint8),
        interface_star=interface_star,
        meta=json.dumps(meta),
    )


def write_raw_frames(directory: Path, count: int) -> None:
    """Grey 16 x 140 TIFFs, enough for the transform and the PNG route."""
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    for n in range(1, count + 1):
        Image.fromarray(np.full((140, 16), 90 + n, dtype=np.uint8)).save(directory / f"{n}.tif")
    (directory / "conditions.json").write_text(json.dumps({"flow_rate_mL_hr": 5.0}))


def train_tiny_run(root: Path, run_id: str) -> None:
    """Compose the dataset's config over the fake root and train two steps."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from naviernet.config import config_dir, register_configs
    from naviernet.pipeline import Pipeline

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir()), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"dataset={DATASET}",
                f"paths.root={root}",
                f"run_name={run_id}",
                *TINY_OVERRIDES,
            ],
        )
    OmegaConf.set_readonly(cfg, True)
    pipeline = Pipeline(cfg)
    pipeline.train()
    pipeline.evaluate()
    snapshot = pipeline.paths.output_dir / ".hydra"
    snapshot.mkdir(exist_ok=True)
    OmegaConf.save(cfg, snapshot / "config.yaml")


@pytest.fixture(scope="session")
def trained_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One fake repository with one trained tiny run, shared by the session."""
    root = tmp_path_factory.mktemp("repo")
    processed = root / "data" / "processed" / DATASET
    processed.mkdir(parents=True)
    write_synthetic_tensors(processed / "tensors.npz")
    write_raw_frames(root / "data" / "raw" / DATASET, 12)
    train_tiny_run(root, RUN_ID)
    return root


@pytest.fixture
def repo_root(trained_root: Path, tmp_path: Path) -> Path:
    """A per-test copy of the trained repository, so a test may mutate it."""
    root = tmp_path / "repo"
    shutil.copytree(trained_root, root)
    return root


@pytest.fixture
def client(repo_root: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(repo_root=repo_root)
    return TestClient(app)
