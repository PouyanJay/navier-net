"""API test fixtures.

Tests are hermetic: they stage a fake repo root under `tmp_path` with one run's
artifacts and point the app at it via a dependency override, so they never depend
on the developer's real `outputs/` or a trained checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from naviernet_api.main import create_app
from naviernet_api.settings import Settings, get_settings


def write_synthetic_tensors(path: Path) -> None:
    """A tiny but schema-complete `tensors.npz`: an 11-frame growing bubble.

    Small enough that a real 2-step training run over it takes a second, so the
    solver tests stay in the fast, data-free tier.
    """
    import numpy as np

    n_t, height, width = 11, 12, 16
    alpha = np.zeros((n_t, height, width), dtype=np.float32)
    for i in range(n_t):
        alpha[i, 3:9, : 3 + i] = 1.0  # a bubble growing downstream, frame by frame
    sdf = ((0.5 - alpha) * 0.1).astype(np.float32)
    valid = np.ones_like(alpha)
    x_star = ((np.arange(width) + 0.5) / width).astype(np.float32)
    y_star = ((np.arange(height) + 0.5) / height).astype(np.float32)
    t_star = (np.arange(n_t) * 0.1).astype(np.float32)
    # The smooth interface curve the QC overlay draws: a box round each frame's
    # bubble, growing downstream. [x*, y*] per point, one closed loop per frame.
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
        "dataset": "highest_t",
        "um_per_px": 4.3,
        "L_ref_um": 300.0,
        "U_ref": 0.2,
        "t_ref_ms": 1.5,
        "x_pin_star": float(x_star[1]),
        "n_frames_usable": n_t,
        "n_frames_event": 10,
        # Raw-frame geometry, so the overlays figure can render on synthetic
        # raws: the imaged band's rows within the (16 x 140) raw frames.
        "y_roi": [64, 76],
        # Snapshot of the conditions baked into these tensors (matches the
        # highest_t defaults), so the staleness check has something to compare.
        "baked_conditions": {
            "dt_frame_ms": 0.5,
            "channel_width_um": 300.0,
            "U_ref": 0.2,
        },
        # Dimensionless groups, as preprocess records them, so these tensors can
        # join a conditioned multi-dataset (transfer-learning) run.
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


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """A minimal repo layout with one fully-populated run, `demo_run`."""
    run = tmp_path / "outputs" / "demo_run"
    (run / ".hydra").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    (run / "figures").mkdir()
    (run / "video").mkdir()

    (run / "metrics.json").write_text(
        json.dumps(
            {
                "run_name": "demo_run",
                "dataset": "highest_t",
                "iou_holdout": 0.968,
                "iou_mean": 0.962,
                "holdout_frame": 6,
                "iou_per_frame": {"1": 0.973, "6": 0.968},
                "nose_speed_mm_s": 177.0,
            }
        )
    )
    (run / ".hydra" / "config.yaml").write_text(
        "dataset: highest_t\nrun_name: demo_run\ntraining:\n  steps: 1500\n"
    )
    (run / "dimensionless_groups.json").write_text(
        json.dumps(
            {
                "groups": {
                    "Re": 215.5,
                    "We": 2.302,
                    "Ca": 0.01068,
                    "Pr": 9.411,
                    "hele_shaw": 0.2228,
                    "bretherton_film_um": 4.875,
                }
            }
        )
    )
    # The front-velocity report, in the shape the evaluate stage writes it: a
    # whole-front speed, the apex's two components, and a profile whose nose-cap
    # bin is suppressed (null) because the level-set estimate is untrustworthy
    # there.
    (run / "front_velocity.json").write_text(
        json.dumps(
            {
                "front_geometry": True,
                "nose_speed": {
                    "t_ms": [0.0, 0.5],
                    "v_um_per_ms": [120.5, 138.25],
                    "measured": {
                        "t_ms": [0.25],
                        "v_um_per_ms": [131.0],
                        "heldout": [True],
                    },
                },
                "apex": {
                    "t_ms": [0.0, 0.5],
                    "x_um": [180.0, 210.0],
                    "y_um": [64.0, 64.2],
                    "vx_um_per_ms": [118.0, 137.0],
                    "vy_um_per_ms": [0.4, -0.2],
                    "measured": {
                        "t_ms": [0.25],
                        "vx_um_per_ms": [130.0],
                        "vy_um_per_ms": [0.1],
                        "heldout": [True],
                    },
                },
                "profile": {
                    "s": [0.25, 0.75],
                    "segments": [
                        {
                            "name": "upper_body",
                            "bin_start": 0,
                            "bin_end": 1,
                            "s_start": 0.0,
                            "s_end": 0.5,
                            "measured": True,
                        },
                        {
                            "name": "nose_cap",
                            "bin_start": 1,
                            "bin_end": 2,
                            "s_start": 0.5,
                            "s_end": 1.0,
                            "measured": False,
                        },
                    ],
                    "times": [
                        {
                            "t_ms": 0.0,
                            "frames": [1, 2],
                            "heldout": False,
                            "model": [3.2, 120.0],
                            "measured": [3.0, None],
                        }
                    ],
                    "kymograph": {
                        "t_ms": [0.0, 0.5],
                        "v_um_per_ms": [[3.2, 120.0], [3.6, 131.0]],
                    },
                },
            }
        )
    )
    (run / "figures" / "trajectories.png").write_bytes(b"\x89PNG\r\n")
    (run / "video" / "growth.mp4").write_bytes(b"\x00")

    # A real (tiny) first-party checkpoint so step-count and loss-history read.
    import torch

    torch.save(
        {
            "model": {},
            "opt": {},
            "state": {
                "done": 1500,
                "hist": [{"step": 200, "data": 5e-3, "vof": 4e-2, "div": 4e-3, "bc": 2e-3}],
                "w": {},
            },
        },
        run / "checkpoints" / "ckpt.pt",
    )

    # Preprocessed tensors are dataset-scoped (data/processed/<dataset>/).
    # They are real (tiny) tensors so a launched run can actually train on them.
    tensors = tmp_path / "data" / "processed" / "highest_t"
    tensors.mkdir(parents=True)
    write_synthetic_tensors(tensors / "tensors.npz")

    # An "empty" run: a directory with no checkpoint yet.
    (tmp_path / "outputs" / "scratch").mkdir(parents=True)

    # A raw dataset with a few real (tiny) TIFF frames.
    raw = tmp_path / "data" / "raw" / "sample"
    raw.mkdir(parents=True)
    from PIL import Image

    for i in (1, 2, 3):
        Image.new("L", (64, 48), color=20 * i).save(raw / f"{i}.tif", format="TIFF")

    return tmp_path


@pytest.fixture(autouse=True)
def _clear_job_registries():
    """The job registries are process-global; isolate them per test."""
    from naviernet_api.services import jobs, run_manager, sweep_manager

    def clear() -> None:
        jobs._jobs.clear()
        run_manager._jobs.clear()
        sweep_manager._sweeps.clear()

    clear()
    yield
    clear()


@pytest.fixture
def client(repo_root: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(repo_root=repo_root)
    return TestClient(app)


@pytest.fixture
def sample_processed(repo_root: Path) -> Path:
    """Make the `sample` series both raw-present and preprocessed, so staleness
    (which compares tensor meta to the composed config) has tensors to read."""
    processed = repo_root / "data" / "processed" / "sample"
    processed.mkdir(parents=True, exist_ok=True)
    write_synthetic_tensors(processed / "tensors.npz")
    return repo_root


@pytest.fixture
def tiff_bytes() -> bytes:
    """A minimal valid TIFF image as bytes (for upload tests)."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("L", (32, 24)).save(buffer, format="TIFF")
    return buffer.getvalue()
