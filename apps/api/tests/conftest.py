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
    meta = {
        "dataset": "highest_t",
        "um_per_px": 4.3,
        "L_ref_um": 300.0,
        "U_ref": 0.2,
        "t_ref_ms": 1.5,
        "x_pin_star": float(x_star[1]),
        "n_frames_usable": n_t,
        "n_frames_event": 10,
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
    from naviernet_api.services import jobs, run_manager

    jobs._jobs.clear()
    run_manager._jobs.clear()
    yield
    jobs._jobs.clear()
    run_manager._jobs.clear()


@pytest.fixture
def client(repo_root: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(repo_root=repo_root)
    return TestClient(app)


@pytest.fixture
def tiff_bytes() -> bytes:
    """A minimal valid TIFF image as bytes (for upload tests)."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("L", (32, 24)).save(buffer, format="TIFF")
    return buffer.getvalue()
