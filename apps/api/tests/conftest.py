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
    tensors = tmp_path / "data" / "processed" / "highest_t"
    tensors.mkdir(parents=True)
    (tensors / "tensors.npz").write_bytes(b"PK\x03\x04")

    # An "empty" run: a directory with no checkpoint yet.
    (tmp_path / "outputs" / "scratch").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def client(repo_root: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(repo_root=repo_root)
    return TestClient(app)
