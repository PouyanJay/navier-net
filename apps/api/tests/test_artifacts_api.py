"""Read-only artifact endpoints: groups, validation, loss history, figures, video."""

from __future__ import annotations

import json

import pytest
from naviernet_api.services import runs as runs_service
from naviernet_api.settings import Settings


def test_groups_returns_dimensionless_groups(client):
    r = client.get("/api/runs/demo_run/groups")
    assert r.status_code == 200
    groups = r.json()
    assert groups["Re"] == pytest.approx(215.5)
    assert groups["bretherton_film_um"] == pytest.approx(4.875)


def test_validation_composes_metrics_and_groups(client):
    r = client.get("/api/runs/demo_run/validation")
    assert r.status_code == 200
    v = r.json()

    assert v["nose_speed_inferred_mm_s"] == pytest.approx(177.0)
    assert v["nose_speed_measured_mm_s"] == pytest.approx(180.0)  # documented
    assert v["nose_speed_error_pct"] == pytest.approx(100 * 3 / 180, rel=1e-3)
    assert v["bretherton_film_um"] == pytest.approx(4.875)
    assert v["reynolds"] == pytest.approx(215.5)
    assert v["iou_holdout"] == pytest.approx(0.968)
    assert v["holdout_frame"] == 6  # int, exact


def test_loss_history_returns_records(client):
    r = client.get("/api/runs/demo_run/loss-history")
    assert r.status_code == 200
    history = r.json()
    assert history[0]["step"] == 200  # int, exact
    assert history[0]["vof"] == pytest.approx(0.04)


def test_figure_is_served_as_png(client):
    r = client.get("/api/runs/demo_run/figures/trajectories.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"


def test_missing_figure_is_404(client):
    assert client.get("/api/runs/demo_run/figures/nope.png").status_code == 404


def test_video_is_served(client):
    r = client.get("/api/runs/demo_run/video")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"


@pytest.mark.parametrize(
    "evil", ["../../metrics.json", "..%2f..", "sub/dir.png", "no-extension"]
)
def test_figure_path_traversal_is_rejected(repo_root, evil):
    """A crafted figure name must not escape the run's figures dir."""
    settings = Settings(repo_root=repo_root)
    assert runs_service.figure_path(settings, "demo_run", evil) is None


def test_unsafe_dataset_name_cannot_escape_processed_root(repo_root):
    """A malicious `dataset` field must not steer tensors_path outside data/processed."""
    settings = Settings(repo_root=repo_root)
    evil = repo_root / "outputs" / "evil"
    evil.mkdir(parents=True)
    (evil / "metrics.json").write_text(json.dumps({"dataset": "../../../../etc"}))

    # The unsafe name is rejected, so it is never surfaced or used in a path.
    dataset, _ = runs_service.read_dataset_and_metrics(settings, "evil")
    assert dataset is None
    assert runs_service.tensors_path(settings, "evil") is None


def test_checkpoint_download(client):
    r = client.get("/api/runs/demo_run/checkpoint")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"


def test_tensors_download(client):
    r = client.get("/api/runs/demo_run/tensors")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"


@pytest.mark.parametrize(
    "endpoint",
    ["groups", "validation", "video", "checkpoint", "tensors", "loss-history", "figures/x.png"],
)
def test_artifacts_of_unknown_run_are_404(client, endpoint):
    assert client.get(f"/api/runs/ghost/{endpoint}").status_code == 404
