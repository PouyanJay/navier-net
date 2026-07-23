"""Read-only artifact endpoints: groups, validation, loss history, figures, video."""

from __future__ import annotations

import pytest
from naviernet_api.services import runs as runs_service
from naviernet_api.settings import Settings


def test_groups_returns_dimensionless_groups(client):
    r = client.get("/api/runs/demo_run/groups")
    assert r.status_code == 200
    groups = r.json()
    assert groups["Re"] == 215.5
    assert groups["bretherton_film_um"] == 4.875


def test_validation_composes_metrics_and_groups(client):
    r = client.get("/api/runs/demo_run/validation")
    assert r.status_code == 200
    v = r.json()

    assert v["nose_speed_inferred_mm_s"] == 177.0
    assert v["nose_speed_measured_mm_s"] == 180.0  # documented for highest_t
    assert v["nose_speed_error_pct"] == pytest.approx(100 * 3 / 180, rel=1e-3)
    assert v["bretherton_film_um"] == 4.875
    assert v["reynolds"] == 215.5
    assert v["iou_holdout"] == 0.968
    assert v["holdout_frame"] == 6


def test_loss_history_returns_records(client):
    r = client.get("/api/runs/demo_run/loss-history")
    assert r.status_code == 200
    history = r.json()
    assert history[0]["step"] == 200
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


def test_figure_path_traversal_is_rejected(repo_root):
    """A crafted figure name must not escape the run's figures dir."""
    settings = Settings(repo_root=repo_root)
    for evil in ("../../metrics.json", "..%2f..", "sub/dir.png", "no-extension"):
        assert runs_service.figure_path(settings, "demo_run", evil) is None


def test_checkpoint_and_tensors_download(client):
    ckpt = client.get("/api/runs/demo_run/checkpoint")
    assert ckpt.status_code == 200
    assert ckpt.headers["content-type"] == "application/octet-stream"

    tensors = client.get("/api/runs/demo_run/tensors")
    assert tensors.status_code == 200


def test_artifacts_of_unknown_run_are_404(client):
    assert client.get("/api/runs/ghost/groups").status_code == 404
    assert client.get("/api/runs/ghost/validation").status_code == 404
    assert client.get("/api/runs/ghost/video").status_code == 404
    assert client.get("/api/runs/ghost/checkpoint").status_code == 404
