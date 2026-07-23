"""Run endpoints: listing, detail, and path-traversal safety."""

from __future__ import annotations

from pathlib import Path

from naviernet_api.services import runs as runs_service
from naviernet_api.settings import Settings


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_runs_includes_the_trained_run(client):
    r = client.get("/api/runs")
    assert r.status_code == 200
    runs = {run["id"]: run for run in r.json()}

    assert "demo_run" in runs
    demo = runs["demo_run"]
    assert demo["dataset"] == "highest_t"
    assert demo["status"] == "trained"
    assert demo["iou_holdout"] == 0.968


def test_list_marks_a_checkpointless_dir_as_empty(client):
    runs = {run["id"]: run for run in client.get("/api/runs").json()}
    assert runs["scratch"]["status"] == "empty"


def test_run_detail_reads_metrics_config_and_artifacts(client):
    r = client.get("/api/runs/demo_run")
    assert r.status_code == 200
    detail = r.json()

    assert detail["steps"] == 1500
    assert detail["metrics"]["nose_speed_mm_s"] == 177.0
    assert detail["config"]["training"]["steps"] == 1500
    assert detail["artifacts"]["checkpoint"] is True
    assert detail["artifacts"]["video"] is True
    assert "trajectories.png" in detail["artifacts"]["figures"]


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/does-not-exist").status_code == 404


def test_path_traversal_id_is_rejected(repo_root: Path, tmp_path: Path):
    """A run id escaping outputs/ must resolve to nothing (SECURITY.md §3)."""
    settings = Settings(repo_root=repo_root)
    # Plant a secret one level above outputs/ and try to reach it.
    (tmp_path / "secret").mkdir(exist_ok=True)
    for evil in ("../secret", "..", "../../etc", "foo/bar", "with space"):
        assert runs_service.get_run(settings, evil) is None
