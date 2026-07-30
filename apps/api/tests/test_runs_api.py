"""Run endpoints: listing, detail, and path-traversal safety."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
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


def test_delete_run_removes_the_run_and_its_assets(client, repo_root: Path):
    """DELETE removes the run's whole output directory (checkpoint, figures, video, …)
    and it disappears from the listing and detail."""
    run_dir = repo_root / "outputs" / "demo_run"
    assert run_dir.is_dir() and (run_dir / "checkpoints" / "ckpt.pt").exists()

    assert client.delete("/api/runs/demo_run").status_code == 204

    assert not run_dir.exists(), "the run's assets are gone from disk"
    assert client.get("/api/runs/demo_run").status_code == 404
    assert "demo_run" not in {run["id"] for run in client.get("/api/runs").json()}


def test_delete_unknown_run_is_404(client):
    assert client.delete("/api/runs/no-such-run").status_code == 404


def test_delete_active_run_is_rejected(client, repo_root: Path):
    """A run that is currently training cannot be deleted out from under the worker."""
    from naviernet_api.services import run_manager

    run_manager._jobs["demo_run"] = run_manager._RunJob(dataset="highest_t")

    response = client.delete("/api/runs/demo_run")
    assert response.status_code == 409
    assert "training" in response.json()["detail"]
    assert (repo_root / "outputs" / "demo_run").is_dir(), "the live run is left intact"


def test_delete_run_rejects_a_bad_id(client):
    """An id that is not a valid run directory (a forbidden character here) reaches the
    handler and is a 404 -- never a delete outside outputs/. The full path-traversal
    guard (the same `_safe_run_dir` helper `get_run` uses) is covered at the service
    level by test_path_traversal_id_is_rejected."""
    assert client.delete("/api/runs/a~b").status_code == 404


def test_delete_if_idle_blocks_a_live_run_and_only_deletes_an_idle_one():
    """The lock-held guard behind the endpoint: a queued/running run is never handed to
    the delete callback (so a concurrent launch can't have its dir removed mid-run); an
    idle run is, and its existence maps to deleted/missing."""
    from naviernet_api.services import run_manager

    fired: list[str] = []
    run_manager._jobs["live"] = run_manager._RunJob(dataset="d")  # state defaults "running"
    assert run_manager.delete_if_idle("live", lambda: fired.append("live") or True) == "active"
    assert fired == [], "a live run's delete callback never runs"

    assert run_manager.delete_if_idle("gone", lambda: True) == "deleted"
    assert run_manager.delete_if_idle("nope", lambda: False) == "missing"


def _write_project(repo_root: Path, pid: str, datasets: list[str]) -> None:
    projects = repo_root / "projects"
    projects.mkdir(exist_ok=True)
    (projects / f"{pid}.json").write_text(
        json.dumps(
            {
                "id": pid,
                "name": "Test project",
                "description": "",
                "datasets": datasets,
                "created_at": "2026-07-24T00:00:00+00:00",
            }
        )
    )


def test_list_runs_scoped_to_a_project(client, repo_root: Path):
    """`?project=` keeps only runs whose datasets belong to the project."""
    pid = "a" * 32
    _write_project(repo_root, pid, ["highest_t"])
    foreign = repo_root / "outputs" / "foreign_run"
    (foreign / "checkpoints").mkdir(parents=True)
    (foreign / "metrics.json").write_text(json.dumps({"dataset": "elsewhere"}))

    ids = [run["id"] for run in client.get("/api/runs", params={"project": pid}).json()]

    assert "demo_run" in ids
    assert "foreign_run" not in ids


def test_project_scope_includes_joint_runs_spanning_its_dataset(client, repo_root: Path):
    """A joint (v2 metrics) run counts as the project's if any spanned dataset is."""
    pid = "a" * 32
    _write_project(repo_root, pid, ["highest_t"])
    joint = repo_root / "outputs" / "joint_run"
    (joint / "checkpoints").mkdir(parents=True)
    (joint / "metrics.json").write_text(
        json.dumps({"datasets": ["highest_t", "elsewhere"], "per_dataset": {}})
    )

    ids = [run["id"] for run in client.get("/api/runs", params={"project": pid}).json()]

    assert "joint_run" in ids


def test_unknown_project_scope_is_404(client):
    assert client.get("/api/runs", params={"project": "f" * 32}).status_code == 404


def test_summary_carries_datasets_heldout_val_iou_and_date(client, repo_root: Path):
    """Joint (v2) runs surface their spanned/held-out datasets and axis-A IoU."""
    joint = repo_root / "outputs" / "joint_run"
    (joint / "checkpoints").mkdir(parents=True)
    (joint / "metrics.json").write_text(
        json.dumps(
            {
                "datasets": ["highest_t", "second_ds"],
                "training_datasets": ["highest_t"],
                "heldout_datasets": ["second_ds"],
                "val_iou_mean": 0.941,
                "per_dataset": {},
            }
        )
    )

    rows = {run["id"]: run for run in client.get("/api/runs").json()}

    joint_row = rows["joint_run"]
    assert joint_row["datasets"] == ["highest_t", "second_ds"]
    assert joint_row["heldout_datasets"] == ["second_ds"]
    assert joint_row["val_iou_mean"] == pytest.approx(0.941)
    assert joint_row["iou_holdout"] is None
    demo = rows["demo_run"]
    assert demo["datasets"] == ["highest_t"]
    assert demo["date"] is not None  # ISO timestamp from the run directory


def test_summary_reflects_live_job_state(client, repo_root: Path):
    """A run the server is training shows as running (with progress); a run
    whose job errored shows as failed — overriding the on-disk view."""
    from naviernet_api.services import run_manager

    live = repo_root / "outputs" / "live_run"
    live.mkdir(parents=True)
    dead = repo_root / "outputs" / "dead_run"
    (dead / "checkpoints").mkdir(parents=True)

    run_manager._jobs["live_run"] = run_manager._RunJob(
        dataset="highest_t", state="running", steps_done=40, steps_total=200
    )
    run_manager._jobs["dead_run"] = run_manager._RunJob(
        dataset="highest_t", state="error", message="loss diverged"
    )

    rows = {run["id"]: run for run in client.get("/api/runs").json()}

    assert rows["live_run"]["status"] == "running"
    assert rows["live_run"]["steps_done"] == 40
    assert rows["live_run"]["steps_total"] == 200
    assert rows["dead_run"]["status"] == "failed"
