"""The run manifest: a run's project, series, and launch time as stored facts.

Before `outputs/<run_id>/run.json` existed, a run's project was joined through
its dataset name every time the list was read, so membership was retroactive and
a run whose dataset could not be identified belonged nowhere. These tests pin the
stored-fact behavior and the fallback that keeps pre-manifest runs visible.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from helpers import TINY_RUN, read_stream
from naviernet_api.services import projects as projects_service
from naviernet_api.settings import Settings

PROJECT_A = "a" * 32
PROJECT_B = "b" * 32


def _write_project(repo_root: Path, project_id: str, datasets: list[str]) -> None:
    projects = repo_root / "projects"
    projects.mkdir(exist_ok=True)
    (projects / f"{project_id}.json").write_text(
        json.dumps(
            {
                "id": project_id,
                "name": f"Project {project_id[0]}",
                "description": "",
                "datasets": datasets,
                "created_at": "2026-08-01T00:00:00.000000+00:00",
            }
        )
    )


def _write_run(
    repo_root: Path,
    run_id: str,
    *,
    manifest: dict | None = None,
    metrics: dict | None = None,
) -> Path:
    """A run directory as it would exist on disk, with the parts under test."""
    run_dir = repo_root / "outputs" / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    if manifest is not None:
        (run_dir / "run.json").write_text(json.dumps(manifest))
    if metrics is not None:
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
    return run_dir


def _manifest(project_id: str | None, datasets: list[str], created_at: str) -> dict:
    return {
        "schema_version": 1,
        "run_id": "unused",
        "project_id": project_id,
        "datasets": datasets,
        "heldout_datasets": [],
        "created_at": created_at,
    }


def test_launch_records_the_project_and_series_it_was_started_from(
    client: TestClient, repo_root: Path
):
    """A launched run writes its identity before it trains, and reports it back."""
    _write_project(repo_root, PROJECT_A, ["highest_t"])

    run_id = client.post("/api/runs", json={**TINY_RUN, "project": PROJECT_A}).json()["run_id"]
    read_stream(client, run_id)  # drain = wait for the background thread

    manifest = json.loads((repo_root / "outputs" / run_id / "run.json").read_text())
    assert manifest["project_id"] == PROJECT_A
    assert manifest["datasets"] == ["highest_t"]
    assert manifest["run_id"] == run_id
    assert manifest["created_at"].endswith("+00:00")
    assert client.get(f"/api/runs/{run_id}").json()["project_id"] == PROJECT_A


def test_launch_infers_the_project_from_the_series_when_none_is_sent(
    client: TestClient, repo_root: Path
):
    """One project holds the series, so the run's owner is not in doubt."""
    _write_project(repo_root, PROJECT_A, ["highest_t"])

    run_id = client.post("/api/runs", json=TINY_RUN).json()["run_id"]
    read_stream(client, run_id)

    manifest = json.loads((repo_root / "outputs" / run_id / "run.json").read_text())
    assert manifest["project_id"] == PROJECT_A


def test_launch_with_an_unknown_project_is_rejected(client: TestClient):
    """Better a 404 the caller can act on than a run that records no owner."""
    response = client.post("/api/runs", json={**TINY_RUN, "project": "f" * 32})

    assert response.status_code == 404
    assert "unknown project" in response.json()["detail"]


def test_a_sweep_child_records_the_project_too(client: TestClient, repo_root: Path):
    """Children bypass the run launcher, so they need the owner resolved for them
    -- otherwise a swept comparison would belong to no project at all."""
    _write_project(repo_root, PROJECT_A, ["highest_t"])

    sweep = client.post(
        "/api/sweeps", json={**TINY_RUN, "seeds": [0], "project": PROJECT_A}
    ).json()
    child_id = sweep["children"][0]["run_id"]
    read_stream(client, child_id)

    manifest = json.loads((repo_root / "outputs" / child_id / "run.json").read_text())
    assert manifest["project_id"] == PROJECT_A


def test_resume_keeps_the_manifest_the_run_was_launched_with(
    client: TestClient, repo_root: Path
):
    """A resume continues a run; it does not re-launch it, so its identity holds."""
    _write_project(repo_root, PROJECT_A, ["highest_t"])
    run_id = client.post("/api/runs", json={**TINY_RUN, "project": PROJECT_A}).json()["run_id"]
    read_stream(client, run_id)
    launched = json.loads((repo_root / "outputs" / run_id / "run.json").read_text())

    client.post(
        "/api/runs", json={"resume": True, "run_id": run_id, "steps": 2, "render": False}
    )
    read_stream(client, run_id)

    assert json.loads((repo_root / "outputs" / run_id / "run.json").read_text()) == launched


def test_a_recorded_owner_survives_the_series_being_shared(client: TestClient, repo_root: Path):
    """The fix: attaching the series to a second project must not hand that
    project the first project's run history."""
    _write_project(repo_root, PROJECT_A, ["highest_t"])
    _write_project(repo_root, PROJECT_B, ["highest_t"])
    _write_run(
        repo_root,
        "owned_run",
        manifest=_manifest(PROJECT_A, ["highest_t"], "2026-08-02T10:00:00+00:00"),
        metrics={"dataset": "highest_t"},
    )

    in_a = {run["id"] for run in client.get("/api/runs", params={"project": PROJECT_A}).json()}
    in_b = {run["id"] for run in client.get("/api/runs", params={"project": PROJECT_B}).json()}

    assert "owned_run" in in_a
    assert "owned_run" not in in_b


def test_a_pre_manifest_run_is_still_scoped_by_its_series(client: TestClient, repo_root: Path):
    """`demo_run` predates the manifest; the dataset join is all it has."""
    _write_project(repo_root, PROJECT_A, ["highest_t"])

    listed = {
        run["id"]: run for run in client.get("/api/runs", params={"project": PROJECT_A}).json()
    }

    assert "demo_run" in listed
    assert listed["demo_run"]["project_id"] is None


def test_the_manifest_names_the_series_of_a_run_that_never_evaluated(
    client: TestClient, repo_root: Path
):
    """A run that died before `metrics.json` was written still knows what it was
    training on, and reports the moment it was launched rather than a mtime."""
    _write_project(repo_root, PROJECT_A, ["highest_t"])
    _write_run(
        repo_root,
        "died_early",
        manifest=_manifest(PROJECT_A, ["highest_t"], "2026-08-02T10:00:00+00:00"),
    )

    listed = {
        run["id"]: run for run in client.get("/api/runs", params={"project": PROJECT_A}).json()
    }

    assert listed["died_early"]["datasets"] == ["highest_t"]
    assert listed["died_early"]["date"] == "2026-08-02T10:00:00+00:00"


def test_an_unusable_manifest_falls_back_to_the_dataset_join(
    client: TestClient, repo_root: Path
):
    """Corrupt, or written by a newer server: never a run that vanishes from
    every listing because one file could not be parsed."""
    _write_project(repo_root, PROJECT_A, ["highest_t"])
    _write_run(repo_root, "corrupt_run", metrics={"dataset": "highest_t"})
    (repo_root / "outputs" / "corrupt_run" / "run.json").write_text("{not json")
    future = _manifest(PROJECT_B, ["highest_t"], "2026-08-02T10:00:00+00:00")
    _write_run(
        repo_root,
        "future_run",
        manifest={**future, "schema_version": 99},
        metrics={"dataset": "highest_t"},
    )

    ids = {run["id"] for run in client.get("/api/runs", params={"project": PROJECT_A}).json()}

    assert {"corrupt_run", "future_run"} <= ids


def test_deleting_a_project_takes_its_own_runs_and_leaves_anothers(repo_root: Path):
    """Runs follow their recorded project, not their series: a shared series'
    runs are split between the projects that launched them."""
    _write_project(repo_root, PROJECT_A, ["highest_t"])
    _write_project(repo_root, PROJECT_B, ["highest_t"])
    mine = _write_run(
        repo_root,
        "run_of_a",
        manifest=_manifest(PROJECT_A, ["highest_t"], "2026-08-02T10:00:00+00:00"),
        metrics={"dataset": "highest_t"},
    )
    theirs = _write_run(
        repo_root,
        "run_of_b",
        manifest=_manifest(PROJECT_B, ["highest_t"], "2026-08-02T11:00:00+00:00"),
        metrics={"dataset": "highest_t"},
    )

    projects_service.delete_project(Settings(repo_root=repo_root), PROJECT_A)

    assert not mine.exists()
    assert theirs.is_dir()


def test_the_manifest_digests_the_config_snapshot_it_was_launched_with(
    client: TestClient, repo_root: Path
):
    """A run is matched to its exact inputs: the manifest's digest is that of
    the snapshot beside it, without the run's own name."""
    run_id = client.post("/api/runs", json=TINY_RUN).json()["run_id"]
    read_stream(client, run_id)

    from naviernet_api.services.run_manager import config_digest
    from omegaconf import OmegaConf

    run_dir = repo_root / "outputs" / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    snapshot = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    assert manifest["schema_version"] == 2
    assert manifest["config_sha256"] == config_digest(snapshot)
    # The digest is of the inputs, not of the run's name: the same snapshot
    # under another run name digests the same, and a changed input does not.
    assert (
        config_digest(OmegaConf.merge(snapshot, {"run_name": "another"}))
        == (manifest["config_sha256"])
    )
    assert (
        config_digest(OmegaConf.merge(snapshot, {"training": {"seed": 7}}))
        != (manifest["config_sha256"])
    )
