"""Project endpoints: list (with legacy materialization), create, and edit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from naviernet_api.services import projects as projects_service
from naviernet_api.settings import Settings


def test_create_project_returns_identity_and_persists(client):
    r = client.post(
        "/api/projects",
        json={"name": "Condensing slug", "description": "Inverse reconstruction."},
    )
    assert r.status_code == 201
    project = r.json()
    assert projects_service.is_valid_project_id(project["id"])
    assert project["name"] == "Condensing slug"
    assert project["description"] == "Inverse reconstruction."
    assert project["datasets"] == []
    assert project["created_at"]

    listed = {p["id"] for p in client.get("/api/projects").json()}
    assert project["id"] in listed


def test_create_project_rejects_blank_name(client):
    r = client.post("/api/projects", json={"name": "   "})
    assert r.status_code == 400
    assert "name" in r.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "x" * 200},
        {"name": "ok", "description": "y" * 3000},
    ],
    ids=["oversized-name", "oversized-description"],
)
def test_create_project_rejects_oversized_metadata(client, payload):
    r = client.post("/api/projects", json=payload)
    assert r.status_code == 400


def test_legacy_dataset_is_materialized_once(client, repo_root: Path):
    first = [p for p in client.get("/api/projects").json() if "sample" in p["datasets"]]
    assert len(first) == 1
    assert first[0]["name"] == "sample"

    # Listing again must reuse the materialized file, not mint a new identity.
    second = [p for p in client.get("/api/projects").json() if "sample" in p["datasets"]]
    assert second == first
    assert (repo_root / "projects" / f"{first[0]['id']}.json").is_file()


def test_patch_updates_name_and_description(client):
    project = client.post("/api/projects", json={"name": "before"}).json()
    r = client.patch(
        f"/api/projects/{project['id']}",
        json={"name": "after", "description": "now described"},
    )
    assert r.status_code == 200
    assert r.json()["name"] == "after"
    assert r.json()["description"] == "now described"

    # The file on disk is the source of truth — it must reflect the edit.
    fetched = [p for p in client.get("/api/projects").json() if p["id"] == project["id"]]
    assert fetched[0]["name"] == "after"


def test_patch_attaches_an_existing_dataset(client):
    project = client.post("/api/projects", json={"name": "attach me"}).json()
    r = client.patch(f"/api/projects/{project['id']}", json={"datasets": ["sample"]})
    assert r.status_code == 200
    assert r.json()["datasets"] == ["sample"]


def test_patch_with_explicit_null_detaches_the_dataset(client):
    project = client.post("/api/projects", json={"name": "attach me"}).json()
    client.patch(f"/api/projects/{project['id']}", json={"datasets": ["sample"]})

    r = client.patch(f"/api/projects/{project['id']}", json={"datasets": None})
    assert r.status_code == 200
    assert r.json()["datasets"] == []
    # An omitted field must stay untouched — the name survived both patches.
    assert r.json()["name"] == "attach me"


def test_legacy_single_dataset_file_is_migrated(client, repo_root: Path):
    # Files written before multi-series support carried `dataset: str|null`.
    projects_dir = repo_root / "projects"
    projects_dir.mkdir(exist_ok=True)
    legacy_id = "b" * 32
    (projects_dir / f"{legacy_id}.json").write_text(
        json.dumps(
            {
                "id": legacy_id,
                "name": "legacy",
                "description": "",
                "dataset": "sample",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    listed = {p["id"]: p for p in client.get("/api/projects").json()}
    assert listed[legacy_id]["datasets"] == ["sample"]


def test_patch_with_explicit_null_name_is_rejected(client):
    project = client.post("/api/projects", json={"name": "keep me"}).json()
    r = client.patch(f"/api/projects/{project['id']}", json={"name": None})
    assert r.status_code == 400
    assert "name" in r.json()["detail"]


def test_patch_with_explicit_null_description_clears_it(client):
    project = client.post(
        "/api/projects", json={"name": "p", "description": "to be cleared"}
    ).json()
    r = client.patch(f"/api/projects/{project['id']}", json={"description": None})
    assert r.status_code == 200
    assert r.json()["description"] == ""


def test_metadata_whitespace_is_normalized(client):
    r = client.post(
        "/api/projects", json={"name": "  padded  ", "description": "  also padded  "}
    )
    assert r.status_code == 201
    assert r.json()["name"] == "padded"
    assert r.json()["description"] == "also padded"


def test_get_project_by_id(client):
    project = client.post("/api/projects", json={"name": "fetch me"}).json()
    r = client.get(f"/api/projects/{project['id']}")
    assert r.status_code == 200
    assert r.json() == project
    assert client.get(f"/api/projects/{'f' * 32}").status_code == 404


def test_patch_rejects_a_missing_dataset(client):
    project = client.post("/api/projects", json={"name": "attach me"}).json()
    r = client.patch(f"/api/projects/{project['id']}", json={"datasets": ["nope"]})
    assert r.status_code == 400


def test_patch_unknown_or_malformed_id_is_404(client):
    assert client.patch(f"/api/projects/{'0' * 32}", json={"name": "x"}).status_code == 404
    # Not uuid4().hex-shaped: rejected before it can touch the filesystem.
    assert client.patch("/api/projects/not-a-uuid", json={"name": "x"}).status_code == 404


def test_corrupt_project_file_is_skipped(client, repo_root: Path):
    projects_dir = repo_root / "projects"
    projects_dir.mkdir(exist_ok=True)
    (projects_dir / f"{'a' * 32}.json").write_text("{not json")

    r = client.get("/api/projects")
    assert r.status_code == 200
    # The bad file is logged and dropped; the healthy projects still list.
    assert "a" * 32 not in {p["id"] for p in r.json()}
    assert any("sample" in p["datasets"] for p in r.json())


def test_materialized_project_survives_edit(client, repo_root: Path):
    materialized = [p for p in client.get("/api/projects").json() if "sample" in p["datasets"]]
    project_id = materialized[0]["id"]
    r = client.patch(f"/api/projects/{project_id}", json={"name": "FC-72 bubble growth"})
    assert r.status_code == 200

    on_disk = json.loads((repo_root / "projects" / f"{project_id}.json").read_text())
    assert on_disk["name"] == "FC-72 bubble growth"
    assert on_disk["datasets"] == ["sample"]


def test_list_is_sorted_oldest_first(repo_root: Path):
    settings = Settings(repo_root=repo_root)
    a = projects_service.create_project(settings, "first")
    b = projects_service.create_project(settings, "second")
    ids = [p.id for p in projects_service.list_projects(settings)]
    assert ids.index(a.id) < ids.index(b.id)


def _materialized_sample_id(client) -> str:
    """The id of the project auto-created for the `sample` raw dataset."""
    materialized = [p for p in client.get("/api/projects").json() if "sample" in p["datasets"]]
    assert len(materialized) == 1
    return materialized[0]["id"]


def _stage_run(repo_root: Path, run_name: str, dataset: str) -> Path:
    """A minimal run directory under `outputs/` tied to `dataset`."""
    run = repo_root / "outputs" / run_name
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"run_name": run_name, "dataset": dataset}))
    return run


def test_delete_project_removes_file_and_owned_data(client, sample_processed, repo_root: Path):
    project_id = _materialized_sample_id(client)
    run = _stage_run(repo_root, "sample_run", "sample")
    raw = repo_root / "data" / "raw" / "sample"
    processed = repo_root / "data" / "processed" / "sample"
    assert raw.is_dir() and processed.is_dir() and run.is_dir()

    r = client.delete(f"/api/projects/{project_id}")
    assert r.status_code == 200
    assert r.json()["id"] == project_id

    # The project file and everything it exclusively owned are gone from disk.
    assert not (repo_root / "projects" / f"{project_id}.json").exists()
    assert not raw.exists()
    assert not processed.exists()
    assert not run.exists()
    # The cascade is scoped: the fixture's unrelated `highest_t` run and tensors,
    # owned by no project in this test, must be untouched.
    assert (repo_root / "outputs" / "demo_run").is_dir()
    assert (repo_root / "data" / "processed" / "highest_t").is_dir()
    # A second delete finds nothing to remove.
    assert client.delete(f"/api/projects/{project_id}").status_code == 404


def test_delete_keeps_a_dataset_and_its_run_shared_with_another_project(
    client, repo_root: Path
):
    first = client.post("/api/projects", json={"name": "keeper"}).json()
    second = client.post("/api/projects", json={"name": "goner"}).json()
    for project in (first, second):
        client.patch(f"/api/projects/{project['id']}", json={"datasets": ["sample"]})
    shared_run = _stage_run(repo_root, "shared_run", "sample")

    assert client.delete(f"/api/projects/{second['id']}").status_code == 200

    # The shared series, its raw data, and its run survive for the project still
    # using it — deletion is scoped to what the deleted project alone owned.
    assert (repo_root / "data" / "raw" / "sample").is_dir()
    assert shared_run.is_dir()
    kept = client.get(f"/api/projects/{first['id']}").json()
    assert kept["datasets"] == ["sample"]


def test_delete_is_blocked_while_a_run_is_live_on_owned_data(client, repo_root: Path):
    from naviernet_api.services import run_manager

    project_id = _materialized_sample_id(client)
    run = _stage_run(repo_root, "sample_run", "sample")
    # A live training run holds the `sample` dataset (default state is "running").
    run_manager._jobs["live_sample"] = run_manager._RunJob(dataset="sample")

    r = client.delete(f"/api/projects/{project_id}")
    assert r.status_code == 409
    assert "sample" in r.json()["detail"]
    # Nothing was removed: the guard fires before any cascade side effect.
    assert (repo_root / "projects" / f"{project_id}.json").is_file()
    assert run.is_dir()
    assert (repo_root / "data" / "raw" / "sample").is_dir()


def test_remove_series_deletes_owned_data_and_updates_the_project(
    client, sample_processed, repo_root: Path
):
    project_id = _materialized_sample_id(client)
    run = _stage_run(repo_root, "sample_run", "sample")
    raw = repo_root / "data" / "raw" / "sample"
    processed = repo_root / "data" / "processed" / "sample"
    assert raw.is_dir() and processed.is_dir() and run.is_dir()

    r = client.delete(f"/api/projects/{project_id}/datasets/sample")
    assert r.status_code == 200
    assert r.json()["datasets"] == []

    # Owned by this project alone, so the series' runs and data cascade away —
    # but the project itself survives, unlike a whole-project delete.
    assert not raw.exists()
    assert not processed.exists()
    assert not run.exists()
    assert (repo_root / "projects" / f"{project_id}.json").is_file()
    # Removing it again finds no membership to remove.
    assert client.delete(f"/api/projects/{project_id}/datasets/sample").status_code == 404


def test_remove_series_keeps_data_shared_with_another_project(client, repo_root: Path):
    first = client.post("/api/projects", json={"name": "keeper"}).json()
    second = client.post("/api/projects", json={"name": "leaver"}).json()
    for project in (first, second):
        client.patch(f"/api/projects/{project['id']}", json={"datasets": ["sample"]})
    shared_run = _stage_run(repo_root, "shared_run", "sample")

    r = client.delete(f"/api/projects/{second['id']}/datasets/sample")
    assert r.status_code == 200
    assert r.json()["datasets"] == []

    # Membership went; the series, its data, and its run stay for the project
    # still using them.
    assert (repo_root / "data" / "raw" / "sample").is_dir()
    assert shared_run.is_dir()
    kept = client.get(f"/api/projects/{first['id']}").json()
    assert kept["datasets"] == ["sample"]


def test_remove_series_is_blocked_while_a_run_is_live_on_it(client, repo_root: Path):
    from naviernet_api.services import run_manager

    project_id = _materialized_sample_id(client)
    run = _stage_run(repo_root, "sample_run", "sample")
    run_manager._jobs["live_sample"] = run_manager._RunJob(dataset="sample")

    r = client.delete(f"/api/projects/{project_id}/datasets/sample")
    assert r.status_code == 409
    assert "sample" in r.json()["detail"]
    # Nothing was removed, membership included: the guard fires first.
    assert run.is_dir()
    assert (repo_root / "data" / "raw" / "sample").is_dir()
    assert client.get(f"/api/projects/{project_id}").json()["datasets"] == ["sample"]


def test_remove_series_unknown_membership_is_404(client):
    project_id = _materialized_sample_id(client)
    assert (
        client.delete(f"/api/projects/{project_id}/datasets/never_uploaded").status_code == 404
    )
    assert client.delete(f"/api/projects/{'0' * 32}/datasets/sample").status_code == 404


def test_delete_unknown_or_malformed_id_is_404(client):
    assert client.delete(f"/api/projects/{'0' * 32}").status_code == 404
    # Not uuid4().hex-shaped: rejected before it can touch the filesystem.
    assert client.delete("/api/projects/not-a-uuid").status_code == 404


def test_patch_validates_every_dataset_in_the_list(client, tiff_bytes):
    files = [("files", ("1.tif", tiff_bytes, "image/tiff"))]
    assert client.post("/api/datasets/second/upload", files=files).status_code == 200
    project = client.post("/api/projects", json={"name": "multi"}).json()

    # One bad entry anywhere in the list rejects the whole edit.
    r = client.patch(f"/api/projects/{project['id']}", json={"datasets": ["sample", "nope"]})
    assert r.status_code == 400

    # Two valid series persist together, in order.
    r = client.patch(f"/api/projects/{project['id']}", json={"datasets": ["sample", "second"]})
    assert r.status_code == 200
    assert r.json()["datasets"] == ["sample", "second"]
