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
    assert project["dataset"] is None
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
    first = [p for p in client.get("/api/projects").json() if p["dataset"] == "sample"]
    assert len(first) == 1
    assert first[0]["name"] == "sample"

    # Listing again must reuse the materialized file, not mint a new identity.
    second = [p for p in client.get("/api/projects").json() if p["dataset"] == "sample"]
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
    r = client.patch(f"/api/projects/{project['id']}", json={"dataset": "sample"})
    assert r.status_code == 200
    assert r.json()["dataset"] == "sample"


def test_patch_with_explicit_null_detaches_the_dataset(client):
    project = client.post("/api/projects", json={"name": "attach me"}).json()
    client.patch(f"/api/projects/{project['id']}", json={"dataset": "sample"})

    r = client.patch(f"/api/projects/{project['id']}", json={"dataset": None})
    assert r.status_code == 200
    assert r.json()["dataset"] is None
    # An omitted field must stay untouched — the name survived both patches.
    assert r.json()["name"] == "attach me"


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
    r = client.patch(f"/api/projects/{project['id']}", json={"dataset": "nope"})
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
    assert any(p["dataset"] == "sample" for p in r.json())


def test_materialized_project_survives_edit(client, repo_root: Path):
    materialized = [p for p in client.get("/api/projects").json() if p["dataset"] == "sample"]
    project_id = materialized[0]["id"]
    r = client.patch(f"/api/projects/{project_id}", json={"name": "FC-72 bubble growth"})
    assert r.status_code == 200

    on_disk = json.loads((repo_root / "projects" / f"{project_id}.json").read_text())
    assert on_disk["name"] == "FC-72 bubble growth"
    assert on_disk["dataset"] == "sample"


def test_list_is_sorted_oldest_first(repo_root: Path):
    settings = Settings(repo_root=repo_root)
    a = projects_service.create_project(settings, "first")
    b = projects_service.create_project(settings, "second")
    ids = [p.id for p in projects_service.list_projects(settings)]
    assert ids.index(a.id) < ids.index(b.id)
