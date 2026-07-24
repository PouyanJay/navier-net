"""Single-port deployment: the API serves the built web app when it exists."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from naviernet_api.main import create_app
from naviernet_api.settings import Settings, get_settings


@pytest.fixture
def deployed_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """An app created against a repo root that has a web build."""
    dist = tmp_path / "apps" / "web" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>NavierNet</title>")
    (dist / "assets" / "app.js").write_text("console.log('naviernet')")

    # The SPA mount decision happens at app creation, which reads the cached
    # settings — point them at the staged repo for this app instance.
    monkeypatch.setenv("NAVIERNET_ROOT", str(tmp_path))
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(repo_root=tmp_path)
    yield TestClient(app)
    get_settings.cache_clear()


def test_serves_the_web_app_and_its_assets(deployed_client: TestClient):
    assert "NavierNet" in deployed_client.get("/").text
    assert "naviernet" in deployed_client.get("/assets/app.js").text
    # Client-side routes fall back to the SPA shell.
    assert "NavierNet" in deployed_client.get("/solver").text


def test_api_paths_never_leak_the_spa_shell(deployed_client: TestClient):
    response = deployed_client.get("/api/no-such-route")
    assert response.status_code == 404
    assert "NavierNet" not in response.text
    # Real API routes still win over the catch-all.
    assert deployed_client.get("/api/runs").status_code == 200
    assert deployed_client.get("/healthz").json() == {"status": "ok"}


def test_traversal_out_of_dist_falls_back_to_the_shell(deployed_client: TestClient):
    response = deployed_client.get("/..%2f..%2fpyproject.toml")
    assert "NavierNet" in response.text  # never a file outside dist
