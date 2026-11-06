"""The run roster, a run's detail and its artifacts."""

from __future__ import annotations

from fastapi.testclient import TestClient

from conftest import DATASET, RUN_ID


def test_the_roster_lists_the_trained_run_with_its_dataset(client: TestClient):
    body = client.get("/api/runs").json()

    assert [run["id"] for run in body] == [RUN_ID]
    assert body[0]["dataset"] == DATASET
    assert body[0]["has_config"] is True


def test_the_detail_names_the_frames_and_their_roles(client: TestClient):
    body = client.get(f"/api/runs/{RUN_ID}").json()

    frames = {f["frame"]: f for f in body["frames"]}
    assert frames[1]["role"] == "train" and frames[1]["t_ms"] == 0.0
    assert frames[11]["role"] == "truncated"
    assert frames[12]["role"] == "excluded" and frames[12]["t_ms"] is None
    assert body["scales"]["length_um"] == 300.0
    assert "u" in body["fields"]


def test_an_unknown_or_unsafe_run_is_404(client: TestClient):
    assert client.get("/api/runs/nope").status_code == 404
    # Encoded forms reach the handler; a bare ".." is normalised by the client.
    assert client.get("/api/runs/%2E%2E").status_code == 404
    assert client.get("/api/runs/bad%20id").status_code == 404
    assert client.get("/api/runs/..%2Fetc").status_code == 404


def test_artifacts_come_back_verbatim(client: TestClient):
    metrics = client.get(f"/api/runs/{RUN_ID}/artifacts/metrics").json()
    config = client.get(f"/api/runs/{RUN_ID}/artifacts/config").json()

    assert metrics["dataset"] == DATASET
    assert config["dataset"] == DATASET
    assert client.get(f"/api/runs/{RUN_ID}/artifacts/audit").status_code == 200
    assert client.get(f"/api/runs/{RUN_ID}/artifacts/bogus").status_code == 422
