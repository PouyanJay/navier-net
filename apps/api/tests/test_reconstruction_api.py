"""Trajectory + interface endpoints: the data behind the interactive Results.

Integration: a real (tiny) run is trained + evaluated first, then both
endpoints are read back and their geometry sanity-checked.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from helpers import TINY_RUN, final_status, read_stream


@pytest.fixture
def trained_run(client: TestClient) -> str:
    """A freshly trained + evaluated tiny run (2 steps on synthetic tensors)."""
    run_id = client.post("/api/runs", json=TINY_RUN).json()["run_id"]
    events = read_stream(client, run_id)
    final = final_status(events)
    assert final["state"] == "done", f"run failed: {final.get('message')}"
    return run_id


def test_trajectory_is_written_and_served(client: TestClient, trained_run: str):
    """Evaluate persists growth kinematics; the endpoint serves them as data."""
    trajectory = client.get(f"/api/runs/{trained_run}/trajectory").json()
    assert len(trajectory["t_ms"]) == len(trajectory["nose_um"]) > 10
    assert len(trajectory["measured"]["t_ms"]) == len(trajectory["measured"]["nose_um"]) == 10
    # The measured nose advances downstream in the synthetic growing bubble.
    measured_nose = trajectory["measured"]["nose_um"]
    assert measured_nose[-1] > measured_nose[0]


def test_interface_frames_serve_contours(client: TestClient, trained_run: str):
    """The viewport endpoint returns per-timestep interface polylines."""
    payload = client.get(f"/api/runs/{trained_run}/interface?frames=8").json()
    assert len(payload["frames"]) == 8
    assert payload["domain"]["x_um"][1] > payload["domain"]["x_um"][0]
    # Measured contours exist for every camera frame of the event.
    assert len(payload["measured"]) == 10
    measured_with_contours = [f for f in payload["measured"] if f["contours"]]
    assert measured_with_contours, "no measured interface contours extracted"
    first = measured_with_contours[0]["contours"][0]
    assert len(first) >= 8  # a polyline, not speckle
    assert all(len(point) == 2 for point in first)


def test_interface_missing_for_untrained_runs(client: TestClient):
    assert client.get("/api/runs/scratch/interface").status_code == 404
    assert client.get("/api/runs/scratch/trajectory").status_code == 404
    assert client.get("/api/runs/no-such/interface").status_code == 404
