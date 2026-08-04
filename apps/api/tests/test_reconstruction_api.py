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


def test_front_velocity_report_is_written_and_served(client: TestClient, trained_run: str):
    """Evaluate persists the front's motion too, and the endpoint serves it.

    The synthetic bubble grows, so its nose speed is positive somewhere -- a
    report that came back all zeros would mean the derivative never reached the
    trajectory it is supposed to be the slope of.
    """
    report = client.get(f"/api/runs/{trained_run}/front-velocity").json()

    nose = report["nose_speed"]
    assert len(nose["t_ms"]) == len(nose["v_um_per_ms"]) > 10
    assert any(v is not None and v > 0 for v in nose["v_um_per_ms"])


def test_front_velocity_missing_for_untrained_runs(client: TestClient):
    assert client.get("/api/runs/scratch/front-velocity").status_code == 404
    assert client.get("/api/runs/no-such/front-velocity").status_code == 404


def test_front_velocity_rejects_a_traversing_dataset_name(client: TestClient, trained_run: str):
    """The dataset name becomes part of a path (SECURITY.md §3)."""
    response = client.get(
        f"/api/runs/{trained_run}/front-velocity", params={"dataset": "../evil"}
    )
    assert response.status_code == 404


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


def test_interface_frames_carry_the_front_s_own_velocity(client: TestClient, trained_run: str):
    """Each reconstructed instant carries arrows for the viewport overlay:
    position in µm, the outward unit normal, and the normal speed along it.

    The speed is the NORMAL component only -- the tangential one is unobservable
    from masks -- so the arrow is `v * n` and nothing here implies otherwise.
    """
    import math

    payload = client.get(f"/api/runs/{trained_run}/interface?frames=8").json()
    arrows = payload["frames"][0]["front"]

    assert arrows, "the default recipe has an explicit front, so it has arrows"
    assert all(len(arrow) == 5 for arrow in arrows)
    # Sparse enough to read as an annotation rather than a band of ink.
    assert 8 <= len(arrows) < 80
    for x_um, y_um, nx, ny, _v in arrows:
        assert payload["domain"]["x_um"][0] - 1 <= x_um
        assert math.hypot(nx, ny) == pytest.approx(1.0, abs=1e-3), (
            "the normal must be a unit vector; the viewport scales it by the speed"
        )
        assert y_um == y_um  # not NaN — JSON has no token for it


def test_interface_missing_for_untrained_runs(client: TestClient):
    assert client.get("/api/runs/scratch/interface").status_code == 404
    assert client.get("/api/runs/scratch/trajectory").status_code == 404
    assert client.get("/api/runs/no-such/interface").status_code == 404


def test_the_reconstruction_reaches_the_last_camera_frame(client: TestClient, trained_run: str):
    """Every consumer picks the reconstructed frame NEAREST in time to the camera
    instant it is drawing. Stopping the reconstruction one frame short therefore
    left the last two camera frames with the same nearest neighbour -- so they
    rendered an identical PINN contour, whatever the model actually predicted.

    The trajectory legitimately stops early (the final frame is FOV-truncated, so
    its MEASURED nose is unreliable), but the model's own prediction there is
    perfectly well defined.
    """
    payload = client.get(f"/api/runs/{trained_run}/interface?frames=8").json()
    last_predicted = payload["frames"][-1]["t_ms"]
    last_measured = payload["measured"][-1]["t_ms"]
    assert last_predicted == pytest.approx(last_measured, rel=1e-6), (
        f"reconstruction ends at {last_predicted} ms but the camera runs to "
        f"{last_measured} ms — the final frames would share a contour"
    )

    # And the two final camera instants must resolve to DIFFERENT contours.
    def nearest(t_ms):
        return min(payload["frames"], key=lambda f: abs(f["t_ms"] - t_ms))

    a, b = payload["measured"][-2]["t_ms"], payload["measured"][-1]["t_ms"]
    assert nearest(a)["t_ms"] != nearest(b)["t_ms"], (
        "the last two camera frames still map to the same reconstructed frame"
    )
