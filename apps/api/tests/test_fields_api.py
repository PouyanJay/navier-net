"""Field-map endpoint: the model's predicted fields on a grid, as data.

Integration: a real (tiny) run is trained first; the endpoint then evaluates
that run's actual checkpoint — no mocks of the model anywhere.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from helpers import TINY_RUN, final_status, read_stream


@pytest.fixture
def trained_run(client: TestClient) -> str:
    run_id = client.post("/api/runs", json=TINY_RUN).json()["run_id"]
    events = read_stream(client, run_id)
    final = final_status(events)
    assert final["state"] == "done", f"run failed: {final.get('message')}"
    return run_id


def test_field_map_serves_velocity_on_a_grid(client: TestClient, trained_run: str):
    body = client.get(f"/api/runs/{trained_run}/field", params={"name": "u", "t": 0.1}).json()

    assert body["name"] == "u"
    assert body["unit"] == "mm·s⁻¹"
    assert len(body["values"]) == len(body["y_um"])
    assert len(body["values"][0]) == len(body["x_um"])
    assert body["vmin"] <= body["vmax"]
    # A Stage-A run offers the flow fields but not p/T.
    assert "u" in body["fields_available"]
    assert "p" not in body["fields_available"]


def test_field_map_alpha_is_a_volume_fraction(client: TestClient, trained_run: str):
    body = client.get(
        f"/api/runs/{trained_run}/field", params={"name": "alpha", "t": 0.0}
    ).json()

    flat = [v for row in body["values"] for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat)
    assert body["unit"] == "–"


def test_field_map_stage_b_field_on_stage_a_run_is_404(client: TestClient, trained_run: str):
    response = client.get(f"/api/runs/{trained_run}/field", params={"name": "T", "t": 0.1})
    assert response.status_code == 404
    assert "not in this model" in response.json()["detail"]


def test_field_map_rejects_unknown_names_and_missing_runs(client: TestClient, trained_run: str):
    assert (
        client.get(
            f"/api/runs/{trained_run}/field", params={"name": "vorticity", "t": 0}
        ).status_code
        == 422
    )
    assert client.get("/api/runs/nope/field", params={"name": "u", "t": 0}).status_code == 404


def test_field_map_scopes_a_joint_run_by_dataset(client: TestClient, repo_root):
    """A joint run's fields evaluate with the requested dataset's conditioning."""
    from conftest import write_synthetic_tensors

    processed = repo_root / "data" / "processed" / "second"
    processed.mkdir(parents=True)
    write_synthetic_tensors(processed / "tensors.npz")

    joint = {**TINY_RUN, "dataset": None, "datasets": ["highest_t", "second"]}
    run_id = client.post("/api/runs", json=joint).json()["run_id"]
    read_stream(client, run_id)

    body = client.get(
        f"/api/runs/{run_id}/field",
        params={"name": "alpha", "t": 0.1, "dataset": "second"},
    ).json()
    assert body["dataset"] == "second"
    assert len(body["values"]) == len(body["y_um"])


def test_field_map_unknown_joint_dataset_is_404_not_substituted(client: TestClient, repo_root):
    """Asking a joint run for a dataset it does not span must 404 — never
    another condition's values mislabeled under the requested name."""
    from conftest import write_synthetic_tensors

    processed = repo_root / "data" / "processed" / "second"
    processed.mkdir(parents=True)
    write_synthetic_tensors(processed / "tensors.npz")

    joint = {**TINY_RUN, "dataset": None, "datasets": ["highest_t", "second"]}
    run_id = client.post("/api/runs", json=joint).json()["run_id"]
    read_stream(client, run_id)

    response = client.get(
        f"/api/runs/{run_id}/field", params={"name": "alpha", "t": 0, "dataset": "nope"}
    )
    assert response.status_code == 404


def test_residual_maps_evaluate_via_autograd(client: TestClient, trained_run: str):
    """Stage-A residual maps (|r_vof|, |r_div|) serve on any trained run; the
    Stage-B ones state what is missing."""
    body = client.get(
        f"/api/runs/{trained_run}/field", params={"name": "res_vof", "t": 0.1}
    ).json()
    assert body["unit"] == "|r|"
    assert len(body["values"]) == len(body["y_um"])
    assert body["vmin"] >= 0  # magnitudes
    assert "res_div" in body["fields_available"]
    assert "res_mom" not in body["fields_available"]

    response = client.get(
        f"/api/runs/{trained_run}/field", params={"name": "res_mom", "t": 0.1}
    )
    assert response.status_code == 404
    assert "Stage-B" in response.json()["detail"]


def test_velocity_field_serves_a_quiver_with_its_own_front(
    client: TestClient, trained_run: str
):
    """The inferred flow travels with the interface it flows around: an arrow
    field without that boundary cannot be read, and the two must come from the
    same instant, which one payload guarantees and two requests do not."""
    body = client.get(f"/api/runs/{trained_run}/velocity", params={"t": 0.1}).json()

    assert body["unit"] == "mm·s⁻¹"
    # Two lattices: the arrows coarse (a vector per pixel is ink, not data),
    # the contour from the full field grid.
    assert len(body["u"]) == len(body["y_um"])
    assert len(body["u"][0]) == len(body["x_um"])
    assert len(body["v"]) == len(body["u"])
    assert body["speed_max"] >= body["speed_mean"] >= 0
    x0, x1, y0, y1 = body["domain_um"]
    assert x1 > x0 and y1 > y0
    # Every arrow anchor sits inside the channel, never on a wall.
    assert all(x0 < x < x1 for x in body["x_um"])
    assert all(y0 < y < y1 for y in body["y_um"])
    assert isinstance(body["interface"], list)


def test_velocity_field_clamps_to_the_trained_span(client: TestClient, trained_run: str):
    """Past the footage the model is inventing, so the request is clamped rather
    than answered — the same rule the scalar maps follow."""
    body = client.get(f"/api/runs/{trained_run}/velocity", params={"t": 99.0}).json()

    assert body["t_star"] == pytest.approx(body["t_max_star"])
    assert body["t_ms"] >= 0


def test_velocity_field_for_an_unknown_run_is_404(client: TestClient):
    assert client.get("/api/runs/nope/velocity").status_code == 404
