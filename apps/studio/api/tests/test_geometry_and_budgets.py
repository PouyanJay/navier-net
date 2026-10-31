"""Geometry against the images, the interface contour, and the budgets."""

from __future__ import annotations

from fastapi.testclient import TestClient
from PIL import Image

from conftest import RUN_ID


def test_the_transform_carries_the_pixel_mapping(client: TestClient):
    body = client.get(f"/api/runs/{RUN_ID}/geometry/transform").json()

    assert body["frame_px"] == [16, 140]
    assert body["um_per_px"] == 4.3
    assert body["y_roi_rows"] == [64, 76]
    assert body["x_flipped"] is True
    assert body["walls_um"][1] > body["walls_um"][0]


def test_raw_frames_are_served_as_png_and_missing_ones_404(client: TestClient):
    ok = client.get(f"/api/runs/{RUN_ID}/geometry/frame/3.png")
    missing = client.get(f"/api/runs/{RUN_ID}/geometry/frame/99.png")

    assert ok.status_code == 200 and ok.headers["content-type"] == "image/png"
    assert ok.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert missing.status_code == 404


def test_agreement_reports_iou_and_both_contours(client: TestClient):
    body = client.get(f"/api/runs/{RUN_ID}/geometry/agreement", params={"frame": 4}).json()

    assert 0.0 <= body["iou"] <= 1.0
    assert body["area_measured_mm2"] > 0.0
    assert isinstance(body["measured_rings_um"], list)
    assert body["role"] == "train"
    assert (
        client.get(f"/api/runs/{RUN_ID}/geometry/agreement", params={"frame": 99}).status_code
        == 422
    )


def test_the_interface_contour_names_an_instant(client: TestClient):
    body = client.get(f"/api/runs/{RUN_ID}/interface", params={"t_ms": 0.75}).json()

    assert abs(body["t_ms"] - 0.75) < 1e-9
    assert isinstance(body["rings_um"], list)


def test_the_flux_budget_compares_liquid_flux_with_the_rule(client: TestClient):
    body = client.get(f"/api/runs/{RUN_ID}/budgets/flux", params={"frame": 5}).json()

    n = len(body["x_um"])
    assert (
        len(body["q_liquid_mL_h"]) == n == len(body["q_required_mL_h"]) == len(body["defect"])
    )
    assert body["q_inlet_mL_h"] > 0.0


def test_the_heat_budget_needs_an_event_frame(client: TestClient):
    ok = client.get(f"/api/runs/{RUN_ID}/budgets/heat", params={"frame": 2})
    truncated = client.get(f"/api/runs/{RUN_ID}/budgets/heat", params={"frame": 11})

    assert ok.status_code == 200, ok.text
    assert "latent_for_growth_mW" in ok.json()
    assert truncated.status_code == 422


def test_edge_and_gap_budget_say_what_a_stage_a_run_lacks(client: TestClient):
    edge = client.get(f"/api/runs/{RUN_ID}/edge", params={"q": "kappa_par", "frame": 2})
    gap = client.get(f"/api/runs/{RUN_ID}/budgets/gap", params={"frame": 2})

    assert edge.status_code == 404 and "front" in edge.json()["detail"]
    assert gap.status_code == 404 and "pressure" in gap.json()["detail"]


def test_the_cleaned_frame_says_how_to_generate_it_when_missing(client: TestClient):
    response = client.get(
        f"/api/runs/{RUN_ID}/geometry/frame/3.png", params={"mode": "cleaned"}
    )

    assert response.status_code == 404
    assert "apps/studio/tools/generate_cleaned.py" in response.json()["detail"]


def test_a_stored_cleaned_frame_is_served_from_the_repository(client: TestClient, repo_root):
    folder = repo_root / "apps" / "studio" / "cleaned" / RUN_ID
    folder.mkdir(parents=True)
    Image.new("RGB", (16, 140), (90, 90, 90)).save(folder / "frame_03.png")

    response = client.get(
        f"/api/runs/{RUN_ID}/geometry/frame/3.png", params={"mode": "cleaned"}
    )

    assert response.status_code == 200, response.text
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
