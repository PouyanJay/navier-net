"""Grids and probes evaluate the trained checkpoint itself."""

from __future__ import annotations

import json
import struct

import numpy as np
from fastapi.testclient import TestClient
from naviernet_studio import grids

from conftest import RUN_ID


def _unpack(body: bytes) -> tuple[dict, np.ndarray]:
    assert body[:4] == grids.MAGIC
    length = struct.unpack("<I", body[4:8])[0]
    header = json.loads(body[8 : 8 + length])
    return header, np.frombuffer(body[8 + length :], dtype="<f4")


def test_a_grid_packs_one_float32_plane_per_quantity(client: TestClient):
    response = client.get(
        f"/api/runs/{RUN_ID}/grid", params={"q": "alpha,speed", "frame": 3, "nx": 16, "ny": 8}
    )

    assert response.status_code == 200
    header, payload = _unpack(response.content)
    assert header["shape"] == [8, 16]
    assert [q["id"] for q in header["quantities"]] == ["alpha", "speed"]
    assert payload.size == 2 * 8 * 16
    alpha = payload[: 8 * 16]
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    assert header["quantities"][1]["unit"] == "mm/s"


def test_a_grid_can_be_exported_as_csv_with_provenance(client: TestClient):
    response = client.get(
        f"/api/runs/{RUN_ID}/grid",
        params={"q": "u", "frame": 1, "nx": 8, "ny": 8, "fmt": "csv"},
    )

    lines = response.text.splitlines()
    assert response.headers["content-type"].startswith("text/csv")
    assert lines[0].startswith("# run: ")
    header_row = next(line for line in lines if not line.startswith("#"))
    assert header_row == "x_um,y_um,u"
    assert len([line for line in lines if not line.startswith("#")]) == 1 + 64


def test_grid_rejects_unknown_quantities_and_frames(client: TestClient):
    assert (
        client.get(f"/api/runs/{RUN_ID}/grid", params={"q": "bogus", "frame": 1}).status_code
        == 422
    )
    assert (
        client.get(f"/api/runs/{RUN_ID}/grid", params={"q": "u", "frame": 99}).status_code
        == 422
    )
    # Edge quantities are not point quantities.
    assert (
        client.get(f"/api/runs/{RUN_ID}/grid", params={"q": "yl_jump", "frame": 1}).status_code
        == 422
    )


def test_a_line_probe_samples_every_named_frame(client: TestClient):
    response = client.post(
        f"/api/runs/{RUN_ID}/probe",
        json={
            "kind": "line",
            "points_um": [[10.0, 150.0], [290.0, 150.0]],
            "quantities": ["u", "alpha"],
            "frames": [2, 5, 9],
            "n": 12,
        },
    )

    body = response.json()
    assert response.status_code == 200, response.text
    assert [f["frame"] for f in body["frames"]] == [2, 5, 9]
    first = body["frames"][0]
    assert len(first["values"]["u"]) == 12
    assert first["stations"]["s_um"][0] == 0.0
    assert abs(first["stations"]["s_um"][-1] - 280.0) < 1e-6
    assert body["units"]["u"] == "mm/s"


def test_a_section_probe_follows_the_channel(client: TestClient):
    response = client.post(
        f"/api/runs/{RUN_ID}/probe",
        json={
            "kind": "section",
            "section": "centreline",
            "quantities": ["speed"],
            "t_ms": [0.3, 0.37],
            "n": 5,
        },
    )

    body = response.json()
    assert response.status_code == 200, response.text
    ys = body["frames"][0]["stations"]["y_um"]
    # The centreline of the float32 pixel span, within a nanometre.
    assert all(abs(y - 150.0) < 1e-3 for y in ys)
    # 0.3 ms is exactly camera frame 3 of these tensors; 0.37 ms is between frames.
    assert body["frames"][0]["frame"] == 3
    assert body["frames"][1]["frame"] is None


def test_points_probe_exports_csv_rows_per_frame_and_point(client: TestClient):
    response = client.post(
        f"/api/runs/{RUN_ID}/probe",
        params={"fmt": "csv"},
        json={
            "kind": "points",
            "points_um": [[50, 150], [100, 150]],
            "quantities": ["u"],
            "frames": [1, 2],
        },
    )

    rows = [line for line in response.text.splitlines() if not line.startswith("#")]
    assert rows[0] == "frame,t_ms,station,s_um,x_um,y_um,u"
    assert len(rows) == 1 + 4


def test_probe_without_an_instant_is_422(client: TestClient):
    response = client.post(
        f"/api/runs/{RUN_ID}/probe",
        json={"kind": "points", "points_um": [[50, 150]], "quantities": ["u"]},
    )
    assert response.status_code == 422
