"""The illustrated mode: what can be checked without calling the image model."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from naviernet_studio import illustration
from PIL import Image

from conftest import RUN_ID


def _disc(size: int, radius: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return (xx - size // 2) ** 2 + (yy - size // 2) ** 2 <= radius**2


def test_the_protected_band_is_the_outline_and_not_the_interior():
    bubble = _disc(80, 25)

    band = illustration.protected_band(bubble)

    assert band[40, 40 + 25] or band[40, 40 + 24]
    assert not band[40, 40]
    assert not band[0, 0]


def test_the_composite_restores_the_recording_inside_the_protected_band():
    recorded = np.full((10, 10), 50.0, dtype=np.float32)
    generated = np.full((10, 10), 200.0, dtype=np.float32)
    protected = np.zeros((10, 10), dtype=bool)
    protected[4:6, :] = True

    mixed = illustration.composite(generated, recorded, protected)

    assert np.all(mixed[4:6, :] == 50.0)
    assert np.all(mixed[0:4, :] == 200.0)


def test_the_edit_mask_is_opaque_only_where_the_recording_must_be_kept():
    protected = np.zeros((8, 8), dtype=bool)
    protected[2, 3] = True

    png = illustration.edit_mask_png(protected)
    alpha = np.asarray(Image.open(io.BytesIO(png)))[..., 3]

    assert alpha[2, 3] == 255
    assert alpha[0, 0] == 0


def test_without_a_key_the_illustrated_frame_says_what_to_set(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv(illustration.KEY_ENV, raising=False)

    response = client.get(
        f"/api/runs/{RUN_ID}/geometry/frame/3.png", params={"mode": "illustrated"}
    )

    assert response.status_code == 503
    assert illustration.KEY_ENV in response.json()["detail"]


def test_a_bundled_illustrated_frame_is_served_without_a_key(
    client: TestClient, repo_root, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv(illustration.KEY_ENV, raising=False)
    folder = repo_root / illustration.BUNDLED_DIR / RUN_ID
    folder.mkdir(parents=True)
    Image.new("RGB", (16, 140), (120, 120, 120)).save(folder / "frame_03.png")

    response = client.get(
        f"/api/runs/{RUN_ID}/geometry/frame/3.png", params={"mode": "illustrated"}
    )

    assert response.status_code == 200, response.text
    assert response.headers["x-image-provenance"] == illustration.LABEL
