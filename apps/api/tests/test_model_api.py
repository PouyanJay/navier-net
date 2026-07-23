"""Model architecture endpoint."""

from __future__ import annotations

import pytest


def test_model_architecture(client):
    r = client.get("/api/model/sample")
    assert r.status_code == 200
    net = r.json()
    assert net["fields"] == ["phi", "u", "v", "s"]
    assert net["hidden"] == 96
    assert net["layers"] == 4
    assert net["fourier_feats"] == 64
    assert net["fourier_scale"] == pytest.approx(3.0)
    assert net["alpha_eps"] == pytest.approx(0.05)
    assert net["nodewise_activation"] is True


def test_model_of_unknown_dataset_is_404(client):
    r = client.get("/api/model/made-up")
    assert r.status_code == 404
    assert "made-up" in r.json()["detail"]
