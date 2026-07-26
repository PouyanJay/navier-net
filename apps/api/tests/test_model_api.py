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


def test_model_arch_edit_persists_globals_and_per_field(client):
    r = client.put(
        "/api/model/sample",
        json={"hidden": 128, "per_field": {"p": {"hidden": 64, "layers": 3}}},
    )
    assert r.status_code == 200
    net = r.json()
    assert net["hidden"] == 128
    assert net["per_field"]["p"] == {"hidden": 64, "layers": 3}

    again = client.get("/api/model/sample").json()
    assert again["hidden"] == 128 and again["per_field"]["p"]["hidden"] == 64


def test_model_edit_out_of_bounds_is_rejected(client):
    r = client.put("/api/model/sample", json={"hidden": 999999})
    assert r.status_code == 422  # Pydantic bounds reject before the service


def test_concurrent_model_and_physics_saves_do_not_collide(client, repo_root):
    """The two saves land on the same model.json concurrently (the web fires them
    together). A shared temp filename made one write's replace() remove the temp
    the other was about to replace -> 500. Race them for real and assert both
    always succeed, leave no temp file, and keep model.json valid."""
    from concurrent.futures import ThreadPoolExecutor

    raw = repo_root / "data" / "raw" / "sample"
    for _ in range(10):
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_phys = pool.submit(
                client.put,
                "/api/physics/sample",
                json={"enabled": ["mom"], "weights": {"mom": 2.0}},
            )
            f_model = pool.submit(
                client.put,
                "/api/model/sample",
                json={"hidden": 100, "per_field": {"p": {"hidden": 64}}},
            )
        assert f_phys.result().status_code == 200
        assert f_model.result().status_code == 200
        assert not list(raw.glob("*.tmp")), "atomic write left a temp file behind"

    from naviernet_api.services import datasets as datasets_service
    from naviernet_api.settings import Settings

    cfg = datasets_service.read_model_config(Settings(repo_root=repo_root), "sample")
    assert cfg["enabled"] == ["mom"] and cfg["hidden"] == 100
    assert cfg["per_field"]["p"]["hidden"] == 64


def test_saved_model_config_reaches_a_launched_run(client, repo_root):
    """The strongest guarantee: enabling momentum + a per-field override must show
    up in the config a run composes (runs compose via series_overrides)."""
    from naviernet_api.services import datasets as datasets_service
    from naviernet_api.services.config_service import compose_cfg_once
    from naviernet_api.settings import Settings

    client.put("/api/physics/sample", json={"enabled": ["mom"], "weights": {"mom": 3.0}})
    client.put("/api/model/sample", json={"per_field": {"p": {"hidden": 64}}})

    settings = Settings(repo_root=repo_root)
    overrides = datasets_service.series_overrides(settings, "sample")
    cfg = compose_cfg_once("sample", overrides=overrides)

    assert "p" in list(cfg.model.fields), "momentum's pressure field trains"
    assert cfg.model.per_field.p.hidden == 64
    assert cfg.training.weights.mom == pytest.approx(3.0)
