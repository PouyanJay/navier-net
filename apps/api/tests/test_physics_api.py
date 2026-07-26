"""Physics endpoint: the governing equations, enabling Stage B, live groups."""

from __future__ import annotations

import pytest


def _by_id(payload):
    return {e["id"]: e for e in payload["equations"]}


def test_physics_lists_the_registry_equations_with_state(client):
    r = client.get("/api/physics/sample")
    assert r.status_code == 200
    body = r.json()

    eqs = _by_id(body)
    assert set(eqs) == {"vof", "div", "src", "bc", "mom", "energy", "evap"}
    # Stage-A equations are core and on; Stage B is off until its fields exist.
    assert eqs["vof"]["core"] and eqs["vof"]["enabled"]
    assert eqs["mom"]["stage"] == "B" and not eqs["mom"]["enabled"]
    assert eqs["mom"]["fields_added"] == ["p"]
    assert body["fields"] == ["phi", "u", "v", "s"]
    # Momentum carries real TeX and its dimensionless groups.
    assert "\\mathrm{Re}" in eqs["mom"]["tex"]
    assert eqs["mom"]["groups"] == ["Re", "We", "hele_shaw"]


def test_physics_groups_are_the_real_computed_values(client):
    body = client.get("/api/physics/sample").json()
    assert body["groups"]["Re"] == pytest.approx(215.5, rel=1e-3)
    assert body["groups"]["dT_ref"] == pytest.approx(28.74, rel=1e-3)  # Stage-B group


def test_physics_of_unknown_dataset_is_404(client):
    r = client.get("/api/physics/made-up")
    assert r.status_code == 404


def test_enabling_momentum_unlocks_pressure(client):
    r = client.put("/api/physics/sample", json={"enabled": ["mom"], "weights": {}})
    assert r.status_code == 200
    body = r.json()

    assert "p" in body["fields"] and "T" not in body["fields"]
    eqs = _by_id(body)
    assert eqs["mom"]["enabled"] is True
    assert eqs["energy"]["enabled"] is False, "temperature was not unlocked"


def test_enabling_energy_unlocks_temperature_and_evaporation(client):
    body = client.put(
        "/api/physics/sample", json={"enabled": ["mom", "energy"], "weights": {"mom": 2.0}}
    ).json()

    assert {"p", "T"} <= set(body["fields"])
    eqs = _by_id(body)
    assert eqs["energy"]["enabled"] and eqs["evap"]["enabled"]
    assert eqs["mom"]["weight"] == pytest.approx(2.0), "saved weight is reflected"


def test_physics_edit_persists_across_requests(client):
    client.put("/api/physics/sample", json={"enabled": ["mom"], "weights": {}})
    again = client.get("/api/physics/sample").json()
    assert "p" in again["fields"]
    assert _by_id(again)["mom"]["enabled"] is True


def test_unknown_equation_is_rejected(client):
    r = client.put("/api/physics/sample", json={"enabled": ["made_up"], "weights": {}})
    assert r.status_code == 400


def test_negative_weight_is_rejected(client):
    r = client.put("/api/physics/sample", json={"enabled": [], "weights": {"vof": -1.0}})
    assert r.status_code == 400
