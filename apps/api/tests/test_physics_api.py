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
    assert set(eqs) == {
        "vof",
        "div",
        "src",
        "bc",
        "mom",
        "energy",
        "evap",
        "darcy",
        "kinematic",
        "laplace",
        "film",
        "film_depletion",
        "film_source",
    }
    # The sharp-interface equations are listed but inactive: the series does not
    # compose model.sharp_interface, so the diffuse treatment is the active one.
    assert body["sharp_interface"] is False
    assert eqs["laplace"]["mode"] == "sharp" and not eqs["laplace"]["enabled"]
    assert eqs["darcy"]["mode"] == "sharp" and not eqs["darcy"]["enabled"]
    # The film term is flag-gated on top of sharp mode, so it is listed inactive.
    assert eqs["film"]["mode"] == "sharp" and not eqs["film"]["enabled"]
    assert eqs["mom"]["mode"] == "diffuse"
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
    r = client.put("/api/physics/sample", json={"enabled": ["mom"], "weights": {"mom": -1.0}})
    assert r.status_code == 400


def test_stage_a_weight_is_not_editable_here(client):
    """Stage-A weights are owned by the run-launch form, not the physics page."""
    r = client.put("/api/physics/sample", json={"enabled": [], "weights": {"vof": 2.0}})
    assert r.status_code == 400
    assert "vof" in r.json()["detail"]


def test_absurd_weight_is_capped(client):
    r = client.put("/api/physics/sample", json={"enabled": ["mom"], "weights": {"mom": 1e300}})
    assert r.status_code == 400


def test_unknown_field_is_forbidden(client):
    r = client.put("/api/physics/sample", json={"enabled": [], "weights": {}, "typo": 1})
    assert r.status_code == 422  # extra="forbid"


def test_validation_carries_the_physics_diagnostics(client, tmp_path):
    """The Results view must be able to see whether the physics holds, not just
    whether the pixels overlap -- IoU alone is what hid the R3 failure."""
    from naviernet_api.services.physics import build_validation

    metrics = {
        "iou_mean": 0.93,
        "physics": {
            "laplace_error_nose": 0.06,
            "laplace_error_front": 0.18,
            "axial_capillary_gradient": 0.41,
            "neck_depth_model": 0.44,
            "neck_depth_measured": 0.47,
            "neck_location_model": 0.5,
            "neck_location_measured": 0.5,
            "profile_stations": [0.1, 0.5, 0.9],
            "per_frame": [
                {
                    "frame": 11,
                    "neck_depth_model": 0.44,
                    "neck_depth_measured": 0.47,
                    "neck_location_model": 0.5,
                    "neck_location_measured": 0.5,
                    "half_width_model": [0.2, 0.12, 0.39],
                    "half_width_measured": [0.21, 0.11, 0.39],
                }
            ],
            "residual_convergence": {"darcy": {"first": 6.8, "last": 0.31, "ratio": 0.046}},
        },
    }
    physics = build_validation("Series-1", metrics, {}).physics
    assert physics is not None
    assert physics.laplace_error_nose == pytest.approx(0.06)
    assert physics.per_frame[0].frame == 11
    assert physics.residual_convergence["darcy"].ratio == pytest.approx(0.046)


def test_validation_drops_unmeasurable_physics_values(client):
    """A Stage-A run cannot score the jump, so it writes NaN -- which is not JSON
    and must not reach the UI as a number it would plot."""
    from naviernet_api.services.physics import build_validation

    metrics = {"physics": {"laplace_error_nose": float("nan"), "neck_depth_measured": 0.47}}
    physics = build_validation("Series-1", metrics, {}).physics
    assert physics.laplace_error_nose is None
    assert physics.neck_depth_measured == pytest.approx(0.47)


def test_validation_without_an_explicit_front_has_no_physics_block(client):
    from naviernet_api.services.physics import build_validation

    assert build_validation("Series-1", {"iou_mean": 0.9, "physics": None}, {}).physics is None


def test_validation_drops_a_nan_nested_inside_residual_convergence(client):
    """`ratio` is NaN when a term's first window averaged exactly zero -- a real,
    reachable state. NaN is not JSON, so it must not survive into the response:
    `json.dumps` would emit a bare `NaN` token that JSON.parse rejects, breaking
    the Validation tab for exactly the runs these diagnostics exist to explain."""
    import json

    from naviernet_api.services.physics import build_validation

    metrics = {
        "physics": {
            "neck_depth_measured": 0.47,
            "residual_convergence": {
                "laplace": {"first": 0.0, "last": 0.0, "ratio": float("nan")},
                "darcy": {"first": 6.8, "last": 0.31, "ratio": 0.046},
            },
        }
    }
    physics = build_validation("Series-1", metrics, {}).physics

    assert physics.residual_convergence["darcy"].ratio == pytest.approx(0.046)
    assert physics.residual_convergence["laplace"].ratio is None
    # And the whole payload really is serialisable as strict JSON.
    json.dumps(physics.model_dump(), allow_nan=False)
