"""The advanced-overrides gate: the strict allowlist, and the round trip.

The overrides field is free text from the network headed for Hydra compose, so
the validator is exercised as a matrix (every rejection names its reason) and
the accept path is proven end-to-end: a launched run's own config snapshot
carries the overridden value.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from helpers import TINY_RUN, read_stream
from naviernet_api.models import RunLaunchRequest
from naviernet_api.services.overrides import OverrideRejected, validate_overrides


class TestValidateOverrides:
    def test_accepts_cli_only_dials_and_canonicalises(self):
        tokens = ["training.kin_grid=64", "training.rba_gamma=0.995"]
        assert validate_overrides(tokens) == [
            "training.kin_grid=64",
            "training.rba_gamma=0.995",
        ]

    def test_accepts_model_flags_the_form_does_not_own(self):
        # pressure_shape / nucleation_pulse exist on the schema but not on the
        # launch form; the overrides row is exactly how they stay reachable.
        assert validate_overrides(["model.pressure_shape=TRUE"]) == [
            "model.pressure_shape=true"
        ]

    def test_accepts_the_servers_one_computed_key(self):
        # stage_b_warmup_steps is computed server-side; an explicit override is
        # appended after it, so the expert's choice wins by last-wins order.
        assert validate_overrides(["training.stage_b_warmup_steps=0"]) == [
            "training.stage_b_warmup_steps=0"
        ]

    @pytest.mark.parametrize(
        ("token", "reason"),
        [
            ("training.kin_grid", "key=value"),
            ("training.frobnicate=1", "no such field"),
            ("training.weights.data=5", "nested"),
            ("paths.root=/etc", "outside training.* / model.*"),
            ("hydra.run.dir=/tmp", "outside training.* / model.*"),
            ("training.steps=99", "set by the launch form"),
            ("model.sharp_interface=true", "set by the launch form"),
            ("model.hidden=64", "series' saved model configuration"),
            ("training.causal_eps_schedule=1", "not a numeric or boolean"),
            ("model.per_field=3", "not a numeric or boolean"),
            ("model.pressure_shape=maybe", "not true/false"),
            ("training.kin_grid=3.5", "does not parse"),
            ("training.rba_gamma=1e12", "magnitude"),
            ("training.rba_gamma=nan", "magnitude"),
        ],
    )
    def test_rejects_with_the_reason(self, token: str, reason: str):
        with pytest.raises(OverrideRejected, match=reason.replace("*", r"\*")):
            validate_overrides([token])

    def test_rejects_a_duplicated_key(self):
        with pytest.raises(OverrideRejected, match="more than once"):
            validate_overrides(["training.kin_grid=64", "training.kin_grid=48"])

    def test_rejects_a_flood_of_tokens(self):
        tokens = [f"training.kin_grid={n}" for n in range(17)]
        with pytest.raises(OverrideRejected, match="at most 16"):
            validate_overrides(tokens)


def test_launch_rejects_a_form_owned_override_as_422(client: TestClient):
    response = client.post("/api/runs", json={**TINY_RUN, "overrides": ["training.steps=99"]})
    assert response.status_code == 422
    assert "set by the launch form" in response.json()["detail"]


def test_sweep_rejects_an_invalid_override_before_reserving(client: TestClient):
    response = client.post(
        "/api/sweeps",
        json={**TINY_RUN, "seeds": [0, 1], "overrides": ["model.hidden=64"]},
    )
    assert response.status_code == 422
    assert "Physics & model" in response.json()["detail"]
    # The gate fired before the slot was claimed: a good launch still goes through.
    assert client.post("/api/runs", json=TINY_RUN).status_code == 202


def test_launched_run_snapshot_records_the_override(client: TestClient, repo_root: Path):
    """The round trip: an accepted dial lands in the run's own config snapshot."""
    payload = {**TINY_RUN, "overrides": ["training.kin_grid=64"]}
    response = client.post("/api/runs", json=payload)
    assert response.status_code == 202
    run_id = response.json()["run_id"]

    events = read_stream(client, run_id)
    final = [e["data"] for e in events if e["event"] == "status"][-1]
    assert final["state"] == "done", f"run failed: {final.get('message')}"

    snapshot = yaml.safe_load(
        (repo_root / "outputs" / run_id / ".hydra" / "config.yaml").read_text()
    )
    assert snapshot["training"]["kin_grid"] == 64
    # The platform's own keys still composed around it.
    assert snapshot["training"]["steps"] == 2


def test_form_owned_keys_mirror_the_built_override_list():
    """FORM_OWNED_KEYS claims to mirror what the worker composes; hold it.

    The built list is _form_overrides plus the resolved interface flags (built
    here with every field available, so all six resolve to concrete values).
    Nested keys (training.weights.*) are excluded: the validator refuses them
    structurally, so they need no ownership entry.
    """
    from naviernet_api.services import run_manager
    from naviernet_api.services.overrides import FORM_OWNED_KEYS

    request = RunLaunchRequest(dataset="any")
    built = run_manager._form_overrides(request) + run_manager._resolved_interface_flags(
        request, ["p", "T"]
    )
    keys = [token.split("=", 1)[0] for token in built]
    top_level = {key for key in keys if key.count(".") == 1}
    assert top_level == set(FORM_OWNED_KEYS)
