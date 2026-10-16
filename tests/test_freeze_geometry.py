"""Two-stage training: the geometry frozen at the warm-up's fit once the physics
engages, the fields trained against the physics alone, the rebalancer
equalising the physics terms among themselves."""

from __future__ import annotations

import pytest
import torch

from naviernet import training
from naviernet.training import _physics_reference, _rebalance

from .conftest import staged_capsule_run

TWO_STAGE = [
    "model=stage_b",
    "model.front_geometry=true",
    "training.stage_b_warmup_steps=2",
    "training.freeze_geometry=true",
    "training.frozen_weighting=equalised",
    "training.rebalance_every=1",
    "training.log_every=1",
]


def _geometry(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.nets["phi"].state_dict().items()}


def _fields(model) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{k}": v.detach().clone()
        for name, net in model.nets.items()
        if name != "phi"
        for k, v in net.state_dict().items()
    }


def _changed(before: dict, after: dict) -> list[str]:
    return [k for k in before if not torch.equal(before[k], after[k])]


def test_the_geometry_holds_still_through_the_physics_stage_while_the_fields_train(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*TWO_STAGE, "training.steps=2"])
    warm, _, _ = training.train(cfg, paths)
    geometry, fields = _geometry(warm), _fields(warm)

    cfg, paths = staged_capsule_run(tmp_path, [*TWO_STAGE, "training.steps=3"])
    trained, _, state = training.train(cfg, paths)

    assert state["done"] == 5
    assert _changed(geometry, _geometry(trained)) == [], "the outline is frozen"
    assert _changed(fields, _fields(trained)), "the fields keep training"
    assert all(p.requires_grad is False for p in trained.nets["phi"].parameters())


def test_the_warm_up_still_shapes_the_geometry(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*TWO_STAGE, "training.steps=1"])
    torch.manual_seed(cfg.training.seed)
    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN

    data = BubbleDataset(cfg, paths)
    fresh = BubblePINN(cfg, geometry=training._geometry_priors(cfg, data))
    before = _geometry(fresh)

    model, _, _ = training.train(cfg, paths)

    assert _changed(before, _geometry(model)), "the warm-up fits the outline"


def test_one_run_through_the_engagement_freezes_at_the_handoff(tmp_path, monkeypatch):
    """Engagement inside one call: the warm-up's last geometry is the one the
    physics stage keeps, and the warm-up's fit is recorded."""
    cfg, paths = staged_capsule_run(tmp_path, [*TWO_STAGE, "training.steps=4"])
    snapshots: list[dict] = []
    original = training._record_warmup_fit

    def record(cfg, model, data, state):
        snapshots.append(_geometry(model))
        original(cfg, model, data, state)

    monkeypatch.setattr(training, "_record_warmup_fit", record)

    model, _, state = training.train(cfg, paths)

    assert len(snapshots) == 1
    assert _changed(snapshots[0], _geometry(model)) == []
    assert "warmup_iou" in state
    record_keys = {k for r in state["hist"] if r["step"] > 2 for k in r}
    assert not {"fv_normal", "fv_apex"} & record_keys


def test_the_physics_reference_is_the_median_weighted_norm():
    weights = {"vof": 1.0, "div": 2.0, "bc": 4.0}
    norms = {"vof": 1.0, "div": 1.0, "bc": 1.0}

    assert _physics_reference(weights, norms, ("vof", "div", "bc")) == pytest.approx(2.0)


def test_equalising_against_the_physics_reference_converges_on_one_contribution():
    weights = {"data": 10.0, "vof": 1.0, "div": 1.0, "bc": 1.0}
    norms = {"vof": 0.1, "div": 1.0, "bc": 10.0}
    terms = ("vof", "div", "bc")
    for _ in range(60):
        _rebalance(weights, norms, terms, 10.0, _physics_reference(weights, norms, terms))

    contributions = [weights[k] * norms[k] for k in terms]
    assert max(contributions) / min(contributions) == pytest.approx(1.0, rel=1e-3)


def test_fixed_weighting_holds_the_physics_at_its_configured_weights(tmp_path):
    """The default after the freeze: no rebalancing, so every physics weight is
    the configured one at the end of the run."""
    overrides = [o for o in TWO_STAGE if "frozen_weighting" not in o]
    cfg, paths = staged_capsule_run(tmp_path, [*overrides, "training.steps=5"])

    _, _, state = training.train(cfg, paths)

    configured = training._initial_state(cfg, _equations(cfg))["w"]
    assert state["w"] == configured


def test_an_unknown_frozen_weighting_is_refused(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*TWO_STAGE, "training.frozen_weighting=equal"])

    with pytest.raises(ValueError, match="frozen_weighting"):
        training.train(cfg, paths)


def _equations(cfg):
    from naviernet.physics import registry

    return registry.enabled_equations(cfg.model.fields, **registry.flags_from(cfg.model))


def test_freezing_needs_a_warm_up(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        ["model=stage_b", "model.front_geometry=true", "training.freeze_geometry=true"],
    )

    with pytest.raises(ValueError, match="stage_b_warmup_steps"):
        training.train(cfg, paths)


def test_freezing_needs_the_explicit_geometry(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        ["model=stage_b", "training.stage_b_warmup_steps=1", "training.freeze_geometry=true"],
    )

    with pytest.raises(ValueError, match="front_geometry"):
        training.train(cfg, paths)
