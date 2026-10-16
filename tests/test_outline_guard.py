"""The images as the binding measurement: the warm-up's fit recorded at the
step the physics engages, the outline guard that stops a run drifting off the
masks, the cap on the rebalanced physics weights, and the temperature
conditions held at their configured weight."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from naviernet import training
from naviernet.physics import registry
from naviernet.training import OutlineDrift, _rebalance

from .conftest import staged_capsule_run

STAGE_B = ["model=stage_b", "training.stage_b_warmup_steps=1", "training.log_every=1"]


def test_the_warm_up_fit_is_recorded_at_the_step_the_physics_engages(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*STAGE_B, "training.steps=2"])

    _, data, state = training.train(cfg, paths)

    recorded = state["warmup_iou"]
    assert set(recorded) == {str(f) for f in data.frame_numbers[: data.n_event]}
    assert all(0.0 <= v <= 1.0 for v in recorded.values())
    saved = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    assert saved["state"]["warmup_iou"] == recorded, "the record travels with the checkpoint"


def test_no_record_without_a_warm_up(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, ["model=stage_b", "training.steps=2"])

    _, _, state = training.train(cfg, paths)

    assert "warmup_iou" not in state


def test_the_outline_guard_stops_a_run_that_drifts_and_keeps_its_checkpoint(
    tmp_path, monkeypatch
):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [*STAGE_B, "training.steps=6", "training.outline_guard_every=2"],
    )
    fits = iter([0.9, 0.9, 0.7])  # the warm-up's fit, then a hold, then a drift

    def scripted(cfg, model, data):
        fit = next(fits)
        return {"1": fit, "2": fit}

    monkeypatch.setattr(training, "_outline_fit", scripted)

    with pytest.raises(OutlineDrift, match="step 4"):
        training.train(cfg, paths)

    state = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)["state"]
    assert state["done"] == 4, "the checkpoint at the step the drift was seen"
    assert [r["step"] for r in state["outline_hist"]] == [2, 4]
    assert state["outline_hist"][-1]["iou_mean"] == pytest.approx(0.7)
    assert state["outline_hist"][-1]["floor"] == pytest.approx(0.85)


def test_the_outline_guard_lets_a_steady_run_finish(tmp_path, monkeypatch):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            *STAGE_B,
            "training.steps=5",
            "training.outline_guard_every=2",
            "training.outline_guard_drop=0.1",
        ],
    )
    monkeypatch.setattr(
        training, "_outline_fit", lambda cfg, model, data: {"1": 0.9, "2": 0.85}
    )

    _, _, state = training.train(cfg, paths)

    assert state["done"] == 5
    assert [r["step"] for r in state["outline_hist"]] == [2, 4]
    assert all(r["iou_mean"] >= r["floor"] for r in state["outline_hist"])


def test_the_outline_guard_needs_a_warm_up_to_set_its_floor(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path, ["model=stage_b", "training.steps=2", "training.outline_guard_every=1"]
    )

    with pytest.raises(ValueError, match="stage_b_warmup_steps"):
        training.train(cfg, paths)


def test_the_outline_guard_needs_stage_b_physics_to_engage(tmp_path):
    """A Stage-A model with a warm-up never engages anything, so the guard's
    floor would never be recorded and the guard could never fire."""
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            "training.steps=2",
            "training.stage_b_warmup_steps=1",
            "training.outline_guard_every=1",
        ],
    )

    with pytest.raises(ValueError, match="no Stage-B physics"):
        training.train(cfg, paths)


def test_the_guard_waits_for_the_physics_to_engage(tmp_path, monkeypatch):
    """Through the warm-up there is no floor to hold the run to, so the guard
    measures nothing there."""
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            "model=stage_b",
            "training.stage_b_warmup_steps=3",
            "training.steps=4",
            "training.outline_guard_every=1",
            "training.outline_guard_drop=1.0",  # measure, never stop: the fits are real
        ],
    )
    calls: list[int] = []
    original = training._outline_fit

    def counting(cfg, model, data):
        calls.append(1)
        return original(cfg, model, data)

    monkeypatch.setattr(training, "_outline_fit", counting)

    _, _, state = training.train(cfg, paths)

    # One call records the warm-up's fit at step 4 (engagement); one is the
    # guard's measurement at that same step, the only post-warm-up step.
    assert len(calls) == 2
    assert [r["step"] for r in state["outline_hist"]] == [4]


def test_the_rebalancer_cap_holds_every_physics_weight_under_the_data_weight():
    weights = {"data": 10.0, "vof": 1.0, "div": 1.0, "src": 0.1, "bc": 1.0}
    norms = {"data": 1.0, "vof": 1e-6, "div": 1e-6, "bc": 1e-6}
    for _ in range(40):
        _rebalance(weights, norms, ("vof", "div", "bc"), max_ratio=10.0)

    assert all(weights[k] == pytest.approx(100.0) for k in ("vof", "div", "bc"))
    uncapped = {"data": 10.0, "vof": 1.0, "div": 1.0, "src": 0.1, "bc": 1.0}
    for _ in range(40):
        _rebalance(uncapped, norms, ("vof", "div", "bc"))
    assert all(uncapped[k] == pytest.approx(1e3) for k in ("vof", "div", "bc"))


def test_the_rebalancer_cap_never_exceeds_the_fixed_ceiling():
    weights = {"data": 1e3, "vof": 1.0}
    for _ in range(40):
        _rebalance(weights, {"data": 1.0, "vof": 1e-9}, ("vof",), max_ratio=10.0)

    assert weights["vof"] == pytest.approx(1e3)


def test_a_negative_cap_is_refused(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, ["training.rebalance_max_ratio=-1"])

    with pytest.raises(ValueError, match="rebalance_max_ratio"):
        training.train(cfg, paths)


def test_the_cap_reaches_the_training_loop(tmp_path):
    """A run rebalanced under the cap ends with no physics weight above the
    cap times its data weight."""
    cfg, paths = staged_capsule_run(
        tmp_path,
        ["training.steps=3", "training.rebalance_every=1", "training.rebalance_max_ratio=0.1"],
    )

    _, _, state = training.train(cfg, paths)

    weights = state["w"]
    # The cap is a tenth of the data weight (1.0): bc starts at 5, so an
    # unwired cap would leave it above the bound.
    assert all(weights[k] <= 0.1 * weights["data"] + 1e-9 for k in ("vof", "div", "bc"))


def test_the_temperature_conditions_are_never_rebalanced():
    theta_bc = next(e for e in registry.REGISTRY if e.id == "theta_bc")

    assert theta_bc.rebalanced is False


def test_the_source_penalty_spares_the_footprint_under_film_evaporation():
    """With the film evaporating across the footprint, only the liquid outside
    the bubble is forbidden to the source."""
    from naviernet.physics.residuals import StageAResiduals

    class Model:
        footprint_evaporation = True
        liquid_film = False

        def alpha(self, x, c=None):
            return torch.full_like(x[:, :1], 0.9)

    source = torch.ones(4, 1)
    residuals = StageAResiduals(source, source, source, torch.zeros(4, 1))
    ctx = registry.LossContext(Model(), torch.zeros(4, 3))
    ctx._res_a = residuals

    penalty = next(e for e in registry.REGISTRY if e.id == "src").pointwise(ctx)

    assert torch.allclose(penalty, torch.full((4, 1), 0.1**2))


def test_the_gradient_norm_is_recorded_every_logged_step(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, ["training.steps=2", "training.log_every=1"])

    _, _, state = training.train(cfg, paths)

    assert all(np.isfinite(r["grad_norm"]) and r["grad_norm"] > 0 for r in state["hist"])
