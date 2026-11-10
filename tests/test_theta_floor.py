"""The floor under the liquid temperature (``theta_floor``): the liquid is never
colder than the inlet, a bound the solver cannot rebalance away."""

from __future__ import annotations

import torch

from naviernet import training
from naviernet.physics import registry
from naviernet.physics.groups import compute_groups
from naviernet.physics.residuals import subcooling_residual

from .conftest import staged_capsule_run
from .test_energy_relative import THERMAL, _context


def test_the_residual_is_the_liquid_shortfall_below_the_inlet(tmp_path):
    cfg, model, ctx = _context(tmp_path)
    groups = compute_groups(cfg)

    residual = subcooling_residual(model, ctx.x_coll, groups)

    theta = model.temperature(ctx.x_coll).detach()
    liquid = 1.0 - model.alpha(ctx.x_coll).detach()
    expected = liquid * (groups["theta_inlet"] - theta).clamp(min=0.0)
    assert torch.allclose(residual, expected)
    assert bool((residual >= 0.0).all())
    assert bool((residual[theta >= groups["theta_inlet"]] == 0.0).all()), (
        "liquid at or above the inlet temperature owes nothing"
    )


def test_the_floor_is_a_fixed_weight_sharp_thermal_term_the_temperature_stage_keeps(tmp_path):
    cfg, _, _ = _context(tmp_path)
    floor = next(e for e in registry.REGISTRY if e.id == "theta_floor")
    enabled = registry.enabled_equations(cfg.model.fields, **registry.flags_from(cfg.model))

    assert floor in enabled
    assert floor.rebalanced is False
    assert floor.on_collocation and floor in registry.collocation_equations(enabled)
    assert "theta_floor" in training.TEMPERATURE_TERMS


def test_the_floor_is_off_without_the_sharp_thermal_model(tmp_path):
    cfg, _ = staged_capsule_run(tmp_path, ["model=stage_b"])

    enabled = registry.enabled_equations(cfg.model.fields, **registry.flags_from(cfg.model))

    assert "theta_floor" not in {e.id for e in enabled}


def test_a_run_with_the_floor_trains_and_reports_it(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            *THERMAL,
            "training.stage_b_warmup_steps=1",
            "training.steps=3",
            "training.log_every=1",
            "training.collocation_time_bins=2",
        ],
    )

    _, _, state = training.train(cfg, paths)

    assert state["w"]["theta_floor"] == float(cfg.training.weights.theta_floor)
    assert torch.isfinite(torch.tensor(state["hist"][-1]["theta_floor"]))
