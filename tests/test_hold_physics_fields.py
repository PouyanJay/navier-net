"""The pressure and temperature nets are held at their initialisation through
the warm-up, which does not train them, and train once the physics engages."""

from __future__ import annotations

import torch

from naviernet import training

from .conftest import staged_capsule_run

THERMAL = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.thermal_model=sharp",
    "model.momentum_closure=brinkman",
    "model.outlet_condition=zero_gradient",
    "model.outlet_pressure_Pa=5.0",
    "experiment.T_inlet_C=56.6",
    "experiment.T_init_C=58.0",
    "training.stage_b_warmup_steps=3",
    "training.collocation_time_bins=2",
]


def _net(model, name: str) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.nets[name].state_dict().items()}


def _changed(before: dict, after: dict) -> bool:
    return any(not torch.equal(before[k], after[k]) for k in before)


def test_the_warm_up_leaves_pressure_and_temperature_at_their_initialisation(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*THERMAL, "training.steps=1"])
    first, _, _ = training.train(cfg, paths)
    pressure, temperature, velocity = _net(first, "p"), _net(first, "T"), _net(first, "u")

    cfg, paths = staged_capsule_run(tmp_path, [*THERMAL, "training.steps=2"])
    warm, _, _ = training.train(cfg, paths)

    assert not _changed(pressure, _net(warm, "p")), "the outlet pull alone moves nothing"
    assert not _changed(temperature, _net(warm, "T"))
    assert _changed(velocity, _net(warm, "u")), "the warm-up's own fields still train"


def test_the_physics_stage_trains_them(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*THERMAL, "training.steps=3"])
    warm, _, _ = training.train(cfg, paths)
    pressure, temperature = _net(warm, "p"), _net(warm, "T")

    cfg, paths = staged_capsule_run(tmp_path, [*THERMAL, "training.steps=2"])
    trained, _, state = training.train(cfg, paths)

    assert state["done"] == 5
    assert _changed(pressure, _net(trained, "p"))
    assert _changed(temperature, _net(trained, "T"))
    assert all(p.requires_grad for p in trained.nets["p"].parameters())


def test_the_warm_up_rebalances_without_the_held_fields(tmp_path):
    # The rebalancer measures every term. The temperature conditions reach
    # only the temperature net, held through the warm-up, so they have no
    # gradient there: the step measures the rest and carries on.
    cfg, paths = staged_capsule_run(
        tmp_path, [*THERMAL, "training.rebalance_every=1", "training.steps=2"]
    )

    _, _, state = training.train(cfg, paths)

    assert state["done"] == 2
    assert state["w"]["theta_bc"] == float(cfg.training.weights.theta_bc)
