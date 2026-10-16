"""The quadrature instants of the front and the mass balance: fixed grids by
default, a fresh stratified draw each step under training.quadrature_jitter."""

from __future__ import annotations

import numpy as np
import torch

from naviernet import training
from naviernet.data.dataset import BubbleDataset
from naviernet.physics.groups import compute_groups

from .conftest import staged_capsule_run

SHARP = ["model=stage_b", "model.front_geometry=true", "model.sharp_interface=true"]


def test_fixed_grids_by_default(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*SHARP, "training.mass_balance=true"])
    data = BubbleDataset(cfg, paths)

    times = training._front_times(cfg, data, "cpu")
    plan = training._mass_balance_plan(cfg, data, compute_groups(cfg), "cpu")

    d = data.domain
    assert torch.allclose(
        times.ravel(), torch.linspace(d.t_min, d.t_max, cfg.model.front_times)
    )
    expected = torch.linspace(d.t_min, d.t_max, cfg.training.mass_balance_times)
    assert torch.allclose(torch.unique(plan.points[:, 2]), expected)
    assert training._jitter_rng(cfg.training, np.random.default_rng(0)) is None


def test_jittered_instants_are_stratified_and_change_every_draw(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path, [*SHARP, "training.mass_balance=true", "training.quadrature_jitter=true"]
    )
    data = BubbleDataset(cfg, paths)
    rng = np.random.default_rng(0)
    d = data.domain
    n = int(cfg.model.front_times)
    edges = np.linspace(d.t_min, d.t_max, n + 1)

    first = training._front_times(cfg, data, "cpu", rng).ravel().numpy()
    second = training._front_times(cfg, data, "cpu", rng).ravel().numpy()
    plan = training._mass_balance_plan(cfg, data, compute_groups(cfg), "cpu", rng)

    assert np.all((edges[:-1] <= first) & (first <= edges[1:])), "one instant per stratum"
    assert not np.allclose(first, second), "a fresh draw each step"
    assert plan.n_times == cfg.training.mass_balance_times
    assert torch.all(plan.points[:, 2] >= d.t_min) and torch.all(plan.points[:, 2] <= d.t_max)
    assert training._jitter_rng(cfg.training, rng) is rng


def test_a_jittered_run_trains_and_is_reproducible(tmp_path):
    overrides = [
        *SHARP,
        "training.mass_balance=true",
        "training.quadrature_jitter=true",
        "training.steps=3",
        "training.log_every=1",
    ]
    cfg, paths = staged_capsule_run(tmp_path, overrides)
    _, _, state = training.train(cfg, paths)
    cfg2, paths2 = staged_capsule_run(tmp_path / "again", overrides)
    _, _, again = training.train(cfg2, paths2)

    assert all(np.isfinite(v) for v in state["hist"][-1].values())
    assert state["hist"][-1]["mass_bal"] == again["hist"][-1]["mass_bal"], "seeded draw"
