"""The physics stage's instants on a fixed grid (``training.instant_grid``):
every drawn instant moved to the nearest grid instant, so what the exact
liquid flow solves per instant is solved once in a run and kept."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from naviernet import training
from naviernet.data.dataset import BubbleDataset
from naviernet.models import potential_flow as pf
from naviernet.models.pinn import RATE_STENCIL, BubblePINN, _field_cache_size
from naviernet.physics.groups import compute_groups

from .conftest import make_config, staged_capsule_run, staged_walled_capsule_run

WINDOW = types.SimpleNamespace(t_min=-1.0, t_max=3.0)
GRID = 8  # instants every half unit of time over WINDOW
SHARP = ["model=stage_b", "model.front_geometry=true", "model.sharp_interface=true"]
EXACT = [
    *SHARP,
    "model.walls_at=physical",
    "model.hard_walls=true",
    "model.mirror_symmetry=true",
    "model.exact_continuity=true",
    "model.exact_continuity_body_sources=32",
    "model.exact_continuity_cap_sources=8",
]
PHYSICS_STAGE = [
    "model.outlet_condition=zero_gradient",
    "training.stage_b_warmup_steps=1",
    "training.freeze_geometry=true",
    "training.edge_transport_after_freeze=false",
    "training.collocation_time_bins=2",
    "training.quadrature_jitter=true",
    "training.log_every=1",
]


def _on_grid(times, grid: int = GRID) -> np.ndarray:
    return training._on_instant_grid(np.asarray(times, dtype=np.float32), WINDOW, grid)


# --- the grid ---------------------------------------------------------------------


def test_with_no_grid_the_instants_are_as_drawn():
    drawn = np.array([-0.83, 0.41, 2.2], dtype=np.float32)

    assert _on_grid(drawn, grid=0) is drawn


def test_each_instant_moves_to_the_nearest_grid_instant():
    moved = _on_grid([-0.83, 0.41, 2.2])

    assert moved.tolist() == [-1.0, 0.5, 2.0]
    assert moved.dtype == np.float32


def test_a_grid_instant_is_the_same_number_whichever_draw_lands_on_it():
    grid = 1024

    from_below = _on_grid([0.3345], grid)
    from_above = _on_grid([0.3370], grid)

    assert from_below[0] == from_above[0]


def test_two_instants_that_land_together_are_moved_a_grid_step_apart():
    moved = _on_grid([0.45, 0.55, 0.6])

    assert moved.tolist() == [0.5, 1.0, 1.5]


def test_instants_pushed_past_the_windows_end_come_back_inside():
    moved = _on_grid([2.8, 2.9, 2.95])

    assert moved.tolist() == [2.0, 2.5, 3.0]


def test_a_grid_too_coarse_for_a_draw_is_refused():
    with pytest.raises(ValueError, match="fewer instants"):
        _on_grid(np.linspace(-1.0, 3.0, GRID + 2))


# --- the collocation batch --------------------------------------------------------


def _batch() -> torch.Tensor:
    times = torch.tensor([0.41, 0.41, 2.2, -0.83, 2.2])
    places = torch.arange(10.0).reshape(5, 2)
    return torch.cat([places, times.reshape(-1, 1)], dim=1).requires_grad_(True)


def test_with_no_grid_the_batch_is_the_one_drawn():
    batch = _batch()

    assert training._collocation_on_grid(batch, WINDOW, 0) is batch


def test_the_batchs_instants_move_onto_the_grid_and_its_places_stay():
    batch = _batch()

    moved = training._collocation_on_grid(batch, WINDOW, GRID)

    assert moved[:, 2].tolist() == [0.5, 0.5, 2.0, -1.0, 2.0]
    assert torch.equal(moved[:, :2], batch[:, :2])


def test_the_moved_batch_is_a_fresh_leaf_the_equations_can_differentiate():
    moved = training._collocation_on_grid(_batch(), WINDOW, GRID)

    assert moved.is_leaf and moved.requires_grad


# --- when it acts -----------------------------------------------------------------


def _tcfg(**settings):
    base = dict(freeze_geometry=True, stage_b_warmup_steps=10, instant_grid=64)
    return types.SimpleNamespace(**{**base, **settings})


def test_the_grid_acts_only_once_the_geometry_is_frozen():
    tcfg, physics = _tcfg(), ("brinkman",)

    assert training._instant_grid(10, tcfg, physics) == 0
    assert training._instant_grid(11, tcfg, physics) == 64


def test_a_run_whose_geometry_trains_draws_every_instant_afresh():
    tcfg = _tcfg(freeze_geometry=False)

    assert training._instant_grid(11, tcfg, ("brinkman",)) == 0


def test_the_default_is_no_grid():
    assert make_config().training.instant_grid == 0


def test_a_negative_grid_is_refused(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, ["training.instant_grid=-1"])

    with pytest.raises(ValueError, match="instant_grid"):
        training.train(cfg, paths)


def test_a_grid_needs_the_collocation_in_time_bins(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path, ["training.instant_grid=64", "training.collocation_time_bins=0"]
    )

    with pytest.raises(ValueError, match="collocation_time_bins"):
        training.train(cfg, paths)


# --- the quadratures --------------------------------------------------------------


def _is_on_grid(times: torch.Tensor, domain, grid: int) -> bool:
    spacing = (float(domain.t_max) - float(domain.t_min)) / grid
    index = (times.double() - float(domain.t_min)) / spacing
    return bool(torch.allclose(index, index.round(), atol=1e-3))


def test_the_fronts_and_the_mass_balances_instants_are_drawn_on_the_grid(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path, [*SHARP, "training.mass_balance=true", "training.quadrature_jitter=true"]
    )
    data = BubbleDataset(cfg, paths)
    rng = np.random.default_rng(0)
    grid = 256

    front = training._front_times(cfg, data, "cpu", rng, grid).ravel()
    plan = training._mass_balance_plan(cfg, data, compute_groups(cfg), "cpu", rng, grid)

    assert _is_on_grid(front, data.domain, grid)
    assert _is_on_grid(torch.unique(plan.points[:, 2]), data.domain, grid)
    assert len(torch.unique(front)) == cfg.model.front_times, "still one per stratum"


def test_the_grid_leaves_the_random_draws_as_they_were(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*SHARP, "training.quadrature_jitter=true"])
    data = BubbleDataset(cfg, paths)
    grid = 256
    spacing = (data.domain.t_max - data.domain.t_min) / grid

    drawn = training._front_times(cfg, data, "cpu", np.random.default_rng(0)).ravel()
    on_grid = training._front_times(cfg, data, "cpu", np.random.default_rng(0), grid).ravel()

    assert float((on_grid - drawn).abs().max()) <= 0.5 * spacing + 1e-6


# --- the solves kept --------------------------------------------------------------


def test_without_a_grid_the_model_keeps_only_the_fixed_quadratures_solves():
    cfg = make_config()

    assert _field_cache_size(cfg) == pf.CACHE_SIZE


def test_with_a_grid_the_model_keeps_every_grid_instant_and_its_rate_stencil():
    cfg = make_config(["training.instant_grid=100"])

    assert _field_cache_size(cfg) == pf.CACHE_SIZE + RATE_STENCIL * 101


def _solved_instants(monkeypatch) -> list[float]:
    """Records every instant the model solves the liquid's potentials at."""
    solved: list[float] = []
    solve = BubblePINN._solve_fields

    def recording(self, instants):
        solved.extend(instants)
        return solve(self, instants)

    monkeypatch.setattr(BubblePINN, "_solve_fields", recording)
    return solved


def _train(tmp_path, extra, steps: int):
    cfg, paths = staged_walled_capsule_run(
        tmp_path, [*EXACT, *PHYSICS_STAGE, f"training.steps={steps}", *extra]
    )
    return training.train(cfg, paths)


def test_on_a_grid_no_instant_is_solved_twice_in_a_run(tmp_path, monkeypatch):
    solved = _solved_instants(monkeypatch)

    _train(tmp_path, ["training.instant_grid=16"], steps=8)

    keys = [round(float(instant), 7) for instant in solved]
    assert len(keys) == len(set(keys))


def test_on_a_grid_a_run_stops_solving_once_it_has_met_its_instants(tmp_path, monkeypatch):
    solved = _solved_instants(monkeypatch)

    _train(tmp_path, ["training.instant_grid=16"], steps=4)
    met = len(solved)
    _train(tmp_path, ["training.instant_grid=16"], steps=4)

    # The second call starts a new model, so it solves its instants once more,
    # and no more than the grid holds with the rate's stencil.
    assert len(solved) - met <= pf.CACHE_SIZE + RATE_STENCIL * 17


def test_without_a_grid_every_step_solves_fresh_instants(tmp_path, monkeypatch):
    solved = _solved_instants(monkeypatch)

    _train(tmp_path, [], steps=3)
    after_three = len(solved)
    _train(tmp_path / "longer", [], steps=6)

    assert len(solved) - after_three > after_three


def test_a_run_on_a_grid_trains_and_is_reproducible(tmp_path):
    _, _, state = _train(tmp_path, ["training.instant_grid=16"], steps=3)
    _, _, again = _train(tmp_path / "again", ["training.instant_grid=16"], steps=3)

    last = state["hist"][-1]
    assert all(np.isfinite(v) for v in last.values())
    assert last == again["hist"][-1]
