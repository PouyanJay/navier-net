"""The Young-Laplace jump imposed on average around the front
(``model.laplace_condition=mean``), and its scatter reported in pascals."""

from __future__ import annotations

import pytest
import torch

from naviernet import training
from naviernet.physics.registry import in_view_mean, per_instant_mean

from .conftest import staged_capsule_run

SHARP = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.momentum_closure=brinkman",
    "model.pressure_representation=two_phase",
]


def test_the_mean_is_taken_per_instant():
    values = torch.tensor([[1.0], [3.0], [10.0], [-2.0], [2.0]])
    times = torch.tensor([0.5, 0.5, 1.0, 2.0, 2.0])

    assert torch.equal(per_instant_mean(values, times), torch.tensor([[2.0], [10.0], [0.0]]))


def test_a_jump_error_that_averages_out_costs_nothing_on_average():
    # A pointwise departure with zero mean at each instant is the scatter a
    # uniform vapour pressure cannot remove; the mean condition leaves it.
    residual = torch.tensor([[0.3], [-0.3], [0.1], [-0.1]])
    times = torch.tensor([0.0, 0.0, 1.0, 1.0])

    assert float((per_instant_mean(residual, times) ** 2).mean()) == pytest.approx(0.0)
    assert float((residual**2).mean()) > 0.0


def test_the_pointwise_jump_leaves_out_the_front_past_the_camera_view():
    # A nose that has left the view has a guessed shape: its samples carry no weight.
    squared = torch.tensor([[1.0], [3.0], [100.0]])
    x = torch.tensor([[0.5], [1.0], [1.2]])

    assert float(in_view_mean(squared, x, view_x_max=1.0)) == pytest.approx(2.0)
    assert float(in_view_mean(squared, x, view_x_max=None)) == pytest.approx(104.0 / 3.0)


def test_a_front_wholly_past_the_view_costs_nothing():
    squared = torch.tensor([[4.0], [9.0]])
    x = torch.tensor([[2.0], [3.0]])

    assert float(in_view_mean(squared, x, view_x_max=1.0)) == 0.0


def test_a_mean_jump_run_trains_and_reports_the_scatter_in_pascals(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            *SHARP,
            "model.laplace_condition=mean",
            "training.stage_b_warmup_steps=1",
            "training.steps=3",
            "training.log_every=1",
        ],
    )
    model, data, state = training.train(cfg, paths)
    from naviernet.audit import front_conditions
    from naviernet.physics.groups import compute_groups

    record, _ = front_conditions(model, compute_groups(cfg), float(data.t[2]), pixel=0.01)

    assert torch.isfinite(torch.tensor(state["hist"][-1]["laplace"]))
    assert {"vapour_pressure_Pa", "laplace_mean_Pa", "laplace_scatter_Pa"} <= set(record)
    assert record["laplace_scatter_Pa"] >= 0.0
    assert set(record["laplace_scatter_Pa_by_segment"]) <= {"upper", "lower", "root", "nose"}
