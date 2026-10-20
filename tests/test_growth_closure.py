"""Growth closure: the vapour the source makes, integrated over the domain,
against the measured growth of the bubble's area at every event frame; it
trains the footprint's wetting and never the temperature."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from naviernet import training
from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BubblePINN
from naviernet.physics import growth_closure
from naviernet.training import _geometry_priors, bind_cover_times

from .conftest import staged_capsule_run

CLOSURE = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.thermal_model=sharp",
    "model.source_mode=thermal",
    "model.footprint_evaporation=true",
    "experiment.T_inlet_C=56.6",
    "experiment.T_init_C=58.0",
    "experiment.footprint_wetting=microlayer",
    "experiment.microlayer_mode=fit",
    "training.growth_closure_weight=1.0",
]


def test_the_measured_growth_is_central_inside_the_event_and_one_sided_at_its_ends():
    alpha = np.zeros((3, 1, 10))
    for row, width in enumerate((2, 4, 8)):
        alpha[row, 0, :width] = 1.0
    data = SimpleNamespace(
        alpha=alpha,
        valid=np.ones_like(alpha),
        x=np.arange(10.0),
        y=np.arange(2.0),
        t=np.array([0.0, 1.0, 2.0]),
        n_event=3,
    )

    rates = [growth_closure.measured_growth_rate(data, r) for r in range(3)]

    assert rates == pytest.approx([2.0, 3.0, 4.0])


def _model(tmp_path, overrides=()):
    cfg, paths = staged_capsule_run(tmp_path, [*CLOSURE, *overrides])
    data = BubbleDataset(cfg, paths)
    torch.manual_seed(0)
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    bind_cover_times(cfg, model, data)
    return data, model


def test_the_footprint_total_is_the_source_summed_over_the_sampled_floor(tmp_path):
    data, model = _model(tmp_path)
    plan = growth_closure.build_plan(data, stride=2, device="cpu")
    instant = plan.times[3]
    x = torch.cat([plan.points, instant.expand(len(plan.points), 1)], dim=1)

    footprint, _ = growth_closure.source_totals(model, plan)

    assert float(footprint[3]) == pytest.approx(
        float(model.footprint_source(x).sum()) * plan.cell, rel=1e-5
    )


def test_the_closure_trains_the_film_and_never_the_temperature(tmp_path):
    data, model = _model(tmp_path)
    plan = growth_closure.build_plan(data, stride=2, device="cpu")

    loss, record = growth_closure.growth_closure_loss(model, plan)
    loss.backward()

    assert model._log_microlayer_raw.grad is not None
    assert float(model._log_microlayer_raw.grad.abs()) > 0.0
    assert all(p.grad is None for p in model.nets["T"].parameters())
    assert record["growth_closure"] == pytest.approx(float(loss))


def test_the_closure_needs_a_thermal_source_with_a_footprint(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*CLOSURE, "model.source_mode=free"])

    with pytest.raises(ValueError, match="growth_closure_weight"):
        training.train(cfg, paths)


def test_a_closure_run_logs_the_gap_and_the_film(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            *CLOSURE,
            "training.stage_b_warmup_steps=1",
            "training.steps=2",
            "training.log_every=1",
            "training.growth_closure_stride=2",
            "training.collocation_time_bins=2",
        ],
    )

    _, _, state = training.train(cfg, paths)

    last = state["hist"][-1]
    assert "growth_closure" in last and last["growth_closure"] >= 0.0
    assert last["film_um"] > 0.0


def test_the_outline_part_is_the_same_value_with_or_without_its_gradient(tmp_path):
    data, model = _model(tmp_path)
    plan = growth_closure.build_plan(data, stride=2, device="cpu")

    _, detached = growth_closure.source_totals(model, plan)
    _, carried = growth_closure.source_totals(model, plan, outline_gradient=True)

    assert torch.equal(detached, carried.detach())
    assert not detached.requires_grad
    assert all(p.requires_grad for p in model.nets["T"].parameters()), "flags restored"


def test_the_switch_lets_the_closure_train_the_temperature(tmp_path):
    data, model = _model(tmp_path)
    plan = growth_closure.build_plan(data, stride=2, device="cpu")

    loss, _ = growth_closure.growth_closure_loss(model, plan, outline_gradient=True)
    loss.backward()

    assert any(p.grad is not None for p in model.nets["T"].parameters())
