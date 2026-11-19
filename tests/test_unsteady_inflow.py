"""An inlet flow that changes in time (``model.unsteady_inflow``): the curve the
run trains, the liquid flow built on it, and the terms that read the inflow."""

from __future__ import annotations

import math

import pytest
import torch

from naviernet import training
from naviernet.data.dataset import BubbleDataset
from naviernet.models import potential_flow as pf
from naviernet.models.inflow import InflowCurve
from naviernet.models.pinn import BubblePINN, _arclength_weights
from naviernet.physics import mass_balance
from naviernet.physics.groups import compute_groups
from naviernet.physics.residuals import gradients

from .conftest import staged_walled_capsule_run

EXACT = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.walls_at=physical",
    "model.hard_walls=true",
    "model.mirror_symmetry=true",
    "model.exact_continuity=true",
    "model.exact_continuity_body_sources=32",
    "model.exact_continuity_cap_sources=8",
]
UNSTEADY = [*EXACT, "model.unsteady_inflow=true"]


def _curve(**overrides) -> InflowCurve:
    settings = dict(t_min=-1.0, t_max=3.0, intervals=8, lowest=0.0, highest=2.0)
    return InflowCurve(**{**settings, **overrides})


def _times(curve: InflowCurve, n: int = 41) -> torch.Tensor:
    return torch.linspace(curve.t_min, curve.t_max, n).reshape(-1, 1)


def _model(tmp_path, overrides=UNSTEADY):
    cfg, paths = staged_walled_capsule_run(tmp_path, list(overrides))
    data = BubbleDataset(cfg, paths)
    torch.manual_seed(0)
    return cfg, data, BubblePINN(cfg, geometry=training._geometry_priors(cfg, data))


def _raise_inflow(model, by: float = 0.6) -> None:
    """Every coefficient moved alike: the whole curve lifted off the set flow."""
    with torch.no_grad():
        model.inflow_curve.coefficients.fill_(by)


def _tilt_inflow(model) -> None:
    """Coefficients rising through the window: an inflow that grows in time."""
    with torch.no_grad():
        n = len(model.inflow_curve.coefficients)
        model.inflow_curve.coefficients.copy_(torch.linspace(-0.5, 0.5, n))


def _liquid_points(data, model, t: float, n: int = 400) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    lo, hi = model.wall_span
    xy = torch.rand(n, 2, generator=g) * torch.tensor([float(data.x.max()), hi - lo])
    xy[:, 1] += lo
    points = torch.cat([xy, torch.full((n, 1), t)], dim=1)
    return points[model.liquid_side(points)].clone().requires_grad_(True)


def _section_flux(model, x: float, t: float, n: int = 4001) -> float:
    lo, hi = model.wall_span
    ys = torch.linspace(lo, hi, n)
    section = torch.stack([torch.full_like(ys, x), ys, torch.full_like(ys, t)], dim=1)
    u, _ = model.velocity(section)
    liquid = model.liquid_side(section)
    return float((u[:, 0] * liquid).sum()) * (hi - lo) / (n - 1)


def _set_flux(model) -> float:
    lo, hi = model.wall_span
    return float(model._inflow_groups["u_inlet_star"]) * (hi - lo)


# --- the curve -----------------------------------------------------------------


def test_the_curve_starts_at_the_set_flow():
    curve = _curve()

    assert torch.allclose(curve(_times(curve)), torch.ones(41, 1), atol=1e-6)


def test_the_curve_stays_inside_its_band_whatever_it_learns():
    curve = _curve(lowest=0.2, highest=1.8)
    with torch.no_grad():
        curve.coefficients.copy_(torch.linspace(-50.0, 50.0, len(curve.coefficients)))

    factor = curve(_times(curve)).detach()

    rounding = 1e-6
    assert float(factor.min()) >= 0.2 - rounding and float(factor.max()) <= 1.8 + rounding
    assert float(factor.min()) < 0.25 and float(factor.max()) > 1.75, "it can reach both ends"


def test_one_coefficient_moves_the_curve_only_near_its_own_time():
    curve = _curve()
    with torch.no_grad():
        curve.coefficients[1] = 1.0
    times = _times(curve, 81)

    moved = (curve(times) - 1.0).abs()[:, 0] > 1e-6

    assert bool(moved[:20].any()), "the early part moved"
    assert not bool(moved[40:].any()), "the late part did not"


def test_a_flat_curve_averages_the_same_over_every_span():
    curve = _curve()
    with torch.no_grad():
        curve.coefficients.fill_(0.3)

    averages = curve.running_mean(1.5).detach()

    assert len(averages) > 10, "one average per start inside the window"
    assert torch.allclose(averages, curve.mean().detach().expand_as(averages), atol=1e-5)


def test_a_rising_curve_averages_more_over_the_later_spans():
    curve = _curve()
    with torch.no_grad():
        curve.coefficients.copy_(torch.linspace(-0.3, 0.3, len(curve.coefficients)))

    averages = curve.running_mean(1.5).detach()

    assert bool((averages.diff() > 0).all())
    assert float(averages[0]) < float(curve.mean().detach()) < float(averages[-1])


def test_a_span_average_is_the_curve_integrated_over_that_span():
    curve = _curve()
    with torch.no_grad():
        curve.coefficients.copy_(torch.sin(torch.arange(len(curve.coefficients)) * 1.3))
    span = 1.5
    fine = torch.linspace(curve.t_min, curve.t_min + span, 20001).reshape(-1, 1)

    first = float(curve.running_mean(span)[0].detach())

    expected = float(torch.trapezoid(curve(fine)[:, 0], fine[:, 0]).detach()) / span
    assert first == pytest.approx(expected, abs=2e-3)


def test_a_span_as_long_as_the_window_gives_the_one_window_average():
    curve = _curve()
    with torch.no_grad():
        curve.coefficients.copy_(torch.linspace(-0.3, 0.3, len(curve.coefficients)))

    averages = curve.running_mean(curve.t_max - curve.t_min)

    assert averages.shape == (1,)
    assert float(averages[0].detach()) == pytest.approx(float(curve.mean().detach()))


def test_a_span_must_have_a_length():
    with pytest.raises(ValueError, match="span"):
        _curve().running_mean(0.0)


def _largest_slope_jump(curve: InflowCurve, n: int) -> float:
    t = _times(curve, n).requires_grad_(True)
    (slope,) = torch.autograd.grad(curve(t).sum(), t)
    return float(slope.diff(dim=0).abs().max())


def test_the_curve_has_no_kink_in_time():
    curve = _curve()
    with torch.no_grad():
        curve.coefficients.copy_(torch.sin(torch.arange(len(curve.coefficients)) * 1.3))

    coarse, fine = _largest_slope_jump(curve, 401), _largest_slope_jump(curve, 1601)

    # A smooth slope's jump between neighbours shrinks with their spacing; a
    # kink's does not.
    assert fine < 0.35 * coarse


def test_times_outside_the_window_read_its_ends():
    curve = _curve()
    with torch.no_grad():
        curve.coefficients.copy_(torch.linspace(-1.0, 1.0, len(curve.coefficients)))
    ends = curve(torch.tensor([[curve.t_min], [curve.t_max]]))

    beyond = curve(torch.tensor([[curve.t_min - 0.5], [curve.t_max + 0.5]]))

    assert torch.allclose(beyond, ends)


def test_a_set_flow_at_the_bands_edge_starts_just_inside_it():
    capped = _curve(lowest=0.0, highest=1.0)

    factor = capped(_times(capped)).detach()

    assert float(factor.max()) < 1.0
    assert float(factor.min()) > 0.9


def test_the_curves_average_over_the_window():
    curve = _curve()
    assert float(curve.mean()) == pytest.approx(1.0, abs=1e-6)

    with torch.no_grad():
        curve.coefficients.fill_(0.4)

    assert float(curve.mean()) == pytest.approx(float(curve(_times(curve)).mean()), abs=1e-4)
    assert float(curve.mean()) > 1.05


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(lowest=1.5, highest=1.0), "below its highest"),
        (dict(lowest=1.2, highest=2.0), "set flow"),
        (dict(intervals=0), "at least one interval"),
        (dict(t_min=1.0, t_max=1.0), "time window"),
    ],
)
def test_a_band_that_cannot_hold_the_set_flow_is_refused(overrides, message):
    with pytest.raises(ValueError, match=message):
        _curve(**overrides)


# --- the liquid's flow -----------------------------------------------------------


def test_at_the_set_flow_the_liquid_moves_as_under_a_steady_inlet(tmp_path):
    _, data, steady = _model(tmp_path / "steady", EXACT)
    _, _, unsteady = _model(tmp_path / "unsteady")
    x = _liquid_points(data, steady, float(data.t[3])).detach()

    (u, v), (u_steady, v_steady) = unsteady.velocity(x), steady.velocity(x)

    assert torch.allclose(u, u_steady, atol=1e-6) and torch.allclose(v, v_steady, atol=1e-6)


def test_each_section_carries_that_instants_inflow_plus_the_growth_behind_it(tmp_path):
    _, data, model = _model(tmp_path)
    _raise_inflow(model)
    t = float(data.t[3])
    factor = float(model.inflow_factor(torch.tensor([[t]])))
    front = model.front(torch.tensor([[t]], requires_grad=True), 256, 64)
    xf = front.points[:, 0].detach()
    weights = _arclength_weights(front.points.detach(), 256, 64)[:, 0]
    speed = front.normal_speed.detach()[:, 0]
    q_in = factor * _set_flux(model)
    inside = float(xf.mean())

    assert factor > 1.2, "the inflow is well above the set flow"
    assert _section_flux(model, 0.05, t) == pytest.approx(q_in, rel=1e-3)
    growth = float((speed * weights)[xf < inside].sum())
    assert _section_flux(model, inside, t) == pytest.approx(q_in + growth, abs=0.03 * q_in)


def test_the_liquid_still_moves_with_the_front_and_keeps_its_volume(tmp_path):
    _, data, model = _model(tmp_path)
    _raise_inflow(model)
    t = float(data.t[3])
    front = model.front(torch.tensor([[t]], requires_grad=True), 128, 32)
    x = _liquid_points(data, model, t)

    u_f, v_f = model.velocity(front.points.detach())
    normal = front.normal.detach()
    miss = u_f * normal[:, :1] + v_f * normal[:, 1:2] - front.normal_speed.detach()
    u, v = model.velocity(x)
    u_x, u_y, _ = gradients(u, x)
    v_x, v_y, _ = gradients(v, x)

    assert float(miss.pow(2).mean().sqrt()) < 0.05 * float(front.normal_speed.abs().max())
    scale = float(torch.cat([u_x, u_y, v_x, v_y]).abs().max())
    assert float((u_x + v_y).abs().max()) < 1e-5 * max(scale, 1.0)


def test_the_liquids_rate_of_change_carries_the_inflows(tmp_path):
    _, data, model = _model(tmp_path)
    _tilt_inflow(model)
    t = float(data.t[3])
    lo, hi = model.wall_span
    inlet = torch.tensor([[pf.INLET_X, 0.5 * (lo + hi), t]], requires_grad=True)
    instant = torch.tensor([[t]], requires_grad=True)
    (factor_rate,) = torch.autograd.grad(model.inflow_factor(instant).sum(), instant)

    u, v = model.velocity(inlet)
    _, _, u_t = gradients(u, inlet)
    _, _, v_t = gradients(v, inlet)
    u_rate, _ = model.velocity_rate(inlet, u_t, v_t)

    # At the inlet plane the potentials pass nothing, so the liquid there is
    # the inflow itself.
    expected = float(factor_rate) * float(model._inflow(inlet[:, 1:2]))
    assert abs(float(factor_rate)) > 1e-3
    assert float(u_rate) == pytest.approx(expected, rel=2e-2)


def test_the_flow_terms_can_move_the_curve(tmp_path):
    _, data, model = _model(tmp_path)
    x = _liquid_points(data, model, float(data.t[3]))

    u, v = model.velocity(x)
    (u.pow(2).sum() + v.pow(2).sum()).backward()

    grad = model.inflow_curve.coefficients.grad
    assert grad is not None and float(grad.abs().max()) > 0.0


def test_a_steady_inlet_reports_the_set_flow_at_every_instant(tmp_path):
    _, _, model = _model(tmp_path, EXACT)

    factor = model.inflow_factor(torch.tensor([[0.0], [1.0]]))

    assert model.inflow_curve is None
    assert torch.equal(factor, torch.ones(2, 1))


# --- the terms that read the inflow ---------------------------------------------


def test_the_mass_balance_reads_each_instants_inflow(tmp_path):
    cfg, data, model = _model(tmp_path)
    _raise_inflow(model)
    plan = mass_balance.build_plan(
        data.domain, model.wall_span, compute_groups(cfg), cfg.training, "cpu"
    )

    defect = mass_balance.axial_flux_defect(model, plan)

    assert float(defect[:, 0].abs().max()) < 0.02, "the first station carries the inflow"


def test_the_inlet_target_follows_the_curve(tmp_path):
    cfg, data, model = _model(tmp_path)
    _tilt_inflow(model)
    inlet = torch.tensor(
        [[pf.INLET_X, 0.3, float(data.t[1])], [pf.INLET_X, 0.6, float(data.t[5])]]
    )
    groups = compute_groups(cfg)

    target = training._inlet_velocity(cfg, groups, data, inlet, groups["u_inlet_star"], model)

    expected = groups["u_inlet_star"] * model.inflow_factor(inlet[:, 2:3])
    assert torch.allclose(target, expected)
    assert float(target[0]) != pytest.approx(float(target[1]))


def test_the_mean_prior_is_zero_at_the_set_flow_and_grows_away_from_it(tmp_path):
    _, _, model = _model(tmp_path)

    at_set_flow = float(training._inflow_mean_penalty(model))
    _raise_inflow(model)
    lifted = float(training._inflow_mean_penalty(model))

    assert at_set_flow == pytest.approx(0.0, abs=1e-10)
    assert lifted > 0.05


def test_the_mean_prior_leaves_an_average_inside_its_tolerance_alone(tmp_path):
    _, _, model = _model(tmp_path)
    _raise_inflow(model, by=0.05)
    miss = abs(float(model.inflow_curve.mean()) - 1.0)

    inside = float(training._inflow_mean_penalty(model, tolerance=2.0 * miss))
    outside = float(training._inflow_mean_penalty(model, tolerance=0.5 * miss))

    assert miss > 0.01
    assert inside == 0.0
    assert outside == pytest.approx((0.5 * miss) ** 2, rel=1e-3)


def _borrow_early_for_late(model) -> None:
    """Low through the first half of the window and high through the second, by
    the same amount: the window's average stays near the set flow."""
    with torch.no_grad():
        n = len(model.inflow_curve.coefficients)
        model.inflow_curve.coefficients.copy_(torch.linspace(-0.1, 0.1, n).sign() * 0.1)


def test_the_cycle_prior_counts_what_the_window_average_hides(tmp_path):
    _, _, model = _model(tmp_path)
    curve = model.inflow_curve
    _borrow_early_for_late(model)
    half_window = 0.5 * (curve.t_max - curve.t_min)
    tolerance = 0.05

    over_the_window = float(training._inflow_mean_penalty(model, tolerance))
    over_each_cycle = float(training._inflow_mean_penalty(model, tolerance, span=half_window))

    assert abs(float(curve.mean()) - 1.0) < tolerance, "the window's average looks fine"
    assert over_the_window == 0.0
    assert over_each_cycle > 1e-3


# --- configuration and training --------------------------------------------------


def _training_run(tmp_path, extra=()):
    return staged_walled_capsule_run(
        tmp_path,
        [
            *UNSTEADY,
            "model.outlet_condition=zero_gradient",
            "training.stage_b_warmup_steps=1",
            "training.freeze_geometry=true",
            "training.edge_transport_after_freeze=false",
            "training.steps=3",
            "training.log_every=1",
            *extra,
        ],
    )


def test_it_needs_the_exact_liquid_flow(tmp_path):
    with pytest.raises(ValueError, match="model.exact_continuity"):
        _model(tmp_path, [*UNSTEADY, "model.exact_continuity=false"])


@pytest.mark.parametrize(
    "setting", ["inflow_mean_weight", "inflow_mean_tolerance", "inflow_mean_period_ms"]
)
def test_a_negative_mean_setting_is_refused(tmp_path, setting):
    cfg, paths = _training_run(tmp_path, [f"training.{setting}=-1.0"])

    with pytest.raises(ValueError, match=setting):
        training.train(cfg, paths)


def test_a_run_trains_with_finite_losses_and_records_its_inflow(tmp_path):
    cfg, paths = _training_run(tmp_path, ["training.inflow_mean_weight=1.0"])

    _, _, state = training.train(cfg, paths)

    last = state["hist"][-1]
    losses = [float(v) for v in last.values() if isinstance(v, int | float)]
    assert losses and all(math.isfinite(v) for v in losses)
    set_flow = float(cfg.experiment.flow_rate_mL_hr)
    assert last["inflow_min_mL_hr"] <= last["inflow_mean_mL_hr"] <= last["inflow_max_mL_hr"]
    assert last["inflow_mean_mL_hr"] == pytest.approx(set_flow, rel=0.05)


def test_a_run_holding_each_cycle_records_its_cycle_averages(tmp_path):
    cfg, paths = _training_run(
        tmp_path, ["training.inflow_mean_weight=1.0", "training.inflow_mean_period_ms=3.0"]
    )

    _, _, state = training.train(cfg, paths)

    last = state["hist"][-1]
    assert last["inflow_min_mL_hr"] <= last["inflow_cycle_low_mL_hr"]
    assert last["inflow_cycle_low_mL_hr"] <= last["inflow_cycle_high_mL_hr"]
    assert last["inflow_cycle_high_mL_hr"] <= last["inflow_max_mL_hr"]


def test_a_run_holding_the_window_average_records_no_cycle_averages(tmp_path):
    cfg, paths = _training_run(tmp_path, ["training.inflow_mean_weight=1.0"])

    _, _, state = training.train(cfg, paths)

    assert "inflow_cycle_low_mL_hr" not in state["hist"][-1]


def test_a_steady_run_records_no_inflow(tmp_path):
    cfg, paths = _training_run(tmp_path, ["model.unsteady_inflow=false"])

    _, _, state = training.train(cfg, paths)

    assert "inflow_mean_mL_hr" not in state["hist"][-1]


def test_a_warm_up_saved_under_a_steady_inlet_resumes_whole(tmp_path):
    cfg_steady, paths = _training_run(
        tmp_path, ["model.unsteady_inflow=false", "training.steps=1"]
    )
    training.train(cfg_steady, paths)
    saved = torch.load(paths.checkpoint, weights_only=False)["state"]
    cfg_unsteady, _ = _training_run(tmp_path, ["training.steps=2"])

    _, _, state = training.train(cfg_unsteady, paths)

    assert state["done"] == saved["done"] + 2
    assert set(saved) <= set(state), "the warm-up's whole state carried forward"


def test_a_steady_run_past_its_warm_up_continues_with_its_optimiser(tmp_path):
    cfg_steady, paths = _training_run(
        tmp_path, ["model.unsteady_inflow=false", "training.steps=3"]
    )
    training.train(cfg_steady, paths)
    saved = torch.load(paths.checkpoint, weights_only=False)
    cfg_unsteady, _ = _training_run(tmp_path, ["training.steps=1"])
    model = BubblePINN(
        cfg_unsteady,
        geometry=training._geometry_priors(cfg_unsteady, BubbleDataset(cfg_unsteady, paths)),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.load_state_dict(saved["model"], strict=False)

    carried = training._carry_optimiser(
        opt, model, saved["opt"], fresh=["inflow_curve.coefficients"]
    )

    assert carried
    first = next(p for n, p in model.named_parameters() if not n.startswith("inflow_curve."))
    assert (
        opt.state[first]["step"]
        == saved["opt"]["state"][saved["opt"]["param_groups"][0]["params"][0]]["step"]
    )
    assert model.inflow_curve.coefficients not in opt.state, "the curve starts fresh"

    _, _, state = training.train(cfg_unsteady, paths)
    assert state["done"] == saved["state"]["done"] + 1


def test_a_curve_trained_past_the_warm_up_is_not_dropped(tmp_path):
    cfg_unsteady, paths = _training_run(tmp_path, ["training.steps=2"])
    training.train(cfg_unsteady, paths)
    cfg_steady, _ = _training_run(tmp_path, ["model.unsteady_inflow=false", "training.steps=1"])

    with pytest.raises(RuntimeError, match="unsteady_inflow past its warm-up"):
        training.train(cfg_steady, paths)
