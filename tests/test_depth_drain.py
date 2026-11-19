"""The depth's share of the edge rule (``model.depth_drain``): the root's law,
the smallest drain that keeps the rebuilt bubble near the real outline, and the
loss, the trainer and the audit that read the split."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from naviernet import training
from naviernet.audit import depth_drain_record
from naviernet.data.dataset import BubbleDataset
from naviernet.models.depth_curve import DepthRootCurve
from naviernet.models.measured_outline import MeasuredOutline
from naviernet.models.pinn import BubblePINN
from naviernet.physics import depth_drain, registry
from naviernet.physics.groups import compute_groups
from naviernet.physics.residuals import (
    front_arc_lengths,
    laplace_jump_residual,
    root_cap_mask,
)

from .conftest import staged_capsule_run

SHARP = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.momentum_closure=brinkman",
    "model.pressure_representation=two_phase",
]
DRAIN = [*SHARP, "model.laplace_weighting=length", "model.depth_drain=true"]

# The synthetic outline: an ellipse 3 long and 1 wide about the axis, in a
# pressure scale where one unit of pressure bends the edge by one unit.
AXIS = 0.5
HALF_LENGTH, HALF_WIDTH = 1.5, 0.5
STIFFNESS = 1.0


def _outline() -> MeasuredOutline:
    q = 2.0 * math.pi * np.arange(128) / 128
    curve = np.stack([2.0 + HALF_LENGTH * np.cos(q), AXIS + HALF_WIDTH * np.sin(q)], axis=1)
    stations = np.stack([curve, curve + [0.2, 0.0], curve + [0.4, 0.0]])
    return MeasuredOutline(
        np.array([-1.0, 0.0, 1.0]), stations, AXIS, nose_cap_q=0.2, root_cap_q=0.2
    )


def _front(times=(0.0,), n_body: int = 96, n_cap: int = 33):
    return _outline().front(torch.tensor(times, dtype=torch.float32), n_body, n_cap)


def _edges(**kwargs) -> depth_drain.HalfEdges:
    return depth_drain.half_edges(_front(**kwargs), starts_at=0.0)


def _along(edges: depth_drain.HalfEdges) -> torch.Tensor:
    """Each sample's distance along its half from the nose tip, ``(C, n)``."""
    start = torch.zeros(len(edges.steps), 1, dtype=torch.float64)
    return torch.cat([start, edges.steps.cumsum(dim=1)], dim=1)


def _bump(edges: depth_drain.HalfEdges, size: float) -> torch.Tensor:
    """A mismatch that changes sign once along each half: no uniform pressure
    shift can take it."""
    along = _along(edges)
    return size * torch.sin(2.0 * math.pi * along / along[:, -1:])


# --- the root's law ---------------------------------------------------------------


def test_the_root_law_starts_flat_and_ends_at_zero_at_its_reach():
    law = DepthRootCurve(reach=0.5, knots=10)
    law.knot_values.fill_(1.0)
    distance = torch.tensor([[0.0], [0.5], [0.7]])

    fresh = DepthRootCurve(reach=0.5, knots=10)(distance)
    fitted = law(distance)

    assert torch.equal(fresh, torch.zeros(3, 1))
    assert float(fitted[0]) > 0.0
    assert torch.equal(fitted[1:], torch.zeros(2, 1)), "zero at the reach and beyond"


def test_the_root_law_runs_straight_between_its_values():
    law = DepthRootCurve(reach=1.0, knots=4)
    law.knot_values.copy_(torch.tensor([4.0, 2.0, 0.0, 0.0]))
    at_knots = law(torch.tensor([[0.0], [0.25]]))

    midway = law(torch.tensor([[0.125]]))

    assert float(midway) == pytest.approx(float(at_knots.mean()), rel=1e-6)


def test_the_fit_recovers_a_law_of_its_own_kind():
    source = DepthRootCurve(reach=0.5, knots=10)
    source.knot_values.copy_(torch.linspace(-0.5, 0.0, 10))
    distance = torch.linspace(0.0, 0.8, 400).reshape(-1, 1)
    target = source(distance)
    law = DepthRootCurve(reach=0.5, knots=10)

    law.fit(distance, target, torch.ones_like(distance))

    assert torch.allclose(law(distance), target, atol=1e-4)


def test_the_root_law_is_saved_with_the_run_and_never_trained():
    law = DepthRootCurve(reach=0.5, knots=10)

    assert list(law.parameters()) == []
    assert "knot_values" in law.state_dict()


def test_the_first_image_time_must_be_bound():
    law = DepthRootCurve(reach=0.5, knots=10)

    with pytest.raises(RuntimeError, match="bind_first_image"):
        _ = law.starts_at


# --- the half outlines ------------------------------------------------------------


def test_each_half_runs_from_the_nose_tip_to_the_root_tip():
    edges = _edges()

    x = edges.points[..., 0]

    assert edges.order.shape == (2, 96 + 2 * 17)
    assert torch.allclose(
        x[:, 0], torch.full((2,), 2.2 + HALF_LENGTH, dtype=x.dtype), atol=1e-5
    )
    assert torch.allclose(
        x[:, -1], torch.full((2,), 2.2 - HALF_LENGTH, dtype=x.dtype), atol=1e-5
    )
    assert bool((x[:, 1:] < x[:, :-1]).all()), "the half is walked one way"


def test_the_lower_half_is_mirrored_onto_the_upper():
    edges = _edges()

    upper, lower = edges.points
    mirrored_about_axis = torch.stack([lower[:, 0], lower[:, 1] + 2.0 * AXIS], dim=1)

    assert torch.allclose(upper, mirrored_about_axis, atol=1e-5)
    assert torch.allclose(edges.normal[0], edges.normal[1], atol=1e-5)


def test_instants_before_the_first_image_are_left_out():
    front = _front(times=(-0.5, 0.0, 0.5))

    edges = depth_drain.half_edges(front, starts_at=0.0)

    assert len(edges.order) == 4, "two halves at each of the two imaged instants"
    assert bool((front.points[edges.order.reshape(-1), 2] >= 0.0).all())
    assert depth_drain.half_edges(front, starts_at=2.0) is None


def test_the_tail_lies_behind_the_widest_point():
    edges = _edges()

    distance, on_tail = depth_drain.root_tail(edges)

    x = edges.points[..., 0]
    assert float(distance.min()) == pytest.approx(0.0, abs=1e-9)
    assert bool((x[on_tail] < 2.2).all()) and bool((x[~on_tail] >= 2.2 - 1e-3).all())


# --- the smallest drain -----------------------------------------------------------


def test_the_running_integral_is_exact_for_a_straight_line():
    edges = _edges()
    along = _along(edges)

    summed = (depth_drain._running_integral(edges.steps) @ along.unsqueeze(2)).squeeze(2)

    assert torch.allclose(summed, 0.5 * along**2, atol=1e-12)


def _walked_tip(edges: depth_drain.HalfEdges, bend: torch.Tensor) -> torch.Tensor:
    """Where a walk of the upper half ends when its bend at every sample is
    ``bend`` ``(n,)``: the direction summed from the bend, the position from
    the direction."""
    steps = edges.steps[0]
    nx, ny = edges.normal[0, 0]
    heading = torch.atan2(nx, -ny) + torch.cat(
        [torch.zeros(1, dtype=bend.dtype), (0.5 * (bend[1:] + bend[:-1]) * steps).cumsum(0)]
    )
    middle = 0.5 * (heading[1:] + heading[:-1])
    return torch.stack([(torch.cos(middle) * steps).sum(), (torch.sin(middle) * steps).sum()])


def test_the_linear_rebuild_matches_a_walk_of_the_edge_for_a_small_mismatch():
    front = _front()
    edges = depth_drain.half_edges(front, starts_at=0.0)
    bend = front.kappa_par.double()[edges.order[0], 0]
    along = _along(edges)
    # Lopsided along the half, so the tip moves both along and across the channel.
    mismatch = 1e-3 * torch.sin(3.0 * along / along[:, -1:])

    linear = depth_drain._rebuild(edges, STIFFNESS).tip_of_error[0] @ mismatch[0]
    walked = _walked_tip(edges, bend + mismatch[0] / STIFFNESS) - _walked_tip(edges, bend)

    # 2%, and the walk's own second-order terms (the mismatch squared).
    assert float(linear.abs().min()) > 1e-4
    assert torch.allclose(linear, walked, rtol=0.02, atol=5e-6)


def test_a_uniform_mismatch_drains_nothing():
    # A uniform pressure shift is the vapour pressure's to take: with both
    # tips held it leaves the edge where it is.
    edges = _edges()
    uniform = torch.full(edges.cells.shape, 0.3, dtype=torch.float64)

    solved = depth_drain.smallest_drain(uniform, edges, STIFFNESS, tolerance=1e-6)

    assert float(solved.move_before.max()) < 1e-9
    assert torch.equal(solved.drain, torch.zeros_like(uniform))


def test_a_mismatch_that_keeps_the_shape_drains_nothing():
    edges = _edges()
    mismatch = _bump(edges, size=0.05)
    moved = depth_drain.smallest_drain(mismatch, edges, STIFFNESS, tolerance=1e9).move_before

    solved = depth_drain.smallest_drain(
        mismatch, edges, STIFFNESS, tolerance=2.0 * float(moved.max())
    )

    assert torch.equal(solved.drain, torch.zeros_like(mismatch))
    assert torch.equal(solved.move_after, solved.move_before)


def test_the_drain_brings_the_rebuilt_edge_to_the_tolerance_and_no_further():
    edges = _edges()
    mismatch = _bump(edges, size=0.5)
    moved = float(depth_drain.smallest_drain(mismatch, edges, STIFFNESS, 1e9).move_before.max())
    tolerance = 0.25 * moved

    solved = depth_drain.smallest_drain(mismatch, edges, STIFFNESS, tolerance)

    assert bool((solved.move_after <= tolerance).all())
    assert float(solved.move_after.min()) == pytest.approx(tolerance, rel=1e-3)


def test_the_drain_is_less_than_the_whole_mismatch():
    edges = _edges()
    mismatch = _bump(edges, size=0.5)
    moved = float(depth_drain.smallest_drain(mismatch, edges, STIFFNESS, 1e9).move_before.max())

    solved = depth_drain.smallest_drain(mismatch, edges, STIFFNESS, 0.25 * moved)

    drained = (edges.cells * solved.drain**2).sum()
    assert 0.0 < float(drained) < float((edges.cells * mismatch**2).sum())


def test_a_tighter_tolerance_drains_more():
    edges = _edges()
    mismatch = _bump(edges, size=0.5)
    moved = float(depth_drain.smallest_drain(mismatch, edges, STIFFNESS, 1e9).move_before.max())

    def drained(share: float) -> float:
        solved = depth_drain.smallest_drain(mismatch, edges, STIFFNESS, share * moved)
        return float((edges.cells * solved.drain**2).sum())

    assert drained(0.5) < drained(0.25) < drained(0.1)


def test_no_drain_is_placed_where_it_is_barred():
    edges = _edges()
    mismatch = _bump(edges, size=0.5)
    moved = float(depth_drain.smallest_drain(mismatch, edges, STIFFNESS, 1e9).move_before.max())
    ahead_of_the_middle = edges.points[..., 0] > 2.2

    solved = depth_drain.smallest_drain(
        mismatch, edges, STIFFNESS, 0.25 * moved, open_to_drain=~ahead_of_the_middle
    )

    assert float(solved.drain[ahead_of_the_middle].abs().max()) == 0.0
    assert float(solved.drain[~ahead_of_the_middle].abs().max()) > 0.0
    assert bool((solved.move_after <= 0.25 * moved * (1.0 + 1e-6)).all())


def _carried_sizes(mismatch: torch.Tensor, edges, tolerance: float) -> torch.Tensor:
    """Each half's drain size with the drain carried on ``mismatch``'s
    gradient, the way the split carries it on the pressure step's."""
    drain = depth_drain.smallest_drain(mismatch.detach(), edges, STIFFNESS, tolerance).drain
    return depth_drain.drain_size(drain + mismatch - mismatch.detach(), edges.cells)


def test_the_drain_size_follows_the_mismatch_the_way_its_gradient_says():
    # The gradient is not taken through the solve: it is the multiplier rule,
    # checked here against the solve itself on a nudged mismatch.
    edges = _edges()
    mismatch = _bump(edges, size=0.5).requires_grad_(True)
    moved = float(
        depth_drain.smallest_drain(mismatch.detach(), edges, STIFFNESS, 1e9).move_before.max()
    )
    tolerance = 0.25 * moved
    along = _along(edges)
    nudge = 1e-3 * torch.cos(5.0 * along / along[:, -1:])

    gradient = torch.autograd.grad(_carried_sizes(mismatch, edges, tolerance).sum(), mismatch)[
        0
    ]
    predicted = (gradient * nudge).sum(dim=1)
    with torch.no_grad():
        ahead = _carried_sizes(mismatch + nudge, edges, tolerance)
        behind = _carried_sizes(mismatch - nudge, edges, tolerance)

    assert float(predicted.abs().min()) > 1e-6
    assert torch.allclose(predicted, 0.5 * (ahead - behind), rtol=1e-3)


def test_a_larger_drain_is_pulled_no_harder():
    # Charged by the unit: an instant the solver cannot bring within the
    # tolerance does not outweigh one it nearly has.
    edges = _edges()
    small = _bump(edges, size=0.5).requires_grad_(True)
    large = _bump(edges, size=5.0).requires_grad_(True)
    moved = float(
        depth_drain.smallest_drain(small.detach(), edges, STIFFNESS, 1e9).move_before.max()
    )
    tolerance = 0.25 * moved

    def pull(mismatch: torch.Tensor) -> torch.Tensor:
        sizes = _carried_sizes(mismatch, edges, tolerance)
        gradient = torch.autograd.grad(sizes.sum(), mismatch)[0]
        return torch.sqrt((gradient**2 / edges.cells).sum(dim=1))

    with torch.no_grad():
        larger = _carried_sizes(large, edges, tolerance).min()
        smaller = _carried_sizes(small, edges, tolerance).max()
    assert float(larger) > 5.0 * float(smaller)
    assert torch.allclose(pull(large), pull(small), rtol=1e-6)


def test_a_half_that_drains_nothing_has_no_size_and_no_gradient():
    edges = _edges()
    mismatch = _bump(edges, size=0.05).requires_grad_(True)

    sizes = _carried_sizes(mismatch, edges, tolerance=1e9)
    gradient = torch.autograd.grad(sizes.sum(), mismatch)[0]

    assert torch.equal(sizes, torch.zeros_like(sizes))
    assert torch.equal(gradient, torch.zeros_like(gradient))


# --- the split on a model ---------------------------------------------------------


def _model(tmp_path, overrides=DRAIN):
    cfg, paths = staged_capsule_run(tmp_path, list(overrides))
    data = BubbleDataset(cfg, paths)
    torch.manual_seed(0)
    model = BubblePINN(cfg, geometry=training._geometry_priors(cfg, data))
    training.bind_depth_drain(model, data)
    return cfg, data, model


def _model_front(data, model, n: int = 3):
    times = torch.linspace(float(data.domain.t_min), float(data.domain.t_max), n)
    return model.front(times.reshape(-1, 1), n_body=24, n_cap=9)


def _bend_the_root_law(model) -> None:
    law = model.depth_root.knot_values
    law.copy_(torch.linspace(-0.2, 0.0, len(law)))


def _tight(model) -> None:
    """A tolerance no untrained pressure meets: every instant drains."""
    model.depth_drain_tolerance = 1e-4


def test_the_three_parts_sum_to_the_mismatch(tmp_path):
    cfg, data, model = _model(tmp_path)
    _bend_the_root_law(model)
    _tight(model)
    front, groups = _model_front(data, model), compute_groups(cfg)

    split = depth_drain.depth_split(model, front, groups, view_x_max=None)

    mismatch = laplace_jump_residual(model, front, groups)
    assert float(split.drained.abs().max()) > 0.0
    assert float(split.root.abs().max()) > 0.0
    assert torch.allclose(split.kept + split.drained + split.root, mismatch, atol=1e-5)


def test_the_mismatch_outside_the_law_keeps_its_gradient(tmp_path):
    # Neither the drain nor the root's law takes the pressure step's gradient
    # away: the plain rule is charged on all of it.
    cfg, data, model = _model(tmp_path)
    _bend_the_root_law(model)
    _tight(model)
    front, groups = _model_front(data, model), compute_groups(cfg)
    pressure = next(model.vapor_pressure.parameters())

    split = depth_drain.depth_split(model, front, groups, view_x_max=None)
    mismatch = laplace_jump_residual(model, front, groups)

    outside_the_law = (split.kept + split.drained).sum()
    of_split = torch.autograd.grad(outside_the_law, pressure, retain_graph=True)[0]
    of_mismatch = torch.autograd.grad(mismatch.sum(), pressure)[0]
    assert not split.drained.requires_grad and not split.root.requires_grad
    assert torch.allclose(of_split, of_mismatch, atol=1e-6)


def test_the_drain_needed_is_its_size_averaged_over_the_halves(tmp_path):
    cfg, data, model = _model(tmp_path)
    _tight(model)
    front, groups = _model_front(data, model), compute_groups(cfg)

    split = depth_drain.depth_split(model, front, groups, view_x_max=None)

    edges = depth_drain.half_edges(front, model.depth_root.starts_at)
    drain = split.drained.double()[edges.order, 0]
    sizes = depth_drain.drain_size(drain, edges.cells)
    assert split.needed.requires_grad
    assert float(split.needed.detach()) == pytest.approx(float(sizes.mean()), rel=1e-5)


def test_nothing_is_drained_past_the_cameras_view(tmp_path):
    cfg, data, model = _model(tmp_path)
    _tight(model)
    front, groups = _model_front(data, model), compute_groups(cfg)
    middle = float(front.points[:, 0].detach().median())

    split = depth_drain.depth_split(model, front, groups, view_x_max=middle)

    past = front.points[:, 0:1].detach() > middle
    assert float(split.drained[past].abs().max()) == 0.0
    assert float(split.drained[~past].abs().max()) > 0.0


def test_before_the_first_image_everything_is_kept(tmp_path):
    cfg, data, model = _model(tmp_path)
    _bend_the_root_law(model)
    _tight(model)
    model.depth_root.bind_first_image(float(data.domain.t_max) + 1.0)
    front, groups = _model_front(data, model), compute_groups(cfg)

    split = depth_drain.depth_split(model, front, groups, view_x_max=None)

    assert torch.equal(split.kept, laplace_jump_residual(model, front, groups))
    assert float(split.lengths.sum()) == 0.0
    assert float(split.needed) == 0.0


def _edge_context(cfg, data, model) -> registry.LossContext:
    times = torch.linspace(float(data.domain.t_min), float(data.domain.t_max), 3).reshape(-1, 1)
    return registry.LossContext(
        model, torch.zeros(1, 3), groups=compute_groups(cfg), front_times=times
    )


def _edge_term(cfg, data, model) -> torch.Tensor:
    return registry._laplace_term(_edge_context(cfg, data, model))


def _plain_rule(cfg, data, model) -> torch.Tensor:
    model.depth_drain = False
    try:
        return _edge_term(cfg, data, model)
    finally:
        model.depth_drain = True


def test_with_nothing_drained_the_loss_is_the_plain_rule(tmp_path):
    cfg, data, model = _model(tmp_path)
    model.depth_drain_tolerance = 1e9

    with_split = _edge_term(cfg, data, model)

    plain = _plain_rule(cfg, data, model)
    assert float(with_split.detach()) == pytest.approx(float(plain.detach()), rel=1e-6)


def test_a_drain_costs_more_than_the_plain_rule(tmp_path):
    # The first version charged a drain less than the error it stood for, and
    # the solver drained instead of answering.
    cfg, data, model = _model(tmp_path)
    _tight(model)

    with_split = _edge_term(cfg, data, model)

    plain = _plain_rule(cfg, data, model)
    assert float(with_split.detach()) > float(plain.detach())


def test_the_drain_is_charged_by_the_unit_on_top_of_the_plain_rule(tmp_path):
    cfg, data, model = _model(tmp_path)
    _tight(model)
    ctx = _edge_context(cfg, data, model)

    with_split = registry._laplace_term(ctx)

    plain = _plain_rule(cfg, data, model)
    second_charge = 2.0 * model.depth_drain_price * ctx.depth_split.needed
    assert float(second_charge.detach()) > 0.0
    assert float(with_split.detach()) == pytest.approx(
        float((plain + second_charge).detach()), rel=1e-6
    )


def test_at_no_price_the_split_is_only_counted(tmp_path):
    cfg, data, model = _model(tmp_path, [*DRAIN, "model.depth_drain_price_Pa=0.0"])
    _tight(model)

    with_split = _edge_term(cfg, data, model)

    plain = _plain_rule(cfg, data, model)
    assert float(with_split.detach()) == pytest.approx(float(plain.detach()), rel=1e-6)


def test_the_plain_rule_is_charged_outside_the_roots_law(tmp_path):
    cfg, data, model = _model(tmp_path)
    _bend_the_root_law(model)
    model.depth_drain_tolerance = 1e9
    ctx = _edge_context(cfg, data, model)

    with_law = registry._laplace_term(ctx)

    front = ctx.front
    outside = laplace_jump_residual(model, front, ctx.groups) - ctx.depth_split.root
    weights = front_arc_lengths(front)
    expected = (outside**2 * weights).sum() / weights.sum()
    assert float(ctx.depth_split.root.abs().max()) > 0.0
    assert float(with_law.detach()) == pytest.approx(float(expected.detach()), rel=1e-6)


def test_a_root_cap_left_out_stays_out_at_every_instant(tmp_path):
    cfg, data, model = _model(tmp_path, [*DRAIN, "model.laplace_root=exclude"])
    ctx = _edge_context(cfg, data, model)

    weights = registry._laplace_weights(ctx)

    root = root_cap_mask(ctx.front)
    assert float(weights[root].abs().max()) == 0.0
    assert float(weights[~root].min()) > 0.0


# --- configuration, training and the audit ----------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        (["model.sharp_interface=false"], "model.sharp_interface"),
        (["model.laplace_condition=mean"], "laplace_condition=pointwise"),
        (["model.laplace_weighting=samples"], "laplace_weighting=length"),
        (["model.depth_drain_tolerance_um=0.0"], "depth_drain_tolerance_um"),
        (["model.depth_drain_price_Pa=-1.0"], "depth_drain_price_Pa"),
    ],
)
def test_a_setting_the_split_cannot_work_with_is_refused(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        _model(tmp_path, [*DRAIN, *overrides])


WARMUP_STEPS = 1


def _training_run(tmp_path, steps: int, drain: bool):
    return staged_capsule_run(
        tmp_path,
        [
            *DRAIN,
            f"model.depth_drain={str(drain).lower()}",
            f"training.stage_b_warmup_steps={WARMUP_STEPS}",
            "training.freeze_geometry=true",
            f"training.steps={steps}",
            "training.log_every=1",
        ],
    )


def _train_past_the_warm_up_without_the_flag(tmp_path):
    """A run whose physics stage has trained the pressure, as the flag needs."""
    cfg, paths = _training_run(tmp_path, steps=WARMUP_STEPS + 2, drain=False)
    training.train(cfg, paths)
    return paths


def _continue_with_the_flag(tmp_path, paths, steps: int = 1):
    cfg, _ = _training_run(tmp_path, steps=steps, drain=True)
    return training.train(cfg, paths)


def test_a_run_that_takes_the_flag_on_fits_its_root_law_once_and_holds_it(tmp_path):
    paths = _train_past_the_warm_up_without_the_flag(tmp_path)
    done = torch.load(paths.checkpoint, weights_only=False)["state"]["done"]

    model, _, state = _continue_with_the_flag(tmp_path, paths)
    fitted = model.depth_root.knot_values.clone()
    model, _, _ = _continue_with_the_flag(tmp_path, paths)

    assert state["done"] == done + 1
    assert float(fitted.abs().max()) > 0.0
    assert torch.equal(model.depth_root.knot_values, fitted), "held, not fitted again"


def test_a_run_that_takes_the_flag_on_keeps_its_optimiser(tmp_path, caplog):
    paths = _train_past_the_warm_up_without_the_flag(tmp_path)

    with caplog.at_level("INFO", logger="naviernet.training"):
        _continue_with_the_flag(tmp_path, paths)

    assert "across model.depth_drain (the optimiser carried over)" in caplog.text


def test_a_run_with_the_flag_records_the_split(tmp_path):
    paths = _train_past_the_warm_up_without_the_flag(tmp_path)

    _, _, state = _continue_with_the_flag(tmp_path, paths)

    last = state["hist"][-1]
    assert math.isfinite(last["laplace"])
    shares = [last["edge_kept"], last["edge_drained"], last["edge_root"]]
    assert all(0.0 <= share <= 1.0 for share in shares)
    assert sum(shares) == pytest.approx(1.0, abs=1e-6)
    assert last["edge_drain_Pa"] >= 0.0


def test_a_run_without_the_flag_records_no_split(tmp_path):
    paths = _train_past_the_warm_up_without_the_flag(tmp_path)

    state = torch.load(paths.checkpoint, weights_only=False)["state"]

    assert "edge_drained" not in state["hist"][-1]


def test_a_run_with_no_trained_pressure_cannot_take_the_flag_on(tmp_path):
    cfg, paths = _training_run(tmp_path, steps=3, drain=True)

    with pytest.raises(ValueError, match="not past its warm-up"):
        training.train(cfg, paths)


def test_a_run_at_its_warm_ups_end_cannot_take_the_flag_on(tmp_path):
    # The warm-up holds the pressure: there is no trained pressure step to fit yet.
    cfg, paths = _training_run(tmp_path, steps=WARMUP_STEPS, drain=False)
    training.train(cfg, paths)

    with pytest.raises(ValueError, match="not past its warm-up"):
        _continue_with_the_flag(tmp_path, paths)


def test_a_run_that_held_a_root_law_cannot_drop_the_flag(tmp_path):
    paths = _train_past_the_warm_up_without_the_flag(tmp_path)
    _continue_with_the_flag(tmp_path, paths)
    cfg_plain, _ = _training_run(tmp_path, steps=1, drain=False)

    with pytest.raises(RuntimeError, match="depth_drain past its warm-up"):
        training.train(cfg_plain, paths)


def test_the_audit_reports_the_split_of_a_frame(tmp_path):
    cfg, data, model = _model(tmp_path)
    _tight(model)

    record = depth_drain_record(model, compute_groups(cfg), float(data.t[1]), view_x_max=None)

    shares = [record[f"{part}_share"] for part in ("kept", "drained", "root")]
    assert sum(shares) == pytest.approx(1.0, abs=1e-6)
    assert record["drained_force_uN"] > 0.0
    assert record["drain_rms_Pa"] > 0.0
    assert record["edge_move_um"] < record["edge_move_undrained_um"]


def test_the_audit_reports_no_split_without_the_flag(tmp_path):
    cfg, data, model = _model(tmp_path, SHARP)

    assert model.depth_root is None
    assert depth_drain_record(model, compute_groups(cfg), float(data.t[1]), None) is None
