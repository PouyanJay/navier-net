"""The liquid's velocity divergence-free by construction (``model.exact_continuity``):
the strip's Green functions, their closed-form derivatives, and the assembled
velocity, which conserves volume exactly in the liquid, moves with the front,
passes nothing through the walls and carries exactly the inlet flux."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from naviernet import training
from naviernet.data.dataset import BubbleDataset
from naviernet.models import potential_flow as pf
from naviernet.models.pinn import STREAM_PREFIX, BubblePINN, _arclength_weights
from naviernet.physics.residuals import gradients

from .conftest import staged_walled_capsule_run

EXACT = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.walls_at=physical",
    "model.hard_walls=true",
    "model.exact_continuity=true",
    "model.exact_continuity_body_sources=32",
    "model.exact_continuity_cap_sources=8",
]
CHANNEL = pf.Channel(y_lo=0.0, y_hi=1.0, x_in=0.0)
SOURCE = torch.tensor([[0.8, 0.3]], dtype=torch.float64)


def _field(boundary: str = pf.NEUMANN, strength: float = 1.0) -> pf.HarmonicField:
    return pf.HarmonicField(
        CHANNEL, SOURCE, torch.tensor([strength], dtype=torch.float64), boundary=boundary
    )


def _grid(n: int = 200) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    points = torch.rand(n, 2, generator=g, dtype=torch.float64) * torch.tensor([3.0, 1.0])
    return points[(points - SOURCE).norm(dim=1) > 0.05]


def _model(tmp_path, extra=(), mirror: bool = True):
    overrides = [*EXACT, *(["model.mirror_symmetry=true"] if mirror else []), *extra]
    cfg, paths = staged_walled_capsule_run(tmp_path, overrides)
    data = BubbleDataset(cfg, paths)
    torch.manual_seed(0)
    return data, BubblePINN(cfg, geometry=training._geometry_priors(cfg, data))


def _with_stream(model) -> None:
    """A nonzero stream function, so its part of the velocity is exercised."""
    with torch.no_grad():
        model.stream_net.mlp[-1].weight.normal_(0.0, 0.3)
        model.stream_net.mlp[-1].bias.fill_(0.2)


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


# --- the Green functions -----------------------------------------------------


def test_the_neumann_field_is_harmonic_and_passes_nothing_through_walls_or_inlet():
    field = _field()
    points = _grid()
    xs = torch.linspace(0.0, 3.0, 50, dtype=torch.float64)

    laplacian = field.value(points, 2, 0) + field.value(points, 0, 2)
    lower = torch.stack([xs, torch.zeros_like(xs)], dim=1)
    upper = torch.stack([xs, torch.ones_like(xs)], dim=1)
    inlet = torch.stack([torch.zeros_like(xs), xs / 3.0], dim=1)

    assert float(laplacian.abs().max()) < 1e-9
    assert float(field.value(lower, 0, 1).abs().max()) < 1e-12
    assert float(field.value(upper, 0, 1).abs().max()) < 1e-12
    assert float(field.value(inlet, 1, 0).abs().max()) < 1e-12


def test_a_unit_source_sends_its_whole_flux_downstream():
    ys = torch.linspace(0.0, 1.0, 2001, dtype=torch.float64)
    far = torch.stack([torch.full_like(ys, 3.5), ys], dim=1)

    flux = float(_field().value(far, 1, 0).mean())  # mean over a unit width

    assert flux == pytest.approx(1.0, abs=1e-6)


def _outflow_round_the_source(field: pf.HarmonicField, radius: float = 0.02) -> float:
    angles = torch.linspace(0.0, 2.0 * math.pi, 2001, dtype=torch.float64)[:-1]
    normal = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    circle = SOURCE + radius * normal
    flux = normal[:, :1] * field.value(circle, 1, 0) + normal[:, 1:2] * field.value(
        circle, 0, 1
    )
    return float(flux.mean()) * 2.0 * math.pi * radius


@pytest.mark.parametrize("boundary", [pf.NEUMANN, pf.DIRICHLET])
def test_a_unit_source_has_unit_outflow_round_it(boundary):
    assert _outflow_round_the_source(_field(boundary)) == pytest.approx(1.0, abs=1e-6)


def test_the_dirichlet_field_vanishes_on_both_walls():
    xs = torch.linspace(0.0, 3.0, 50, dtype=torch.float64)
    field = _field(pf.DIRICHLET)

    for wall in (0.0, 1.0):
        line = torch.stack([xs, torch.full_like(xs, wall)], dim=1)
        assert float(field.value(line, 0, 0).abs().max()) < 1e-12
    assert float(field.value(_grid(), 0, 0).max()) < 0.0, "a source sinks below the walls' zero"


def test_closed_form_derivatives_match_finite_differences():
    field = _field()
    points = _grid(20)
    h = 1e-5
    step = torch.tensor([h, 0.0], dtype=torch.float64)

    for a, b in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 1)):
        difference = (field.value(points + step, a, b) - field.value(points - step, a, b)) / (
            2 * h
        )
        assert torch.allclose(field.value(points, a + 1, b), difference, rtol=1e-5, atol=1e-6)


def test_autograd_differentiates_the_fields_to_third_order():
    points = _grid(6).requires_grad_(True)

    for field in (_field(), _field(pf.DIRICHLET)):
        for a, b in ((0, 0), (1, 0), (0, 1)):
            assert torch.autograd.gradgradcheck(
                lambda z, f=field, a=a, b=b: pf.harmonic(f, z, a, b), (points,), eps=1e-6
            )


# --- the assembled velocity --------------------------------------------------


def test_the_liquid_is_divergence_free_and_nothing_crosses_the_walls(tmp_path):
    data, model = _model(tmp_path)
    _with_stream(model)
    t = float(data.t[3])
    x = _liquid_points(data, model, t)

    u, v = model.velocity(x)
    u_x, u_y, _ = gradients(u, x)
    v_x, v_y, _ = gradients(v, x)

    scale = float(torch.cat([u_x, u_y, v_x, v_y]).abs().max())
    assert float((u_x + v_y).abs().max()) < 1e-5 * max(scale, 1.0)
    lo, hi = model.wall_span
    xs = torch.linspace(0.0, float(data.x.max()), 64)
    for wall in (lo, hi):
        line = torch.stack([xs, torch.full_like(xs, wall), torch.full_like(xs, t)], dim=1)
        assert float(model.velocity(line)[1].abs().max()) < 1e-6


def test_each_section_carries_the_inlet_flux_plus_the_growth_behind_it(tmp_path):
    data, model = _model(tmp_path)
    _with_stream(model)
    t = float(data.t[3])
    front = model.front(torch.tensor([[t]], requires_grad=True), 256, 64)
    xf = front.points[:, 0].detach()
    weights = _arclength_weights(front.points.detach(), 256, 64)[:, 0]
    speed = front.normal_speed.detach()[:, 0]
    lo, hi = model.wall_span
    q_in = float(model._inflow_groups["u_inlet_star"]) * (hi - lo)
    upstream, inside = 0.05, float(xf.mean())

    assert _section_flux(model, upstream, t) == pytest.approx(q_in, rel=1e-3)
    growth = float((speed * weights)[xf < inside].sum())
    assert _section_flux(model, inside, t) == pytest.approx(q_in + growth, abs=0.03 * q_in)


def test_the_stream_function_adds_no_flux_through_any_section(tmp_path):
    data, model = _model(tmp_path)
    t = float(data.t[3])
    sections = (0.05, 0.4, 0.7, 1.0, 1.6)
    potential_only = [_section_flux(model, x, t) for x in sections]

    _with_stream(model)
    with_stream = [_section_flux(model, x, t) for x in sections]

    # Not zero: the section's liquid rows stop up to a row short of the front,
    # where W N is small but not zero.
    q_in = float(model._inflow_groups["u_inlet_star"]) * (
        model.wall_span[1] - model.wall_span[0]
    )
    assert with_stream == pytest.approx(potential_only, abs=1e-3 * q_in)


def test_the_liquid_moves_with_the_front(tmp_path):
    data, model = _model(tmp_path)
    _with_stream(model)
    front = model.front(torch.tensor([[float(data.t[3])]], requires_grad=True), 128, 32)

    u, v = model.velocity(front.points.detach())
    normal = front.normal.detach()
    miss = u * normal[:, :1] + v * normal[:, 1:2] - front.normal_speed.detach()

    assert float(miss.pow(2).mean().sqrt()) < 0.05 * float(front.normal_speed.abs().max())


def test_the_capacity_lies_between_the_front_and_the_walls(tmp_path):
    data, model = _model(tmp_path)
    t = float(data.t[3])
    x = _liquid_points(data, model, t)

    capacity = model._fields_at(t).capacity
    front = model.front(torch.tensor([[t]], requires_grad=True), 128, 32).points.detach()
    lo, hi = model.wall_span
    xs = torch.linspace(0.0, float(data.x.max()), 64, dtype=torch.float64)

    inside = capacity.value(x[:, :2].detach().double(), 0, 0)
    assert float(inside.min()) > -1e-3
    assert float(inside.max()) < 1.0 + 1e-6
    assert float(capacity.value(front[:, :2].double(), 0, 0).abs().max()) < 0.02
    for wall in (lo, hi):
        line = torch.stack([xs, torch.full_like(xs, wall)], dim=1)
        assert torch.allclose(
            capacity.value(line, 0, 0), torch.ones(64, 1, dtype=torch.float64)
        )


def test_the_velocity_keeps_the_mirror_symmetry(tmp_path):
    data, model = _model(tmp_path)
    _with_stream(model)
    x = _liquid_points(data, model, float(data.t[2])).detach()
    mirrored = x.clone()
    mirrored[:, 1] = 2.0 * model.mirror_axis - x[:, 1]

    (u, v), (u_m, v_m) = model.velocity(x), model.velocity(mirrored)

    assert torch.allclose(u, u_m, atol=1e-4)
    assert torch.allclose(v, -v_m, atol=1e-4)


def test_without_a_mirror_axis_the_whole_front_is_solved(tmp_path):
    data, model = _model(tmp_path, mirror=False)
    t = float(data.t[3])

    assert _section_flux(model, 0.05, t) == pytest.approx(
        float(model._inflow_groups["u_inlet_star"]) * (model.wall_span[1] - model.wall_span[0]),
        rel=1e-3,
    )


def test_the_vapour_keeps_the_networks_velocity_and_finite_gradients(tmp_path):
    data, model = _model(tmp_path)
    t = float(data.t[3])
    x = torch.tensor([[0.7, 0.5, t], [0.5, 0.52, t]], requires_grad=True)
    assert not bool(model.liquid_side(x).any()), "both points inside the capsule"

    u, v = model.velocity(x)
    (u_x,) = torch.autograd.grad(u.sum(), x)
    u_net, v_net = model._network_velocity(x, None)

    assert torch.allclose(u, u_net) and torch.allclose(v, v_net)
    assert bool(torch.isfinite(u_x).all())


def test_it_starts_as_the_potential_flow(tmp_path):
    data, model = _model(tmp_path)
    t = float(data.t[3])
    x = _liquid_points(data, model, t).detach()
    fields = model._fields_at(t)

    u, v = model.velocity(x)

    xy = x[:, :2]
    assert torch.allclose(
        u, model._inflow(x[:, 1:2]) + fields.potential.value(xy, 1, 0), atol=1e-6
    )
    assert torch.allclose(v, fields.potential.value(xy, 0, 1), atol=1e-6)


def test_it_waits_for_the_physics_to_engage(tmp_path):
    data, model = _model(tmp_path)
    x = _liquid_points(data, model, float(data.t[3])).detach()
    model.kinematics_engaged = False

    (u, v), (u_net, v_net) = model.velocity(x), model._network_velocity(x, None)

    assert torch.equal(u, u_net) and torch.equal(v, v_net)


# --- configuration and training ----------------------------------------------


def test_with_the_flag_off_every_other_net_starts_the_same(tmp_path):
    cfg_off, paths = staged_walled_capsule_run(
        tmp_path, [*EXACT[:-3], "model.exact_continuity=false"]
    )
    data = BubbleDataset(cfg_off, paths)
    _, model_on = _model(tmp_path, mirror=False)
    torch.manual_seed(0)
    model_off = BubblePINN(cfg_off, geometry=training._geometry_priors(cfg_off, data))

    on = {k: v for k, v in model_on.state_dict().items() if not k.startswith(STREAM_PREFIX)}
    assert on.keys() == model_off.state_dict().keys()
    assert all(torch.equal(on[k], model_off.state_dict()[k]) for k in on)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["model.hard_kinematics=true"], "hard_kinematics"),
        (["model.hard_walls=false"], "hard_walls"),
        (["model.momentum_inertia=true"], "momentum_inertia"),
        (["model.exact_continuity_offset=0"], "offset"),
        (["model.sharp_interface=false"], "sharp_interface"),
    ],
)
def test_conflicting_settings_are_refused(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        _model(tmp_path, extra)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["training.freeze_geometry=false"], "freeze_geometry"),
        (["training.edge_transport_after_freeze=true"], "edge_transport"),
    ],
)
def test_the_trainer_requires_a_frozen_outline_without_edge_transport(tmp_path, extra, message):
    cfg, paths = staged_walled_capsule_run(
        tmp_path,
        [
            *EXACT,
            "training.stage_b_warmup_steps=1",
            "training.freeze_geometry=true",
            "training.edge_transport_after_freeze=false",
            *extra,
        ],
    )

    with pytest.raises(ValueError, match=message):
        training.train(cfg, paths)


def _training_run(tmp_path, extra=()):
    return staged_walled_capsule_run(
        tmp_path,
        [
            *EXACT,
            "model.mirror_symmetry=true",
            "model.outlet_condition=zero_gradient",
            "training.stage_b_warmup_steps=1",
            "training.freeze_geometry=true",
            "training.edge_transport_after_freeze=false",
            "training.steps=3",
            "training.log_every=1",
            *extra,
        ],
    )


def test_a_run_trains_with_finite_losses(tmp_path):
    cfg, paths = _training_run(tmp_path)

    _, _, state = training.train(cfg, paths)

    losses = [float(v) for v in state["hist"][-1].values() if isinstance(v, int | float)]
    assert losses and all(math.isfinite(v) for v in losses)


def test_the_boundary_batches_move_to_the_collocation_instants():
    x_coll = torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.5]])
    inlet = torch.tensor([[0.0, 0.2, 0.15], [0.0, 0.4, 0.45]])
    outlet = torch.tensor([[1.0, 0.2, 0.32]], requires_grad=True)

    moved_inlet, _, moved_outlet = training._at_collocation_instants(
        x_coll, inlet, inlet, outlet
    )

    assert moved_inlet[:, 2].tolist() == pytest.approx([0.1, 0.5])
    assert moved_outlet[:, 2].tolist() == pytest.approx([0.5])
    assert moved_outlet.requires_grad and moved_outlet.is_leaf


def test_a_warm_up_saved_without_the_flag_resumes_whole(tmp_path):
    cfg_off, paths = _training_run(
        tmp_path, ["model.exact_continuity=false", "training.steps=1"]
    )
    training.train(cfg_off, paths)
    saved = torch.load(paths.checkpoint, weights_only=False)["state"]
    cfg_on, _ = _training_run(tmp_path, ["training.steps=2"])

    _, _, state = training.train(cfg_on, paths)

    assert state["done"] == saved["done"] + 2
    assert set(saved) <= set(state), "the warm-up's whole state carried forward"


def test_a_warm_up_saved_with_the_flag_feeds_a_run_without_it(tmp_path):
    cfg_on, paths = _training_run(tmp_path, ["training.steps=1"])
    training.train(cfg_on, paths)
    cfg_off, _ = _training_run(tmp_path, ["model.exact_continuity=false", "training.steps=1"])

    _, _, state = training.train(cfg_off, paths)

    assert state["done"] == 2


def test_a_stream_net_trained_past_the_warm_up_is_not_dropped(tmp_path):
    cfg_on, paths = _training_run(tmp_path, ["training.steps=2"])
    training.train(cfg_on, paths)
    cfg_off, _ = _training_run(tmp_path, ["model.exact_continuity=false", "training.steps=1"])

    with pytest.raises(RuntimeError, match="exact_continuity past its warm-up"):
        training.train(cfg_off, paths)


def test_the_image_plane_is_where_the_inlet_is_scored(tmp_path):
    cfg, paths = staged_walled_capsule_run(tmp_path, EXACT)
    data = BubbleDataset(cfg, paths)

    inlet, _ = data.sample_boundary(16, np.random.default_rng(0))

    assert torch.all(inlet[:, 0] == pf.INLET_X)
