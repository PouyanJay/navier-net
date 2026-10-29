"""The shear-free front (``model.interface_shear``) and the momentum law's sharp
phase properties (``model.momentum_phases=sharp``): the tangential stress the
liquid puts on the front, and the liquid's own friction and viscosity up to it."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from naviernet import training, warmup_cache
from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BubblePINN
from naviernet.physics import registry
from naviernet.physics.residuals import (
    _momentum_phase,
    brinkman_residuals,
    interface_shear_residual,
)

from .conftest import make_config, staged_walled_capsule_run

GROUPS = {"Re": 2.0}
SQRT_HALF = math.sqrt(0.5)

EXACT_BRINKMAN = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.walls_at=physical",
    "model.hard_walls=true",
    "model.momentum_closure=brinkman",
    "model.pressure_representation=two_phase",
    "model.mirror_symmetry=true",
    "model.exact_continuity=true",
    "model.exact_continuity_body_sources=32",
    "model.exact_continuity_cap_sources=8",
]
BOTH_FLAGS = ["model.interface_shear=true", "model.momentum_phases=sharp"]


class _Flow:
    """A stand-in model whose velocity is a closed-form field of ``(x, y)``."""

    def __init__(self, field):
        self.field = field

    def velocity(self, x, c=None):
        return self.field(x[:, 0:1], x[:, 1:2])


def _front(normal: tuple[float, float]) -> SimpleNamespace:
    points = torch.tensor([[0.3, 0.2, 0.0], [0.7, -0.4, 0.0]])
    return SimpleNamespace(points=points, normal=torch.tensor([normal, normal]))


HORIZONTAL, VERTICAL, DIAGONAL = (0.0, 1.0), (1.0, 0.0), (SQRT_HALF, SQRT_HALF)


@pytest.mark.parametrize(
    "field, normal, expected",
    [
        (lambda x, y: (0.0 * x + 1.0, 0.0 * y + 2.0), DIAGONAL, 0.0),
        (lambda x, y: (-y, x), HORIZONTAL, 0.0),
        (lambda x, y: (-y, x), DIAGONAL, 0.0),
        (lambda x, y: (y, 0.0 * x), HORIZONTAL, -0.5),
        (lambda x, y: (y, 0.0 * x), VERTICAL, 0.5),
        (lambda x, y: (y, 0.0 * x), DIAGONAL, 0.0),
        (lambda x, y: (x, -y), HORIZONTAL, 0.0),
        (lambda x, y: (x, -y), DIAGONAL, -1.0),
    ],
    ids=[
        "uniform",
        "rotation-horizontal",
        "rotation-diagonal",
        "shear-horizontal",
        "shear-vertical",
        "shear-principal",
        "extension-principal",
        "extension-diagonal",
    ],
)
def test_the_residual_is_the_tangential_stress_over_re(field, normal, expected):
    residual = interface_shear_residual(_Flow(field), _front(normal), GROUPS)

    assert residual.shape == (2, 1)
    assert residual.flatten().tolist() == pytest.approx([expected, expected], abs=1e-6)


def _model(tmp_path, extra=()):
    cfg, paths = staged_walled_capsule_run(tmp_path, [*EXACT_BRINKMAN, *extra])
    data = BubbleDataset(cfg, paths)
    torch.manual_seed(0)
    model = BubblePINN(cfg, geometry=training._geometry_priors(cfg, data))
    model.kinematics_engaged = True
    with torch.no_grad():
        model.stream_net.mlp[-1].weight.normal_(0.0, 0.3)
        model.stream_net.mlp[-1].bias.fill_(0.2)
    return data, model


def test_on_the_exact_liquid_flow_it_is_finite_and_trains_the_stream_net(tmp_path):
    data, model = _model(tmp_path, BOTH_FLAGS)
    front = model.front(torch.tensor([[float(data.t[1])]]), n_body=24, n_cap=8)

    residual = interface_shear_residual(model, front, {"Re": 50.0})
    (residual**2).mean().backward()

    assert torch.isfinite(residual).all() and residual.abs().max() > 0.0
    grads = [p.grad for p in model.stream_net.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "the condition reaches N"


def _liquid_and_vapour(data, model, n: int = 600) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(2)
    lo, hi = model.wall_span
    xy = torch.rand(n, 2, generator=g) * torch.tensor([float(data.x.max()), hi - lo])
    xy[:, 1] += lo
    points = torch.cat([xy, torch.full((n, 1), float(data.t[1]))], dim=1)
    return points, model.liquid_side(points)


def test_sharp_phases_take_the_liquid_up_to_the_front_and_nothing_in_the_vapour(tmp_path):
    data, model = _model(tmp_path, BOTH_FLAGS)
    points, liquid = _liquid_and_vapour(data, model)
    assert liquid.any() and (~liquid).any()

    phase = _momentum_phase(model, points, None)
    x = points.clone().requires_grad_(True)
    residual = brinkman_residuals(model, x, {"hele_shaw": 0.2, "mu_ratio": 37.0, "Re": 50.0})

    assert phase[liquid].eq(0.0).all() and phase[~liquid].eq(1.0).all()
    assert residual.mom_x[~liquid].eq(0.0).all() and residual.mom_y[~liquid].eq(0.0).all()
    assert residual.mom_x[liquid].abs().sum() > 0.0


def test_mixture_phases_are_the_smoothed_fraction(tmp_path):
    data, model = _model(tmp_path)
    points, _ = _liquid_and_vapour(data, model)

    assert torch.equal(_momentum_phase(model, points, None), model.alpha(points))


def test_the_registry_admits_the_condition_only_on_its_flag_and_the_brinkman_law():
    fields = ("phi", "u", "v", "s", "p")

    def ids(**flags):
        return {e.id for e in registry.enabled_equations(fields, sharp_interface=True, **flags)}

    assert "interface_shear" in ids(momentum_closure="brinkman", interface_shear=True)
    assert "interface_shear" not in ids(momentum_closure="brinkman")
    assert "interface_shear" not in ids(momentum_closure="darcy", interface_shear=True)
    on = make_config(["model.interface_shear=true"]).model
    assert registry.flags_from(on)["interface_shear"] is True


@pytest.mark.parametrize(
    "extra, message",
    [
        (["model.momentum_closure=darcy", "model.interface_shear=true"], "momentum_closure"),
        (
            ["model.pressure_representation=single", "model.momentum_phases=sharp"],
            "two_phase",
        ),
        (["model.momentum_phases=blended"], "momentum_phases"),
    ],
)
def test_the_trainer_refuses_settings_the_conditions_do_not_fit(tmp_path, extra, message):
    cfg, paths = staged_walled_capsule_run(
        tmp_path,
        [
            *EXACT_BRINKMAN,
            "training.stage_b_warmup_steps=1",
            "training.freeze_geometry=true",
            "training.edge_transport_after_freeze=false",
            *extra,
        ],
    )

    with pytest.raises(ValueError, match=message):
        training.train(cfg, paths)


def test_a_run_with_both_trains_with_finite_losses(tmp_path):
    cfg, paths = staged_walled_capsule_run(
        tmp_path,
        [
            *EXACT_BRINKMAN,
            *BOTH_FLAGS,
            "model.outlet_condition=zero_gradient",
            "training.stage_b_warmup_steps=1",
            "training.freeze_geometry=true",
            "training.edge_transport_after_freeze=false",
            "training.steps=3",
            "training.log_every=1",
        ],
    )

    _, _, state = training.train(cfg, paths)

    last = state["hist"][-1]
    assert "interface_shear" in last
    losses = [float(v) for v in last.values() if isinstance(v, int | float)]
    assert losses and all(math.isfinite(v) for v in losses)


def test_both_join_the_warm_up_others_already_share():
    base = make_config(["model=stage_b", "model.sharp_interface=true"])
    variant = make_config(
        [
            "model=stage_b",
            "model.sharp_interface=true",
            *BOTH_FLAGS,
            "training.weights.interface_shear=3.0",
        ]
    )

    assert warmup_cache._key_config(base, ()) == warmup_cache._key_config(variant, ())


# --- inertia on the exact liquid flow ------------------------------------------


def test_the_exact_flow_rate_is_the_difference_of_its_solves(tmp_path):
    data, model = _model(tmp_path)
    points, liquid = _liquid_and_vapour(data, model)
    x = points.clone().requires_grad_(True)
    u, v = model.velocity(x)
    autograd_u_t = torch.autograd.grad(u.sum(), x, retain_graph=True)[0][:, 2:3]
    autograd_v_t = torch.autograd.grad(v.sum(), x)[0][:, 2:3]

    u_t, v_t = model.velocity_rate(x, autograd_u_t, autograd_v_t)

    step = torch.tensor([0.0, 0.0, 1e-3])
    ahead, behind = (
        model._liquid_velocity((points[liquid] + step).requires_grad_(True), False),
        model._liquid_velocity((points[liquid] - step).requires_grad_(True), False),
    )
    expected = (ahead[0] - behind[0]) / 2e-3
    assert torch.allclose(u_t[liquid], expected.detach(), atol=1e-4)
    assert u_t[liquid].abs().max() > 0.0, "the capsule grows, so its liquid flow changes"
    assert torch.equal(u_t[~liquid], autograd_u_t[~liquid])
    assert torch.equal(v_t[~liquid], autograd_v_t[~liquid])


@pytest.mark.parametrize(("end", "sign"), [(0, 1.0), (-1, -1.0)])
def test_at_the_window_ends_the_rate_looks_only_inward(tmp_path, end, sign):
    data, model = _model(tmp_path)
    points, _ = _liquid_and_vapour(data, model)
    at_end = points.clone()
    at_end[:, 2] = float(data.t[end])
    at_end = at_end[model.liquid_side(at_end)]
    zeros = torch.zeros(len(at_end), 1)

    u_t, _ = model.velocity_rate(at_end, zeros, zeros)

    h = sign * 1e-3

    def u_at(k: float) -> torch.Tensor:
        shifted = at_end + torch.tensor([0.0, 0.0, k * h])
        return model._liquid_velocity(shifted.requires_grad_(True), False)[0].detach()

    expected = (-3.0 * u_at(0.0) + 4.0 * u_at(1.0) - u_at(2.0)) / (2.0 * h)
    assert torch.allclose(u_t, expected, atol=1e-4)
    assert u_t.abs().max() > 0.0


def test_before_the_physics_engages_the_rate_is_autograd(tmp_path):
    data, model = _model(tmp_path)
    model.kinematics_engaged = False
    u_t, v_t = torch.ones(3, 1), torch.zeros(3, 1)

    assert model.velocity_rate(torch.zeros(3, 3), u_t, v_t) == (u_t, v_t)


def test_the_inertia_trains_the_stream_net(tmp_path):
    data, model = _model(tmp_path, ["model.momentum_inertia=true", *BOTH_FLAGS])
    points, liquid = _liquid_and_vapour(data, model)
    x = points[liquid].clone().requires_grad_(True)
    groups = {"hele_shaw": 0.2, "mu_ratio": 37.0, "Re": 50.0, "rho_ratio": 120.0}

    residual = brinkman_residuals(model, x, groups, inertia=True)
    (residual.mom_x**2 + residual.mom_y**2).mean().backward()

    grads = [p.grad for p in model.stream_net.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    assert torch.isfinite(residual.mom_x).all()


def test_a_run_with_inertia_trains_with_finite_losses(tmp_path):
    cfg, paths = staged_walled_capsule_run(
        tmp_path,
        [
            *EXACT_BRINKMAN,
            *BOTH_FLAGS,
            "model.momentum_inertia=true",
            "model.outlet_condition=zero_gradient",
            "training.stage_b_warmup_steps=1",
            "training.freeze_geometry=true",
            "training.edge_transport_after_freeze=false",
            "training.steps=3",
            "training.log_every=1",
        ],
    )

    _, _, state = training.train(cfg, paths)

    losses = [float(v) for v in state["hist"][-1].values() if isinstance(v, int | float)]
    assert losses and all(math.isfinite(v) for v in losses)
