"""Depth-averaged inertia in the Brinkman law (``model.momentum_inertia``):
rho~ (u_t + 6/5 u.grad u) + grad p + C_HS mu~ u - (mu~/Re) lap u = 0."""

from __future__ import annotations

import pytest
import torch

from naviernet import training
from naviernet.physics.residuals import DEPTH_PROFILE_CONVECTION, brinkman_residuals

from .conftest import staged_capsule_run

GROUPS = {"hele_shaw": 0.25, "mu_ratio": 37.0, "rho_ratio": 120.0, "Re": 200.0}
DRAG = GROUPS["hele_shaw"]


def _tied(x: torch.Tensor) -> torch.Tensor:
    """Zero that keeps every derivative in the graph, so a field linear in x
    still has a (zero) second derivative to take."""
    return 0.0 * (x**3).sum(dim=1, keepdim=True)


class _Fluid:
    """A stand-in model of uniform phase (``vapour`` its vapour fraction) with
    prescribed velocity and liquid pressure."""

    pressure_representation = "two_phase"

    def __init__(self, velocity, pressure, vapour: float = 0.0):
        self._velocity, self._pressure, self._vapour = velocity, pressure, vapour

    def alpha(self, x, c=None):
        return self._vapour + _tied(x)

    def velocity(self, x, c=None):
        u, v = self._velocity(x)
        return u + _tied(x), v + _tied(x)

    def liquid_pressure(self, x, c=None):
        return self._pressure(x) + _tied(x)


def _points(n: int = 64) -> torch.Tensor:
    torch.manual_seed(0)
    return (torch.rand(n, 3) * 2.0 - 1.0).requires_grad_(True)


def _norm(res) -> float:
    return float((res.mom_x.abs() + res.mom_y.abs()).max())


def test_a_uniformly_accelerating_stream_balances_its_pressure_gradient():
    # u = a t: rho u_t = a and drag C a t, carried by p = -(a + C a t) x.
    a = 0.7
    model = _Fluid(
        lambda x: (a * x[:, 2:3], torch.zeros_like(x[:, :1])),
        lambda x: -(a + DRAG * a * x[:, 2:3]) * x[:, 0:1],
    )

    with_inertia = brinkman_residuals(model, _points(), GROUPS, inertia=True)
    without = brinkman_residuals(model, _points(), GROUPS, inertia=False)

    assert _norm(with_inertia) < 1e-5
    assert float(without.mom_x.abs().min()) == pytest.approx(a, rel=1e-4), "inertia missing"


def test_a_steady_stagnation_flow_carries_the_six_fifths_convection():
    # u = x, v = -y: u u_x = x and v v_y = y, so p_x = -(C + 6/5) x and
    # p_y = (C - 6/5) y; the Laplacian vanishes.
    k = DEPTH_PROFILE_CONVECTION
    model = _Fluid(
        lambda x: (x[:, 0:1], -x[:, 1:2]),
        lambda x: -(DRAG + k) * x[:, 0:1] ** 2 / 2 + (DRAG - k) * x[:, 1:2] ** 2 / 2,
    )

    assert _norm(brinkman_residuals(model, _points(), GROUPS, inertia=True)) < 1e-5


def test_the_inertia_is_not_scored_inside_the_vapour():
    # Under the two-phase pressure the law is weighted by the liquid fraction:
    # an accelerating vapour with no pressure gradient leaves no residual.
    vapour = _Fluid(
        lambda x: (x[:, 2:3], torch.zeros_like(x[:, :1])), lambda x: 0.0 * x[:, :1], 1.0
    )

    assert _norm(brinkman_residuals(vapour, _points(), GROUPS, inertia=True)) == 0.0


def test_inertia_is_refused_outside_the_brinkman_law(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        ["model=stage_b", "model.momentum_closure=darcy", "model.momentum_inertia=true"],
    )

    with pytest.raises(ValueError, match="momentum_inertia"):
        training.train(cfg, paths)


def test_a_sharp_two_phase_run_trains_with_inertia(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            "model=stage_b",
            "model.front_geometry=true",
            "model.sharp_interface=true",
            "model.pressure_representation=two_phase",
            "model.momentum_closure=brinkman",
            "model.momentum_inertia=true",
            "training.stage_b_warmup_steps=1",
            "training.steps=3",
            "training.log_every=1",
        ],
    )

    _, _, state = training.train(cfg, paths)

    assert torch.isfinite(torch.tensor(state["hist"][-1]["brinkman"]))
