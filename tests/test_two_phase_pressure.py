"""The two-phase pressure representation and the curl-free vapour velocity."""

from __future__ import annotations

import pytest
import torch

from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BubblePINN
from naviernet.physics import registry
from naviernet.physics.groups import compute_groups
from naviernet.physics.residuals import (
    jump_total_curvature,
    laplace_jump_residual,
    vapour_potential_residual,
)
from naviernet.training import _geometry_priors, train

from .conftest import staged_capsule_run

STAGE_B_FIELDS = ("phi", "u", "v", "s", "p", "T")
SHARP = ["model=stage_b", "model.front_geometry=true", "model.sharp_interface=true"]
TWO_PHASE = [*SHARP, "model.pressure_representation=two_phase"]


def _model(tmp_path, overrides):
    cfg, paths = staged_capsule_run(tmp_path, overrides)
    data = BubbleDataset(cfg, paths)
    torch.manual_seed(0)
    return cfg, data, BubblePINN(cfg, geometry=_geometry_priors(cfg, data))


def _points(data, n: int = 64) -> torch.Tensor:
    torch.manual_seed(1)
    d = data.domain
    spread = torch.tensor([d.x_max - d.x_min, d.y_max - d.y_min, d.t_max - d.t_min])
    origin = torch.tensor([d.x_min, d.y_min, d.t_min])
    return origin + torch.rand(n, 3) * spread


def test_two_phase_pressure_blends_the_vapour_scalar_with_the_liquid_net(tmp_path):
    _, data, model = _model(tmp_path, TWO_PHASE)
    x = _points(data)

    pressure = model.pressure(x)

    alpha = model.alpha(x)
    expected = alpha * model.p_vapor(x[:, 2:3]) + (1 - alpha) * model.liquid_pressure(x)
    assert torch.allclose(pressure, expected, atol=1e-6)
    assert not torch.allclose(pressure, model.liquid_pressure(x)), "the blend differs"


def test_the_blend_cannot_move_the_interface(tmp_path):
    """The indicator enters the blend detached: the pressure's gradient reaches
    the liquid net and p_v, never the geometry."""
    _, data, model = _model(tmp_path, TWO_PHASE)
    geometry_params = list(model.nets["phi"].parameters())

    model.pressure(_points(data)).sum().backward()

    assert all(p.grad is None or torch.all(p.grad == 0) for p in geometry_params)


def test_single_representation_keeps_pressure_and_liquid_pressure_identical(tmp_path):
    _, data, model = _model(tmp_path, SHARP)
    x = _points(data)

    assert torch.equal(model.pressure(x), model.liquid_pressure(x))


def test_jump_condition_reads_the_liquid_branch_on_the_front(tmp_path):
    cfg, data, model = _model(tmp_path, TWO_PHASE)
    groups = compute_groups(cfg)
    front = model.front(torch.tensor([[float(data.t[1])]]), 16, 4)
    capillary = jump_total_curvature(model, front, groups) / groups["We"]
    vapour = model.p_vapor(front.points[:, 2:3])

    residual = laplace_jump_residual(model, front, groups)

    liquid_branch = vapour - model.liquid_pressure(front.points) - capillary
    blended = vapour - model.pressure(front.points) - capillary
    assert torch.allclose(residual, liquid_branch, atol=1e-6)
    assert not torch.allclose(residual, blended), "on the front the blend is half p_v"


class _Vapour:
    """A stand-in model that is vapour everywhere, with a given velocity."""

    def __init__(self, u, v) -> None:
        self._u, self._v = u, v

    def alpha(self, x, c=None):
        return torch.ones_like(x[:, :1])

    def velocity(self, x, c=None):
        return self._u(x), self._v(x)


def test_vapour_potential_residual_vanishes_for_a_potential_flow_only():
    torch.manual_seed(0)
    x = torch.rand(32, 3, requires_grad=True)
    potential = _Vapour(lambda x: x[:, 0:1], lambda x: -x[:, 1:2])  # grad of (x^2 - y^2)/2
    shear = _Vapour(lambda x: x[:, 1:2], lambda x: 0.0 * x[:, 0:1])  # u = y: curl = -1

    assert vapour_potential_residual(potential, x).abs().max().item() < 1e-6
    assert vapour_potential_residual(shear, x).mean().item() == pytest.approx(-1.0)


def test_vapour_potential_is_gated_by_the_two_phase_pressure():
    base = registry.enabled_equations(STAGE_B_FIELDS, sharp_interface=True)
    two_phase = registry.enabled_equations(
        STAGE_B_FIELDS, sharp_interface=True, two_phase_pressure=True
    )
    diffuse = registry.enabled_equations(STAGE_B_FIELDS, two_phase_pressure=True)

    base_ids, two_phase_ids = {e.id for e in base}, {e.id for e in two_phase}
    assert "vapour_potential" not in base_ids
    assert "vapour_potential" in two_phase_ids
    assert two_phase_ids - {"vapour_potential"} == base_ids
    assert "vapour_potential" not in {e.id for e in diffuse}, "a sharp-mode term"


def test_two_phase_pressure_requires_the_sharp_interface(tmp_path):
    with pytest.raises(ValueError, match="sharp_interface"):
        _model(
            tmp_path,
            [
                "model=stage_b",
                "model.front_geometry=true",
                "model.pressure_representation=two_phase",
            ],
        )


def test_a_two_phase_run_trains_and_logs_the_vapour_term(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*TWO_PHASE, "training.log_every=1"])

    _, _, state = train(cfg, paths)

    assert all("vapour_potential" in record for record in state["hist"])
    assert all(
        torch.isfinite(torch.tensor(record["vapour_potential"])) for record in state["hist"]
    )
