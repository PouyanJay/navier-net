"""Dimensionless groups and PDE residuals.

The group tests pin the published values from the README, so a change to the
fluid properties or the geometry that silently shifts the physics shows up as a
failing test rather than as a quietly different result.
"""

from __future__ import annotations

import pytest
import torch

from naviernet.models.pinn import BubblePINN
from naviernet.physics.groups import (
    compute_groups,
    hydraulic_diameter_m,
    inlet_velocity_m_s,
    reference_time_ms,
)
from naviernet.physics.residuals import (
    boundary_losses,
    gradients,
    interface_indicator,
    source_penalty,
    stage_a_residuals,
)


def test_reference_time(cfg):
    # 300 um / 0.2 m/s = 1.5 ms
    assert reference_time_ms(cfg.scales) == pytest.approx(1.5)


def test_hydraulic_diameter(cfg):
    # 300 x 150 um rectangular duct -> Dh = 200 um
    assert hydraulic_diameter_m(cfg.experiment) * 1e6 == pytest.approx(200.0)


def test_inlet_velocity_from_flow_rate(cfg):
    # 5 mL/hr through 300 x 150 um
    assert inlet_velocity_m_s(cfg.experiment) == pytest.approx(0.03086, rel=1e-3)


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("Re", 215.5),
        ("We", 2.302),
        ("Ca", 0.01068),
        ("Pr", 9.411),
        ("hele_shaw", 0.2228),
        ("bretherton_film_um", 4.875),
    ],
)
def test_published_groups(cfg, group, expected):
    assert compute_groups(cfg)[group] == pytest.approx(expected, rel=1e-3)


def test_peclet_is_the_product_of_reynolds_and_prandtl(cfg):
    groups = compute_groups(cfg)
    assert groups["Pe"] == pytest.approx(groups["Re"] * groups["Pr"])


def test_bond_number_is_small(cfg):
    """Surface tension dominates gravity at this scale -- the premise of the model."""
    assert compute_groups(cfg)["Bond"] < 0.1


def test_gradients_of_a_known_function():
    """d/d(x,y,t) of f = 2x + 3y + 5t is exactly (2, 3, 5)."""
    x = torch.rand(16, 3, requires_grad=True)
    f = (x * torch.tensor([2.0, 3.0, 5.0])).sum(dim=1, keepdim=True)

    f_x, f_y, f_t = gradients(f, x)
    assert torch.allclose(f_x, torch.full_like(f_x, 2.0))
    assert torch.allclose(f_y, torch.full_like(f_y, 3.0))
    assert torch.allclose(f_t, torch.full_like(f_t, 5.0))


def test_interface_indicator_peaks_at_the_interface():
    alpha = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
    weight = interface_indicator(alpha)

    assert weight[2].item() == pytest.approx(1.0)  # alpha = 0.5
    assert weight[0].item() == pytest.approx(0.0)  # pure liquid
    assert weight[4].item() == pytest.approx(0.0)  # pure vapour
    assert torch.all(weight >= 0) and torch.all(weight <= 1)


def test_stage_a_residual_shapes(tiny_cfg):
    model = BubblePINN(tiny_cfg)
    x = torch.rand(32, 3, requires_grad=True)

    residuals = stage_a_residuals(model, x)
    for field in (residuals.vof, residuals.div, residuals.source):
        assert field.shape == (32, 1)
    assert not residuals.interface_weight.requires_grad


def test_divergence_residual_equals_div_u_minus_source(tiny_cfg):
    """r_div = u_x + v_y - s, verified against separately taken gradients."""
    model = BubblePINN(tiny_cfg)
    x = torch.rand(24, 3, requires_grad=True)

    residuals = stage_a_residuals(model, x)
    u, v = model.velocity(x)
    u_x, _, _ = gradients(u, x)
    _, v_y, _ = gradients(v, x)

    assert torch.allclose(residuals.div, u_x + v_y - residuals.source, atol=1e-6)


def test_source_penalty_ignores_sources_on_the_interface(tiny_cfg):
    """A source concentrated exactly at alpha=0.5 costs nothing; away from it, it does."""
    from naviernet.physics.residuals import StageAResiduals

    zero = torch.zeros(8, 1)
    source = torch.ones(8, 1)

    on_interface = StageAResiduals(zero, zero, source, torch.ones(8, 1))
    in_bulk = StageAResiduals(zero, zero, source, torch.zeros(8, 1))

    assert source_penalty(on_interface).item() == pytest.approx(0.0)
    assert source_penalty(in_bulk).item() == pytest.approx(1.0)


def test_boundary_loss_is_zero_for_a_perfect_solution(tiny_cfg):
    """A model outputting exactly the BC values incurs no boundary loss."""
    u_inlet = 0.1542

    class PerfectAtBoundaries:
        def velocity(self, x):
            # u = u_inlet on the inlet batch, 0 on the wall batch; v = 0 always.
            n = x.shape[0]
            return torch.full((n, 1), u_inlet if n == 4 else 0.0), torch.zeros(n, 1)

    inlet = torch.zeros(4, 3)
    walls = torch.zeros(6, 3)
    loss = boundary_losses(PerfectAtBoundaries(), inlet, walls, u_inlet)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
