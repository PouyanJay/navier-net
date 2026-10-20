"""The channel's boundaries: the developed inlet profile
(``model.inlet_profile``) and the outlet pressure (``model.outlet_pressure_Pa``)."""

from __future__ import annotations

import math

import pytest
import torch

from naviernet import training
from naviernet.physics.groups import compute_groups
from naviernet.physics.residuals import developed_inlet_profile

from .conftest import make_config, staged_capsule_run

GROUPS = {"hele_shaw": 0.2228, "Re": 215.5}


def test_the_developed_profile_carries_the_flow_rate_and_stops_at_the_walls():
    y = torch.linspace(0.0, 1.0, 20001).reshape(-1, 1)

    u = developed_inlet_profile(y, 0.0, 1.0, 0.15, GROUPS)

    assert float(u.mean()) == pytest.approx(0.15, rel=1e-4), "the mean is the flow rate's"
    assert float(u[0]) == pytest.approx(0.0, abs=1e-6) and float(u[-1]) == pytest.approx(
        0.0, abs=1e-6
    )
    assert torch.allclose(u, u.flip(0), atol=1e-6), "symmetric about the centreline"


def test_the_developed_profile_satisfies_the_straight_channel_law():
    # -dp/dx = C u - (1/Re) u'' is uniform across the width.
    y = torch.linspace(0.1, 0.9, 401, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    u = developed_inlet_profile(y, 0.0, 1.0, 0.15, GROUPS)
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y)[0]

    drive = GROUPS["hele_shaw"] * u - u_yy / GROUPS["Re"]

    assert float(drive.std() / drive.mean()) < 1e-6


def test_its_wall_layer_is_the_depth_over_root_twelve():
    groups = compute_groups(make_config([]))
    k = math.sqrt(groups["hele_shaw"] * groups["Re"])

    assert 1.0 / k == pytest.approx(groups["H_star"] / math.sqrt(12.0), rel=1e-6)


def test_the_outlet_pressure_needs_the_outlet_points(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        ["model=stage_b", "model.outlet_pressure_Pa=0.0", "model.outlet_condition=none"],
    )

    with pytest.raises(ValueError, match="outlet_pressure_Pa"):
        training.train(cfg, paths)


def test_a_run_with_the_new_boundaries_trains(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            "model=stage_b",
            "model.inlet_profile=developed",
            "model.outlet_condition=zero_gradient",
            "model.outlet_pressure_Pa=0.0",
            "training.steps=2",
            "training.log_every=1",
        ],
    )

    _, _, state = training.train(cfg, paths)

    assert math.isfinite(state["hist"][-1]["bc"])
