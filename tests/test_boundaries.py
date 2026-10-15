"""Side walls, momentum closure and outlet: the WP2 domain and boundary choices."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
import torch

from naviernet.data.dataset import BubbleDataset
from naviernet.physics import registry
from naviernet.physics.groups import compute_groups
from naviernet.physics.residuals import (
    boundary_losses,
    brinkman_residuals,
    darcy_residuals,
    outlet_loss,
)
from naviernet.training import _validate_boundaries, train

from .conftest import make_config, staged_run, write_growing_bubble

STAGE_B_FIELDS = ("phi", "u", "v", "s", "p", "T")
WALLS = {"y_wall_lo_star": -0.02, "y_wall_hi_star": 0.52, "y_center_star": 0.25}


class _Liquid:
    """An analytic liquid-only stand-in for the model: given ``(u, v, p)`` of ``x``."""

    def __init__(self, u, v, p) -> None:
        self._u, self._v, self._p = u, v, p

    def alpha(self, x, c=None):
        return torch.zeros_like(x[:, :1])

    def velocity(self, x, c=None):
        return self._u(x), self._v(x)

    def pressure(self, x, c=None):
        return self._p(x)


def _points(n: int = 64) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(n, 3, requires_grad=True)


def test_brinkman_residual_vanishes_on_the_brinkman_poiseuille_profile(cfg):
    """``C u - (1/Re) u_yy = G`` with ``u = (G/C)(1 - cosh(k y')/cosh(k h))``,
    ``k = sqrt(C Re)``: the wall-layer profile a first-order Darcy law cannot hold."""
    groups = compute_groups(cfg)
    drag, re = groups["hele_shaw"], groups["Re"]
    k = (drag * re) ** 0.5
    centre, half = 0.5, 0.5
    gradient = 1.0

    def u(x):
        return (gradient / drag) * (
            1 - torch.cosh(k * (x[:, 1:2] - centre)) / np.cosh(k * half)
        )

    # ``v = 0 * y`` rather than a constant, so the residual's derivatives of ``v``
    # stay connected to ``x`` (autograd refuses a tensor not in the graph).
    liquid = _Liquid(u, lambda x: 0.0 * x[:, 1:2] ** 2, lambda x: -gradient * x[:, 0:1])
    x = _points()

    brinkman = brinkman_residuals(liquid, x, groups)
    darcy = darcy_residuals(liquid, x, groups)

    assert torch.allclose(brinkman.mom_x, torch.zeros_like(brinkman.mom_x), atol=1e-4)
    assert torch.allclose(brinkman.mom_y, torch.zeros_like(brinkman.mom_y), atol=1e-4)
    assert brinkman.mom_x.abs().max() < 1e-2 * darcy.mom_x.abs().max(), (
        "Darcy must NOT hold on this profile: the wall shear is what Brinkman adds"
    )


def test_no_penetration_leaves_the_tangential_wall_velocity_free():
    sliding = _Liquid(
        lambda x: torch.ones_like(x[:, :1]),
        lambda x: torch.zeros_like(x[:, :1]),
        lambda x: torch.zeros_like(x[:, :1]),
    )
    inlet, walls = _points(), _points()

    no_slip = boundary_losses(sliding, inlet, walls, u_inlet=1.0, wall_condition="no_slip")
    no_penetration = boundary_losses(
        sliding, inlet, walls, u_inlet=1.0, wall_condition="no_penetration"
    )

    assert no_slip.item() == pytest.approx(1.0), "u = 1 at a no-slip wall costs u^2"
    assert no_penetration.item() == pytest.approx(0.0), "only v is constrained"


def test_outlet_condition_penalises_streamwise_change_only():
    streamwise = _Liquid(lambda x: x[:, 0:1], lambda x: 0.0 * x[:, 1:2], lambda x: x[:, 0:1])
    transverse = _Liquid(lambda x: x[:, 1:2], lambda x: 0.0 * x[:, 1:2], lambda x: x[:, 0:1])

    assert outlet_loss(streamwise, _points()).item() == pytest.approx(1.0)
    assert outlet_loss(transverse, _points()).item() == pytest.approx(0.0)


def test_closure_selects_darcy_or_brinkman_never_both():
    darcy = {e.id for e in registry.enabled_equations(STAGE_B_FIELDS, sharp_interface=True)}
    brinkman = {
        e.id
        for e in registry.enabled_equations(
            STAGE_B_FIELDS, sharp_interface=True, momentum_closure="brinkman"
        )
    }
    diffuse = {
        e.id for e in registry.enabled_equations(STAGE_B_FIELDS, momentum_closure="brinkman")
    }

    assert "darcy" in darcy and "brinkman" not in darcy
    assert "brinkman" in brinkman and "darcy" not in brinkman
    assert darcy - {"darcy"} == brinkman - {"brinkman"}, "only the momentum law differs"
    assert not {"darcy", "brinkman"} & diffuse, "both are sharp-mode laws"


def _write_with_walls(path) -> None:
    write_growing_bubble(path)
    archive = dict(np.load(path))
    meta = json.loads(str(archive["meta"]))
    meta.update(WALLS)
    archive["meta"] = json.dumps(meta)
    np.savez_compressed(path, **archive)


def test_physical_walls_put_the_boundary_samples_on_the_wall_rows(tmp_path):
    cfg, paths = staged_run(tmp_path, ["model.walls_at=physical"], write=_write_with_walls)
    data = BubbleDataset(cfg, paths)
    rng = np.random.default_rng(0)

    inlet, walls = data.sample_boundary(64, rng)
    outlet = data.sample_outlet(64, rng)
    collocation = data.sample_collocation(4096, rng)

    lo, hi = WALLS["y_wall_lo_star"], WALLS["y_wall_hi_star"]
    assert sorted(np.unique(walls[:, 1].numpy()).tolist()) == pytest.approx([lo, hi])
    assert inlet[:, 1].min() >= lo and inlet[:, 1].max() <= hi
    assert torch.all(outlet[:, 0] == data.domain.x_max) and outlet.requires_grad
    assert collocation[:, 1].min() < data.domain.y_min, "uniform samples reach the wall"


def test_crop_walls_keep_the_boundary_samples_on_the_crop_edges(tmp_path):
    cfg, paths = staged_run(tmp_path, write=_write_with_walls)
    data = BubbleDataset(cfg, paths)

    _, walls = data.sample_boundary(64, np.random.default_rng(0))

    assert sorted(np.unique(walls[:, 1].numpy()).tolist()) == pytest.approx(
        [data.domain.y_min, data.domain.y_max]
    )


def test_physical_walls_refuse_an_archive_without_recorded_walls(tmp_path):
    cfg, paths = staged_run(tmp_path, ["model.walls_at=physical"])

    with pytest.raises(ValueError, match="stage=preprocess"):
        BubbleDataset(cfg, paths)


def test_darcy_with_no_slip_is_warned_as_over_determined(caplog):
    cfg = make_config(
        ["model=stage_b", "model.front_geometry=true", "model.sharp_interface=true"]
    )

    with caplog.at_level(logging.WARNING):
        _validate_boundaries(cfg.model)

    assert "over-determined" in caplog.text


@pytest.mark.parametrize(
    "override",
    ["model.walls_at=ceiling", "model.wall_condition=slippery", "model.outlet_condition=open"],
)
def test_unknown_boundary_choices_fail_at_validation(override):
    cfg = make_config([override])

    with pytest.raises(ValueError, match="must be one of"):
        _validate_boundaries(cfg.model)


def test_a_run_with_the_outlet_condition_trains_to_finite_losses(tmp_path):
    cfg, paths = staged_run(
        tmp_path, ["model.outlet_condition=zero_gradient", "training.log_every=1"]
    )

    _, _, state = train(cfg, paths)

    assert all(np.isfinite(record["bc"]) for record in state["hist"])
