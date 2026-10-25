"""Options that act once the geometry is frozen: the edge transport dropped,
the continuity weight set, and the curl-free vapour condition kept inside the
bubble."""

from __future__ import annotations

import pytest
import torch

from naviernet import training
from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BubblePINN
from naviernet.physics.residuals import vapour_potential_residual

from .conftest import make_config, staged_capsule_run

SHARP = ["model=stage_b", "model.front_geometry=true", "model.sharp_interface=true"]


def test_the_default_physics_stage_keeps_the_weights():
    weights = {"vof": 1.0, "div": 1.0}

    training._physics_stage_weights(weights, make_config([]).training)

    assert weights == {"vof": 1.0, "div": 1.0}


def test_the_physics_stage_drops_edge_transport_and_sets_continuity():
    tcfg = make_config(
        [
            "training.edge_transport_after_freeze=false",
            "training.continuity_weight_after_freeze=10.0",
        ]
    ).training
    weights = {"vof": 1.0, "div": 1.0, "brinkman": 5.0}

    training._physics_stage_weights(weights, tcfg)

    assert weights == {"vof": 0.0, "div": 10.0, "brinkman": 5.0}


def _model(tmp_path, extra=()):
    cfg, paths = staged_capsule_run(tmp_path, [*SHARP, *extra])
    data = BubbleDataset(cfg, paths)
    torch.manual_seed(0)
    return data, BubblePINN(cfg, geometry=training._geometry_priors(cfg, data))


def test_inside_the_vapour_the_curl_free_condition_is_unchanged_and_off_it_is_zero(tmp_path):
    data, band = _model(tmp_path / "band")
    _, vapour = _model(tmp_path / "vapour", ["model.vapour_potential_region=vapour"])
    vapour.load_state_dict(band.state_dict())
    x = torch.rand(400, 3) * torch.tensor(
        [float(data.x.max()), float(data.y.max()), float(data.t[-1])]
    )
    x.requires_grad_(True)
    inside = (band.alpha(x) > 0.5).squeeze(1)
    assert inside.any() and (~inside).any(), "the sample straddles the interface"

    r_band = vapour_potential_residual(band, x).detach()
    r_vapour = vapour_potential_residual(vapour, x).detach()

    assert torch.allclose(r_vapour[inside], r_band[inside] / band.alpha(x).detach()[inside])
    assert torch.equal(r_vapour[~inside], torch.zeros_like(r_vapour[~inside]))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("model.vapour_potential_region=liquid", "vapour_potential_region"),
        ("training.continuity_weight_after_freeze=-1.0", "continuity_weight_after_freeze"),
    ],
)
def test_an_unsupported_physics_stage_setup_is_refused(tmp_path, override, message):
    cfg, paths = staged_capsule_run(tmp_path, [*SHARP, override])

    with pytest.raises(ValueError, match=message):
        training.train(cfg, paths)
