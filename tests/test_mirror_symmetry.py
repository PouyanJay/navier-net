"""Mirror symmetry about the channel centreline, built into the networks."""

from __future__ import annotations

import pytest
import torch

from naviernet.data.dataset import BubbleDataset
from naviernet.models.geometry import GeometryContext
from naviernet.models.pinn import BubblePINN
from naviernet.training import _geometry_priors, train

from .conftest import staged_capsule_run

SHARP = ["model=stage_b", "model.front_geometry=true", "model.sharp_interface=true"]
MIRROR = [*SHARP, "model.mirror_symmetry=true"]
THERMAL = [
    "model.thermal_model=sharp",
    "experiment.T_inlet_C=56.6",
    "experiment.T_init_C=58.0",
]


def _model(tmp_path, overrides):
    cfg, paths = staged_capsule_run(tmp_path, overrides)
    data = BubbleDataset(cfg, paths)
    torch.manual_seed(0)
    return cfg, data, BubblePINN(cfg, geometry=_geometry_priors(cfg, data))


def _mirror_pair(data, model, n: int = 48):
    torch.manual_seed(1)
    d = data.domain
    x = torch.tensor([d.x_min, d.y_min, d.t_min]) + torch.rand(n, 3) * torch.tensor(
        [d.x_max - d.x_min, d.y_max - d.y_min, d.t_max - d.t_min]
    )
    reflected = x.clone()
    reflected[:, 1] = 2.0 * model.mirror_axis - x[:, 1]
    return x, reflected


def test_every_accessor_is_exactly_even_or_odd_about_the_axis(tmp_path):
    _, data, model = _model(tmp_path, [*MIRROR, *THERMAL])
    x, reflected = _mirror_pair(data, model)

    u, v = model.velocity(x)
    u_r, v_r = model.velocity(reflected)

    assert torch.allclose(u, u_r, atol=1e-6) and torch.allclose(v, -v_r, atol=1e-6)
    assert torch.allclose(model.alpha(x), model.alpha(reflected), atol=1e-6)
    assert torch.allclose(model.pressure(x), model.pressure(reflected), atol=1e-6)
    assert torch.allclose(model.temperature(x), model.temperature(reflected), atol=1e-6)
    assert torch.allclose(model.source(x), model.source(reflected), atol=1e-6)


def test_the_legacy_temperature_is_even_too(tmp_path):
    _, data, model = _model(tmp_path, MIRROR)
    x, reflected = _mirror_pair(data, model)

    assert torch.allclose(model.temperature(x), model.temperature(reflected), atol=1e-6)


def test_the_centreline_is_pinned_to_the_axis_and_the_root_hangs_from_it(tmp_path):
    _, data, model = _model(tmp_path, MIRROR)
    geometry = model.nets["phi"]
    xi = torch.linspace(0.0, 1.0, 11).reshape(-1, 1)
    t = torch.full_like(xi, float(data.t[2]))

    centre = geometry.centerline(xi, t, GeometryContext())

    assert model.mirror_axis == pytest.approx(0.5 * (data.y_lo + data.y_hi))
    assert torch.allclose(centre, torch.full_like(centre, model.mirror_axis))
    assert geometry.priors.y_root == pytest.approx(model.mirror_axis)


def test_without_the_flag_the_fields_are_not_symmetric(tmp_path):
    _, data, model = _model(tmp_path, SHARP)
    torch.manual_seed(1)
    x = torch.rand(48, 3)
    reflected = x.clone()
    reflected[:, 1] = (data.y_lo + data.y_hi) - x[:, 1]

    assert model.mirror_axis is None
    assert not torch.allclose(model.velocity(x)[0], model.velocity(reflected)[0])


def test_mirror_symmetry_requires_the_front_geometry(tmp_path):
    with pytest.raises(ValueError, match="front_geometry"):
        _model(tmp_path, ["model=stage_b", "model.mirror_symmetry=true"])


def test_a_mirrored_run_trains_and_keeps_the_symmetry(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*MIRROR, "training.log_every=1"])

    model, data, state = train(cfg, paths)

    assert all(torch.isfinite(torch.tensor(v)).item() for v in state["hist"][-1].values())
    x, reflected = _mirror_pair(data, model)
    assert torch.allclose(model.velocity(x)[1], -model.velocity(reflected)[1], atol=1e-6)
