"""Front geometry (R3): the interface as the parameterized object.

With ``model.front_geometry`` on, phi comes from a geometric construction --
monotone nose, bounded width envelope, bounded centerline -- so a single
connected, root-anchored capsule is the only expressible shape. These tests
drive the real layers: synthetic tensors -> priors -> geometric phi -> trainer
-> checkpoint -> reloaded model.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import staged_run as _staged_run

TINY_GEO = ["model.front_geometry=true"]


def test_front_geometry_trains_and_anchors_the_root_for_all_t(tmp_path):
    """The walking skeleton: a tiny run trains, and the reloaded model's
    interface passes exactly through the root point at every t -- including far
    beyond the training window (the pin, by construction)."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, TINY_GEO)
    train(cfg, paths)
    model, data, _ = load_model(cfg, paths)

    geo = model.nets["phi"]
    for t in (data.domain.t_min, data.domain.t_max, 3.0 * data.domain.t_max + 1.0):
        root = geo.root_point(t)
        alpha = model.alpha(root.unsqueeze(0))
        assert torch.allclose(alpha, torch.tensor([[0.5]]), atol=1e-6), (
            f"interface must pass exactly through the root at t={t}, got {float(alpha)}"
        )


def test_front_geometry_off_keeps_the_raw_field_net(tmp_path):
    """Flag off (the default): phi is the ordinary FieldNet -- no geometry."""
    from naviernet.models.pinn import BubblePINN, FieldNet

    cfg, _ = _staged_run(tmp_path)
    assert cfg.model.front_geometry is False, "front_geometry must default off"

    model = BubblePINN(cfg)
    assert isinstance(model.nets["phi"], FieldNet)


def test_front_geometry_predicts_a_single_component_beyond_the_window(tmp_path):
    """The shape guarantee the level set could not give: even at extrapolated
    times the predicted vapour mask is one connected capsule."""
    from scipy import ndimage

    from naviernet.evaluation import predict_alpha
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, TINY_GEO)
    train(cfg, paths)
    model, data, _ = load_model(cfg, paths)

    for t in (data.domain.t_max, 2.0 * data.domain.t_max):
        mask = predict_alpha(model, data, t, stride=1) > 0.5
        _, n = ndimage.label(mask)
        assert n == 1, f"expected one capsule at t={t}, got {n} components"


def test_front_geometry_rejects_the_hard_pin_combination(tmp_path):
    """The geometry pins the root exactly by construction -- composing it with
    the soft-gate hard_pin is a config error, loudly."""
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, [*TINY_GEO, "model.hard_pin=true"])

    with pytest.raises(ValueError, match="front_geometry"):
        train(cfg, paths)
