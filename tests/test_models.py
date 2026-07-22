"""Network architecture: shapes, invariants, and checkpoint round-tripping."""

from __future__ import annotations

import pytest
import torch

from naviernet.models.layers import AdaptiveTanh, FourierFeatures
from naviernet.models.pinn import BubblePINN, FieldNet


def test_fourier_features_double_the_width():
    ff = FourierFeatures(in_dim=3, n_feats=16, scale=2.0)
    out = ff(torch.rand(7, 3))

    assert ff.out_dim == 32
    assert out.shape == (7, 32)


def test_fourier_features_are_bounded():
    """sin and cos, so the embedding cannot blow up however large the input."""
    out = FourierFeatures(3, 8, 50.0)(torch.rand(32, 3) * 1e3)
    assert out.abs().max() <= 1.0 + 1e-6


def test_fourier_matrix_is_a_buffer_not_a_parameter():
    """B must travel with the checkpoint but never be trained."""
    ff = FourierFeatures(3, 8)
    assert "B" in dict(ff.named_buffers())
    assert list(ff.parameters()) == []


def test_adaptive_tanh_starts_as_plain_tanh():
    x = torch.randn(5, 4)
    assert torch.allclose(AdaptiveTanh(4)(x), torch.tanh(x))


@pytest.mark.parametrize(("nodewise", "expected"), [(True, (4,)), (False, (1,))])
def test_adaptive_tanh_slope_shape(nodewise, expected):
    assert AdaptiveTanh(4, nodewise=nodewise).a.shape == expected


def test_field_net_maps_points_to_scalars(tiny_cfg):
    assert FieldNet(tiny_cfg)(torch.rand(11, 3)).shape == (11, 1)


def test_pinn_exposes_the_configured_fields(tiny_cfg):
    model = BubblePINN(tiny_cfg)
    assert model.fields == ["phi", "u", "v", "s"]


def test_alpha_is_bounded_by_construction(tiny_cfg):
    """alpha = sigmoid(phi/eps) can never leave (0, 1), however extreme phi is."""
    model = BubblePINN(tiny_cfg)
    alpha = model.alpha(torch.rand(64, 3) * 100 - 50)

    assert torch.all(alpha > 0.0) and torch.all(alpha < 1.0)


def test_velocity_returns_two_components(tiny_cfg):
    u, v = BubblePINN(tiny_cfg).velocity(torch.rand(9, 3))
    assert u.shape == (9, 1) and v.shape == (9, 1)


def test_missing_stage_b_field_raises_a_helpful_error(tiny_cfg):
    model = BubblePINN(tiny_cfg)
    with pytest.raises(KeyError, match="cfg.model.fields"):
        model.temperature(torch.rand(4, 3))


def test_stage_b_fields_can_be_enabled_by_config(tiny_cfg):
    model = BubblePINN(tiny_cfg, fields=["phi", "u", "v", "s", "p", "T"])
    assert model.pressure(torch.rand(3, 3)).shape == (3, 1)
    assert model.temperature(torch.rand(3, 3)).shape == (3, 1)


def test_checkpoint_round_trips_exactly(tiny_cfg, tmp_path):
    """A reloaded model reproduces the original's output bit for bit."""
    original = BubblePINN(tiny_cfg)
    x = torch.rand(16, 3)
    expected = original.alpha(x)

    checkpoint = tmp_path / "ckpt.pt"
    torch.save(original.state_dict(), checkpoint)

    restored = BubblePINN(tiny_cfg)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))

    assert torch.equal(restored.alpha(x), expected)


def test_seeding_makes_initialisation_reproducible(tiny_cfg):
    x = torch.rand(8, 3)

    torch.manual_seed(0)
    first = BubblePINN(tiny_cfg).alpha(x)
    torch.manual_seed(0)
    second = BubblePINN(tiny_cfg).alpha(x)

    assert torch.equal(first, second)
