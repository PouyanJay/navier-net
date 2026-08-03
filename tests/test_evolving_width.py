"""Evolving width: a shape that keeps deforming past the data.

The reported symptom: frames 10 and 11 render almost the same interface. They
are the two frames beyond supervision, and the reason is that the width profile
is a raw tanh MLP in `t` -- past the trained range tanh saturates and the output
drifts back toward its bias, the near-flat initial profile. Measured on
sweep-20260802-235209-s2, the model's neck deepens +0.075/frame through the last
supervised frame, then rolls off (+0.048, +0.029, +0.013) and REVERSES: by
t* = 4.7 it is un-necking at -0.028/frame. The bubble heals. The real one runs
away and detaches.

R3 already solved this for the NOSE by making it structural -- `s = s0 + integral
of a rate`, which continues at its last learned rate instead of decaying. The
width never got the same treatment. Here it does: the profile deforms along a
learned direction at a rate tied to the bubble's own elongation, so off-data it
keeps going the way it was going.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import staged_run as _staged_run

GEO = ["model.front_geometry=true"]
EVOLVING = [*GEO, "model.evolving_width=true"]


def _geometry(tmp_path, overrides):
    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths

    cfg, paths = _staged_run(tmp_path, overrides)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    return model.nets["phi"], data, cfg


def _profile(geo, t, n=9):
    u = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    with torch.no_grad():
        return geo.half_width(u, torch.full_like(u, float(t))).squeeze(1)


def test_evolving_width_is_off_by_default(tmp_path):
    _, _, cfg = _geometry(tmp_path, GEO)
    assert cfg.model.evolving_width is False


def test_the_profile_starts_at_the_measured_half_width(tmp_path):
    """Same data-anchored start as the raw net: the deformation begins at zero,
    so the construction still opens as the measured first-frame capsule."""
    geo, data, _ = _geometry(tmp_path, EVOLVING)
    profile = _profile(geo, data.domain.t_min)
    assert float(profile.mean()) == pytest.approx(data.initial_half_width, rel=0.25)


def test_the_deformation_keeps_going_instead_of_reversing(tmp_path):
    """The bug, as a test. Drive a deformation direction and follow the profile
    far past the training window: with the raw net it turns around and decays
    back toward its bias; here it must keep moving the same way."""
    geo, data, _ = _geometry(tmp_path, EVOLVING)
    with torch.no_grad():  # a definite direction to travel in
        geo.width_rate[-1].weight.zero_()
        geo.width_rate[-1].bias.fill_(-0.6)

    span = data.domain.t_max - data.domain.t_min
    widths = [float(_profile(geo, data.domain.t_min + k * span).mean()) for k in range(5)]
    steps = [b - a for a, b in zip(widths, widths[1:], strict=False)]

    assert all(s < 0 for s in steps), f"the profile reversed direction: {widths}"
    assert widths[-1] < widths[0], "and it must end up further along, not back home"


def test_the_deformation_can_reach_zero_width_so_the_neck_can_pinch(tmp_path):
    """After the last frame the real bubble splits at its neck. A profile whose
    deformation saturates at a positive width could never do that; driven far
    enough, this one must reach zero."""
    geo, data, _ = _geometry(tmp_path, EVOLVING)
    with torch.no_grad():
        geo.width_rate[-1].weight.zero_()
        geo.width_rate[-1].bias.fill_(-4.0)

    far = data.domain.t_min + 20.0 * (data.domain.t_max - data.domain.t_min)
    assert float(_profile(geo, far).max()) < 0.02, "the width must be able to vanish"


def test_the_shape_still_varies_along_the_bubble(tmp_path):
    """The deformation is one direction in time, but that direction is a full
    function of u -- so a neck (a dip at mid-bubble) stays expressible."""
    geo, data, _ = _geometry(tmp_path, EVOLVING)
    with torch.no_grad():
        geo.width_rate[-1].weight.normal_(0.0, 2.0)
        geo.width_rate[-1].bias.fill_(0.0)

    profile = _profile(geo, data.domain.t_max, n=17)
    assert float(profile.max() - profile.min()) > 1e-3, (
        "the profile must still be free to vary along the bubble"
    )


def test_the_curvature_stays_twice_differentiable(tmp_path):
    """The jump condition differentiates the profile twice, so the new form has
    to stay smooth in u -- a lookup or a piecewise interpolation would not."""
    geo, _, _ = _geometry(tmp_path, EVOLVING)
    front = geo.front(torch.tensor([[0.4]]), n_body=16, n_cap=6)
    assert torch.isfinite(front.kappa_par).all()
    front.kappa_par.sum().backward()
    assert geo.width_rate[-1].weight.grad is not None


def test_the_checkpoint_refuses_a_mismatch(tmp_path):
    """It replaces the width net with two, so the state dict differs and a
    mismatched invocation must fail loudly rather than half-load."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, EVOLVING)
    train(cfg, paths)
    assert torch.load(paths.checkpoint, map_location="cpu", weights_only=False)[
        "evolving_width"
    ]

    plain, _ = _staged_run(tmp_path, GEO)
    with pytest.raises(ValueError, match="evolving_width"):
        load_model(plain, paths)
