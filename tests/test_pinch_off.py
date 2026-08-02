"""Pinch-off: letting the bubble detach.

The front geometry guarantees a single connected capsule with a never-retreating
nose. Those guarantees are what fixed R3's extrapolation shape collapse -- and
they are also exactly what makes DETACHMENT inexpressible: `R = y_half
sigmoid(.)` is strictly positive, so `phi` on the spine is >= 0 at every station
and alpha never drops below 0.5 there, whatever the physics asks for. Under
`model.allow_pinch` both guarantees become conditional.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import staged_run as _staged_run

GEO = ["model.front_geometry=true"]
PINCH = [*GEO, "model.allow_pinch=true"]


def _geometry(tmp_path, overrides):
    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths

    cfg, paths = _staged_run(tmp_path, overrides)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    return model, data


def _spine_alpha(model, data, u_values):
    geo = model.nets["phi"]
    t = torch.full((len(u_values), 1), 0.6 * data.domain.t_max)
    u = torch.tensor(u_values, dtype=torch.float32).reshape(-1, 1)
    frame = geo.frame(t)
    points = torch.cat([frame.ax + u * (frame.bx - frame.ax), geo.centerline(u, t), t], dim=1)
    with torch.no_grad():
        return model.alpha(points).squeeze(1)


def test_without_the_flag_the_spine_can_never_leave_the_vapour(tmp_path):
    """The standing guarantee, stated as a test: however the width net is driven,
    alpha on the spine stays >= 0.5, so the capsule cannot split."""
    model, data = _geometry(tmp_path, GEO)
    with torch.no_grad():
        model.nets["phi"].width_net[-1].bias.fill_(-50.0)  # radius driven to ~0
    assert (_spine_alpha(model, data, [0.2, 0.5, 0.8]) >= 0.5).all()


def test_with_the_flag_a_negative_radius_empties_the_vapour(tmp_path):
    """The capability the flag buys: the radius is signed, so phi on the spine
    can go negative and a station can stop being vapour at all."""
    model, data = _geometry(tmp_path, PINCH)
    with torch.no_grad():
        model.nets["phi"].width_net[-1].bias.fill_(-50.0)
    assert (_spine_alpha(model, data, [0.2, 0.5, 0.8]) < 0.5).all()


def test_a_pinched_profile_really_yields_two_components(tmp_path):
    """The point of the exercise, end to end: drive the width profile to a waist
    that crosses zero and the predicted vapour mask separates into two bubbles --
    the detachment the construction previously forbade."""
    from scipy import ndimage

    torch.manual_seed(0)
    model, data = _geometry(tmp_path, PINCH)
    # Late enough that the capsule has grown a body to pinch; the synthetic
    # fixture's bubble is nearly all cap at early times.
    when = float(3.0 * data.domain.t_max)
    _fit_width_profile(model, lambda u: 0.25 * torch.cos(2.0 * torch.pi * u), when)

    _, components = ndimage.label(_alpha_field(model, data, when) > 0.5)
    assert components == 2, f"a pinched capsule must read as two bubbles, got {components}"


def _alpha_field(model, data, when: float, n: int = 240):
    """alpha on a grid dense enough to resolve two lobes, spanning the capsule's
    own extent -- the fixture's pixel grid is a handful of cells wide."""
    geo = model.nets["phi"]
    with torch.no_grad():
        nose = float(geo.frame(torch.tensor([[when]])).s)
    x = torch.linspace(data.domain.x_min, nose * 1.05, n)
    y = torch.linspace(data.domain.y_min, data.domain.y_max, n)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    points = torch.stack(
        [grid_x.reshape(-1), grid_y.reshape(-1), torch.full((n * n,), when)], dim=1
    )
    with torch.no_grad():
        return model.alpha(points).reshape(n, n).numpy()


def test_the_nose_may_retreat_only_when_pinching_is_allowed(tmp_path):
    """A detached parent's front is no longer the daughter's advancing nose, so
    the monotone-nose guarantee stops being true of it. It holds by construction
    without the flag and becomes learnable with it."""
    model, _ = _geometry(tmp_path, GEO)
    pinching, _ = _geometry(tmp_path, PINCH)
    t = torch.linspace(0.0, 2.0, 40).reshape(-1, 1)

    def retreats(geo):
        with torch.no_grad():
            geo.rate_net[-1].bias.fill_(-3.0)  # ask the front to move backwards
            nose = geo.nose(t).squeeze(1)
        return bool(((nose[1:] - nose[:-1]) < -1e-6).any())

    assert not retreats(model.nets["phi"]), "the nose is monotone by construction"
    assert retreats(pinching.nets["phi"]), "and becomes learnable under allow_pinch"


def test_pinching_keeps_the_root_pinned_exactly(tmp_path):
    """The signed radius must not cost the exactness the construction exists for:
    where the bubble is present, the interface still passes through the root."""
    model, data = _geometry(tmp_path, PINCH)
    geo = model.nets["phi"]
    for t in (data.domain.t_min, data.domain.t_max, 2.0 * data.domain.t_max):
        alpha = model.alpha(geo.root_point(t).unsqueeze(0))
        assert float(alpha) == pytest.approx(0.5, abs=1e-6)


def test_pinching_is_off_by_default(tmp_path):
    cfg, _ = _staged_run(tmp_path, GEO)
    assert cfg.model.allow_pinch is False


def _fit_width_profile(model, target, when: float, steps: int = 800):
    """Regress the width net onto a chosen profile, so a specific geometry can be
    tested without stubbing the net's forward pass."""
    geo = model.nets["phi"]
    opt = torch.optim.Adam(geo.width_net.parameters(), lr=5e-3)
    u = torch.linspace(0.0, 1.0, 64).reshape(-1, 1)
    t = torch.full_like(u, when)
    for _ in range(steps):
        opt.zero_grad()
        ((geo.half_width(u, t) - target(u)) ** 2).mean().backward()
        opt.step()
