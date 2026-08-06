"""Cap freedom: the end caps get a shape.

Without ``model.cap_freedom`` the caps are exact circles -- past the nose apex
the spine parameter clamps, so the field is ``r_nose - |x - b|`` at every step of
training, and the Young-Laplace residual is handed ``kappa = 1/r`` as a given.
The cap is the one region the data cannot reach (a circle cannot comply) and the
physics cannot reshape (its curvature is asserted).

With the flag on, the radius gains a bounded angular modulation whose gate
vanishes at the apex and at the seam. These tests pin down exactly that: the
freedom is real where it should be, and identically absent everywhere the R3
guarantees live.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import staged_run as _staged_run

TINY_GEO = ["model.front_geometry=true"]
TINY_CAP = [*TINY_GEO, "model.cap_freedom=true"]

# Same anchors the front-geometry suite uses, so a cap test and a shape test are
# talking about the same bubble.
PRIORS = dict(
    x_root=0.2,
    y_root=0.25,
    s0=0.5,
    w0=0.06,
    rate0=0.3,
    y_min=0.0,
    y_max=0.5,
    t_min=0.0,
    t_max=1.0,
)


def _geo(seed: int, **kwargs):
    """A random (untrained) GeometricInterface. The structural guarantees must
    hold on ARBITRARY weights, not just on the trained ones."""
    from naviernet.models.geometry import GeometricInterface, GeometryPriors

    torch.manual_seed(seed)
    return GeometricInterface(GeometryPriors(**PRIORS), **kwargs)


def _free(seed: int, delta: float = 0.2, **kwargs):
    return _geo(seed, cap_freedom=True, cap_delta=delta, **kwargs)


# --------------------------------------------------------------------------
# T0 -- the flag reaches the model, is recorded, and is guarded
# --------------------------------------------------------------------------


def test_cap_freedom_defaults_off(tmp_path):
    """The current recipe must stay reproducible, so the new freedom is opt-in."""
    cfg, _ = _staged_run(tmp_path)
    assert cfg.model.cap_freedom is False
    assert cfg.model.cap_delta == pytest.approx(0.2)


def test_cap_freedom_requires_the_front_geometry(tmp_path):
    """There is no cap to free without the geometric construction."""
    from naviernet.models.pinn import BubblePINN

    cfg, _ = _staged_run(tmp_path, ["model.cap_freedom=true"])
    with pytest.raises(ValueError, match="front_geometry"):
        BubblePINN(cfg)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 2.5])
def test_cap_freedom_rejects_a_delta_outside_its_bound(tmp_path, bad):
    """The bound is what stops a cap folding: at delta >= 1 the modulation can
    drive the radius to zero or negative, and the cap self-intersects."""
    from naviernet.models.geometry import GeometryPriors
    from naviernet.models.pinn import BubblePINN

    cfg, _ = _staged_run(tmp_path, [*TINY_CAP, f"model.cap_delta={bad}"])
    with pytest.raises(ValueError, match="cap_delta"):
        BubblePINN(cfg, geometry=GeometryPriors(**PRIORS))


def test_cap_freedom_is_recorded_in_the_checkpoint(tmp_path):
    """The flag changes what the weights MEAN without changing their shape, so a
    mismatched invocation must not silently consume them."""
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, TINY_CAP)
    train(cfg, paths)

    ckpt = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    assert ckpt["cap_freedom"] is True
    assert ckpt["cap_delta"] == pytest.approx(cfg.model.cap_delta)


def test_a_cap_freedom_checkpoint_is_refused_by_a_circular_run(tmp_path):
    """Loading free-cap weights into a circular-cap model is a different shape
    space; it must fail loudly rather than produce a plausible wrong bubble."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, TINY_CAP)
    train(cfg, paths)

    circular, _ = _staged_run(tmp_path, TINY_GEO)
    with pytest.raises(ValueError, match="cap_freedom"):
        load_model(circular, paths)


def test_cap_delta_change_is_refused_on_reload(tmp_path):
    """The magnitude is a VALUE the weights were trained against, like pin_d_ref."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, [*TINY_CAP, "model.cap_delta=0.2"])
    train(cfg, paths)

    other, _ = _staged_run(tmp_path, [*TINY_CAP, "model.cap_delta=0.4"])
    with pytest.raises(ValueError, match="cap_delta"):
        load_model(other, paths)


def test_flag_off_is_byte_identical_to_the_circular_construction():
    """The head-to-head bench is only honest if the baseline is unchanged: with
    the flag off, phi must be EXACTLY what it was, not merely close."""
    off, plain = _geo(7, cap_freedom=False), _geo(7)

    x = torch.rand(256, 3)
    assert torch.equal(off(x), plain(x))


def test_cap_freedom_trains_end_to_end(tmp_path):
    """The walking skeleton: a run with the flag on trains, checkpoints, and
    reloads through the real trainer."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, TINY_CAP)
    train(cfg, paths)
    model, _, _ = load_model(cfg, paths)

    assert model.nets["phi"].cap_freedom is True
