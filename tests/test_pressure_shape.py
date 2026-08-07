"""Pressure-driven shape: the width profile's modes SOLVED from the pressure.

The soft version of this is measured and understood -- raising
`training.weights.laplace` halves the jump error and costs ~0.015 IoU. That is
what a PENALTY does: it trades physics against pixels. This is the other
mechanism, where the pressure determines the shape and a bounded learned term
carries only what the depth-averaged model cannot.

These tests cover the construction: that the licence is bounded, that it reaches
every consumer of the shape identically, and that an unsolved call is the
construction it always was.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import staged_run as _staged_run

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
SHARP = ["model=stage_b", "model.front_geometry=true", "model.sharp_interface=true"]


def _geo(seed: int = 0, **kwargs):
    from naviernet.models.geometry import GeometricInterface, GeometryPriors

    torch.manual_seed(seed)
    return GeometricInterface(GeometryPriors(**PRIORS), **kwargs)


def _bind(geo, coeffs, times=(0.2, 0.6)):
    """Bind solved modes the way the trainer does, once per step."""
    geo.bind_shape(
        torch.tensor(times).reshape(-1, 1), torch.tensor(coeffs, dtype=torch.float32)
    )
    return geo


# --- the modes themselves -----------------------------------------------------


def test_the_modes_can_move_both_ends():
    """The ends are where the caps are. A basis that vanished there could not move
    a head or a foot, which is most of the shape the pressure has an opinion on."""
    from naviernet.models.geometry import SHAPE_MODES, shape_modes

    u = torch.tensor([[0.0], [0.5], [1.0]])
    modes = shape_modes(u)
    assert modes.shape == (3, SHAPE_MODES)
    assert float(modes[0].abs().min()) > 0.5, "the root end must be reachable"
    assert float(modes[2].abs().min()) > 0.5, "the nose end must be reachable"


# --- the construction ---------------------------------------------------------


def test_off_by_default(tmp_path):
    cfg, _ = _staged_run(tmp_path)
    assert cfg.model.pressure_shape is False
    assert cfg.model.shape_delta == pytest.approx(0.3)


def test_it_requires_the_front_geometry(tmp_path):
    from naviernet.models.pinn import BubblePINN

    cfg, _ = _staged_run(tmp_path, ["model.pressure_shape=true"])
    with pytest.raises(ValueError, match="front_geometry"):
        BubblePINN(cfg)


def test_it_requires_the_sharp_interface(tmp_path):
    """Without the jump condition there is no pressure comparison to solve from."""
    from naviernet.models.geometry import GeometryPriors
    from naviernet.models.pinn import BubblePINN

    cfg, _ = _staged_run(
        tmp_path, ["model=stage_b", "model.front_geometry=true", "model.pressure_shape=true"]
    )
    with pytest.raises(ValueError, match="sharp_interface"):
        BubblePINN(cfg, geometry=GeometryPriors(**PRIORS))


@pytest.mark.parametrize("bad", [-0.1, 1.0, 4.0])
def test_it_rejects_a_licence_outside_its_bound(bad):
    """At 1 the profile can reach zero width; the bound is the rail that stops a
    wrong pressure producing a bubble that is not there."""
    with pytest.raises(ValueError, match="shape_delta"):
        _geo(pressure_shape=True, shape_delta=bad)


def test_an_unsolved_call_is_the_construction_it_always_was():
    """The solve happens once per step in the trainer; every call outside that
    window -- figures, diagnostics, evaluation -- must still work, and must be the
    same bubble."""
    on, off = _geo(3, pressure_shape=True), _geo(3)
    x = torch.rand(256, 3)
    assert torch.equal(on(x), off(x))


def test_the_flag_off_ignores_bound_coefficients():
    """Coefficients bound but the feature off is not a shape change. Otherwise a
    stale binding would silently steer a run that never asked for it."""
    off = _geo(5)
    u = torch.linspace(0, 1, 32).reshape(-1, 1)
    t = torch.full_like(u, 0.2)

    before = off._radius(u, t)
    _bind(off, [[2.0, -1.0, 0.5, 0.0], [1.0, 1.0, -1.0, 0.5]])
    assert torch.equal(off._radius(u, t), before)


def test_clearing_the_binding_restores_the_learned_shape():
    """The trainer binds per step and the shape must not outlive it -- evaluation,
    figures and the diagnostics all run outside that window."""
    geo = _geo(5, pressure_shape=True)
    u = torch.linspace(0, 1, 32).reshape(-1, 1)
    t = torch.full_like(u, 0.2)

    before = geo._radius(u, t)
    _bind(geo, [[3.0, -2.0, 1.0, 0.0], [3.0, -2.0, 1.0, 0.0]])
    assert not torch.allclose(geo._radius(u, t), before, atol=1e-4)

    geo.bind_shape(None, None)
    assert torch.equal(geo._radius(u, t), before)


def test_solved_coefficients_change_the_width_within_their_bound():
    """The licence is real, and it is bounded by shape_delta."""
    delta = 0.3
    geo = _geo(7, pressure_shape=True, shape_delta=delta)
    u = torch.linspace(0, 1, 64).reshape(-1, 1)
    t = torch.full_like(u, 0.2)

    base = geo._radius(u, t)
    _bind(geo, [[4.0, -3.0, 2.0, -1.0], [-4.0, 3.0, -2.0, 1.0]])
    solved = geo._radius(u, t)
    ratio = (solved / base).squeeze(1)

    assert float((ratio - 1.0).abs().max()) > 1e-2, "the pressure must be able to move it"
    assert float((ratio - 1.0).abs().max()) <= delta + 1e-6, "and no further than its licence"
    assert float(solved.min()) > 0.0, "a bounded licence cannot close the bubble"


def test_each_time_takes_its_own_solved_shape():
    """The coefficients are solved per instant; two instants with different
    solutions must give different bubbles."""
    geo = _bind(_geo(11, pressure_shape=True), [[3.0, 0, 0, 0], [-3.0, 0, 0, 0]], (0.2, 0.6))
    u = torch.linspace(0, 1, 32).reshape(-1, 1)

    early = geo._radius(u, torch.full_like(u, 0.2))
    late = geo._radius(u, torch.full_like(u, 0.6))
    assert not torch.allclose(early, late, atol=1e-4)


def test_the_field_and_its_front_take_the_same_solved_shape():
    """The invariant the whole construction rests on. `_radius` is the single
    place the shape is read, so phi, the cap centres and the front samples all
    move together -- if they did not, the data term and the interface conditions
    would be pulling on two different bubbles."""
    geo = _bind(_geo(13, pressure_shape=True), [[3.0, -2.0, 1.0, 0.0]], times=(0.3,))

    front = geo.front(torch.tensor([[0.3]]), n_body=32, n_cap=16)
    assert float(geo(front.points).abs().max()) < 1e-4, "a front sample must lie on phi = 0"
