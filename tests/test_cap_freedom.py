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


# --------------------------------------------------------------------------
# T1 -- the angular modulation: free where it should be, absent where the R3
# guarantees live
# --------------------------------------------------------------------------


def _loud(geo, scale: float = 30.0):
    """Amplify the cap net so its deviation is unmistakable. Initialised, the net
    sits within a hair of the circle on purpose; a test of what the freedom CAN
    do has to open it up first (the same trick the nose-rate regression uses)."""
    with torch.no_grad():
        geo.cap_net[-1].weight.mul_(scale)
        geo.cap_net[-1].bias.add_(0.7)
    return geo


def _front_of(geo, t: float, n_body: int = 24, n_cap: int = 24):
    return geo.front(torch.tensor([[t]]), n_body=n_body, n_cap=n_cap)


def test_the_modulation_is_identically_zero_on_the_body(tmp_path):
    """The gate vanishes at the seam, which is what makes the body untouched --
    and that is what keeps the alpha field and the front samples describing the
    same shape. Points strictly between the two apexes must be unmoved."""
    free, plain = _loud(_free(3)), _geo(3)
    f = free.frame(torch.tensor([[0.4]]))
    ax, bx = float(f.ax.detach()), float(f.bx.detach())

    x = torch.zeros(200, 3)
    x[:, 0] = torch.linspace(ax + 1e-3, bx - 1e-3, 200)  # body only
    x[:, 1] = 0.25 + 0.15 * torch.rand(200)
    x[:, 2] = 0.4

    assert torch.allclose(free(x), plain(x), atol=1e-9)


def test_the_root_apex_stays_exactly_on_the_measured_anchor(tmp_path):
    """The gate is zero at the apex, so the pin the R3 win was built on survives
    the freedom -- on random weights, and far outside the training window."""
    geo = _loud(_free(5))
    for t in (0.0, 0.5, 1.0, 7.3):
        root = geo.root_point(t)
        assert torch.allclose(root[0], torch.tensor(PRIORS["x_root"]), atol=1e-6)
        phi = geo(torch.tensor([[float(root[0]), float(root[1]), t]]))
        assert abs(float(phi.detach())) < 1e-5, f"interface must pass through the root at t={t}"


def test_the_nose_apex_stays_exactly_the_monotone_nose(tmp_path):
    """Same at the far end: the tip is still s(t), so the monotone-nose guarantee
    is untouched and the freed cap cannot smuggle the front forward."""
    free, plain = _loud(_free(5)), _geo(5)
    for t in (0.0, 0.5, 1.0, 7.3):
        tt = torch.tensor([[t]])
        assert torch.allclose(free.nose(tt), plain.nose(tt), atol=1e-9)
        tip = free.nose_point(t)
        assert torch.allclose(tip[0], free.nose(tt).reshape(()), atol=1e-6)
        phi = free(torch.tensor([[float(tip[0]), float(tip[1]), t]]))
        assert abs(float(phi.detach())) < 1e-5


def test_the_cap_is_no_longer_a_circle(tmp_path):
    """The point of the whole exercise: the distance from the cap centre to the
    interface must VARY with angle. On the circular construction it is constant
    to machine precision."""
    free, plain = _loud(_free(11)), _geo(11)

    def cap_radii(geo):
        front = _front_of(geo, 0.6)
        frame = geo.frame(torch.tensor([[0.6]]))
        on_nose = (front.on_cap.squeeze(1) > 0) & (front.u.squeeze(1) == 1.0)
        pts = front.points[on_nose]
        centre = torch.cat(
            [frame.bx, geo.centerline(torch.ones(1, 1), torch.tensor([[0.6]]))], 1
        )
        return (pts[:, :2] - centre).norm(dim=1)

    assert float(cap_radii(plain).std()) < 1e-6, "the baseline cap must be a circle"
    assert float(cap_radii(free).std()) > 1e-3, "the freed cap must not be one"


def test_the_departure_from_the_circle_respects_its_bound(tmp_path):
    """delta is the guarantee the cap cannot fold. Even with the net driven hard,
    the radius must stay within delta of the circle it replaces."""
    delta = 0.2
    geo = _loud(_free(13, delta=delta), scale=200.0)
    front = _front_of(geo, 0.6, n_cap=64)
    frame = geo.frame(torch.tensor([[0.6]]))
    on_nose = (front.on_cap.squeeze(1) > 0) & (front.u.squeeze(1) == 1.0)
    pts = front.points[on_nose]
    centre = torch.cat([frame.bx, geo.centerline(torch.ones(1, 1), torch.tensor([[0.6]]))], 1)

    ratio = (pts[:, :2] - centre).norm(dim=1) / float(frame.r_nose)
    assert float((ratio - 1.0).abs().max()) <= delta + 1e-6


def test_every_front_sample_still_lies_on_the_interface(tmp_path):
    """Field/front consistency, the invariant the whole construction rests on: if
    `forward` and `front` disagreed, the data term and the sharp-interface
    residuals would be pulling on two different bubbles."""
    geo = _loud(_free(17))
    for t in (0.2, 0.9):
        front = _front_of(geo, t, n_body=32, n_cap=32)
        phi = geo(front.points)
        # Measured ~3.6e-6, so 1e-5 is a ~3x margin -- the same margin the apex
        # checks above take. The residual is `forward`'s ABS_SMOOTH floor in the
        # direction it normalises by, which the front's exact cos/sin does not
        # share; it is five orders under alpha_eps and must stay there.
        assert float(phi.abs().max()) < 1e-5, f"front sample off the interface at t={t}"


def test_the_freed_cap_still_closes_one_connected_shape(tmp_path):
    """The topology guarantee: a positive radius everywhere means the vapour
    region cannot split, and delta < 1 is what keeps it positive."""
    geo = _loud(_free(19), scale=200.0)
    front = _front_of(geo, 0.7, n_body=64, n_cap=64)
    frame = geo.frame(torch.tensor([[0.7]]))

    # Every sampled point sits at a strictly positive distance from the spine.
    spine_y = geo.centerline(front.u, front.points[:, 2:3])
    spine_x = frame.ax + front.u * (frame.bx - frame.ax)
    radial = (
        (front.points[:, 0:1] - spine_x) ** 2 + (front.points[:, 1:2] - spine_y) ** 2
    ).sqrt()
    assert float(radial.min()) > 0.0


# --------------------------------------------------------------------------
# T2 -- curvature read off the curve, not asserted as 1/r
# --------------------------------------------------------------------------


def _nose_cap(front):
    return (front.on_cap.squeeze(1) > 0) & (front.u.squeeze(1) == 1.0)


def test_the_circular_cap_still_reports_exactly_one_over_r():
    """The closed form has to survive the rewrite, or the baseline of the bench
    is not the baseline any more."""
    geo = _geo(23)
    front = _front_of(geo, 0.5, n_cap=32)
    frame = geo.frame(torch.tensor([[0.5]]))

    kappa = front.kappa_par[_nose_cap(front)]
    assert torch.allclose(kappa, 1.0 / frame.r_nose.detach(), atol=1e-6)
    assert float(kappa.std()) < 1e-6, "a circle has one curvature"


def test_the_freed_cap_reports_a_curvature_that_varies_along_it():
    """The asserted constant is gone: a cap with a shape has a curvature profile."""
    geo = _loud(_free(23))
    front = _front_of(geo, 0.5, n_cap=32)

    kappa = front.kappa_par[_nose_cap(front)]
    assert float(kappa.std()) > 1e-2, "a shaped cap cannot have one curvature"


def test_the_tip_curvature_is_no_longer_pinned_to_one_over_r():
    """The whole point. `kappa = (r - r'')/r^2` at the apex, so the freed cap can
    present a FLATTER tip than its radius -- which is what an advancing bubble in
    a Hele-Shaw gap actually has, and what the jump condition needs to be able to
    ask for."""
    geo = _loud(_free(29))
    front = _front_of(geo, 0.5, n_cap=65)  # odd, so a sample lands on psi = 0
    frame = geo.frame(torch.tensor([[0.5]]))

    on_cap = _nose_cap(front)
    apex = front.angle[on_cap].abs().argmin()
    kappa_tip = float(front.kappa_par[on_cap][apex])
    circle = float(1.0 / frame.r_nose.detach())

    assert abs(kappa_tip - circle) / circle > 0.05, (
        f"tip curvature {kappa_tip:.3f} is still the circle's {circle:.3f}"
    )


def test_the_outward_normal_follows_the_shaped_cap():
    """A non-circular cap's normal is no longer the radial direction, and that
    normal is what the kinematic condition and the Bretherton capillary number
    are read along -- so it has to follow the curve, not the circle."""
    geo = _loud(_free(31))
    front = _front_of(geo, 0.5, n_cap=32)
    frame = geo.frame(torch.tensor([[0.5]]))

    on_cap = _nose_cap(front)
    normal = front.normal[on_cap]
    centre = torch.cat(
        [frame.bx, geo.centerline(torch.ones(1, 1), torch.tensor([[0.5]]))], dim=1
    ).detach()
    radial = front.points[on_cap][:, :2] - centre
    radial = radial / radial.norm(dim=1, keepdim=True)

    assert torch.allclose(normal.norm(dim=1), torch.ones(int(on_cap.sum())), atol=1e-5)
    # Still outward...
    assert float((normal * radial).sum(dim=1).min()) > 0.0
    # ...but no longer the radial direction itself.
    assert float((normal - radial).norm(dim=1).max()) > 1e-2


def test_the_curvature_comes_from_the_shape_and_not_just_the_radius():
    """The curvature must read the cap's SLOPE and BEND, not only how far out it
    sits. A gradient check alone cannot prove this: the radius is itself a
    function of the cap net, so a `kappa` that had lost its r' and r'' terms and
    collapsed back to 1/r would still be differentiable and still look trained
    through. So assert the thing that separates the two -- kappa is not 1/r."""
    geo = _loud(_free(37))
    front = _front_of(geo, 0.5, n_cap=32)
    frame = geo.frame(torch.tensor([[0.5]]))

    on_cap = _nose_cap(front)
    centre = torch.cat(
        [frame.bx, geo.centerline(torch.ones(1, 1), torch.tensor([[0.5]]))], dim=1
    ).detach()
    radius = (front.points[on_cap][:, :2] - centre).norm(dim=1, keepdim=True)
    kappa = front.kappa_par[on_cap]

    # Where the cap is genuinely shaped, its curvature is nowhere near 1/r.
    assert float((kappa - 1.0 / radius).abs().max()) > 1.0

    # ...and `create_graph` still carries the loss back to the net that set it.
    kappa.sum().backward()
    grads = [p.grad for p in geo.cap_net.parameters() if p.grad is not None]
    assert grads, "no gradient reached the cap net through the curvature"
    assert max(float(g.abs().max()) for g in grads) > 0.0


# --------------------------------------------------------------------------
# T6 -- variants: the freedom has to compose with everything already opt-in
# --------------------------------------------------------------------------


def test_a_vanished_cap_reports_no_curvature_at_all():
    """`allow_pinch` makes the radius SIGNED, so a cap can stop existing. It has
    no curvature to report, and flooring its radius would hand the jump condition
    ~1/ABS_SMOOTH -- an enormous capillary pressure for a point that, by this
    construction's own contract, is not there -- at exactly the pinch the feature
    exists to make trainable.

    `allow_pinch` acts on the WIDTH net, so driving the cap net is not enough to
    reach this branch: the width has to go negative for the cap to vanish."""
    geo = _loud(_free(41, allow_pinch=True), scale=200.0)
    with torch.no_grad():
        geo.width_net[-1].bias.fill_(-8.0)  # every station empty: both caps gone

    frame = geo.frame(torch.tensor([[0.8]]))
    assert float(frame.r_nose.detach()) < 0.0, "the cap must actually have vanished"

    front = _front_of(geo, 0.8, n_cap=32)
    on_cap = front.on_cap.squeeze(1) > 0
    assert torch.equal(front.kappa_par[on_cap], torch.zeros_like(front.kappa_par[on_cap]))
    assert torch.isfinite(front.points).all()
    assert torch.isfinite(front.normal).all()


def test_free_caps_compose_with_a_still_present_pinching_bubble():
    """The other half of `allow_pinch`: a signed radius that has NOT gone negative
    is an ordinary cap, and freeing it changes nothing about that."""
    geo = _loud(_free(41, allow_pinch=True), scale=200.0)
    front = _front_of(geo, 0.8, n_cap=32)

    assert float(geo.frame(torch.tensor([[0.8]])).r_nose.detach()) > 0.0
    assert torch.isfinite(front.kappa_par).all()
    assert torch.isfinite(front.points).all()
    assert torch.isfinite(front.normal).all()


def test_free_caps_compose_with_an_evolving_width():
    """Two shape reparameterisations at once: the width travels with elongation
    while the caps carry their own angular shape."""
    geo = _loud(_free(43, evolving_width=True))
    front = _front_of(geo, 0.8, n_cap=32)

    assert torch.isfinite(front.points).all()
    assert float(front.kappa_par[_nose_cap(front)].std()) > 1e-3


def test_free_caps_survive_a_just_nucleated_bubble():
    """The degenerate frame: a bubble shorter than its own two cap radii, where
    the joint rescale keeps the caps reaching both apexes. The modulation must
    not divide by the vanishing spine."""
    from naviernet.models.geometry import GeometricInterface, GeometryPriors

    torch.manual_seed(47)
    tight = {**PRIORS, "s0": 0.205, "rate0": 0.0, "w0": 0.2}
    geo = _loud(GeometricInterface(GeometryPriors(**tight), cap_freedom=True, cap_delta=0.2))

    front = _front_of(geo, 0.0, n_cap=32)
    assert torch.isfinite(front.points).all()
    assert torch.isfinite(front.kappa_par).all()
    assert float(geo(front.points).abs().max()) < 1e-3


def test_free_caps_carry_the_dataset_conditioning():
    """A joint run binds each dataset's own anchors and conditioning row; the cap
    net takes the row like every other geometry net, so one construction serves
    every condition."""
    from naviernet.models.geometry import GeometryContext

    geo = _loud(_free(53, n_cond=3))
    c = torch.tensor([[0.2, -0.4, 0.9]])
    front = geo.front(torch.tensor([[0.5]]), n_body=8, n_cap=16, ctx=GeometryContext(c=c))

    assert torch.isfinite(front.points).all()
    # A different condition is a different shape, or the row is being ignored.
    other = geo.front(
        torch.tensor([[0.5]]),
        n_body=8,
        n_cap=16,
        ctx=GeometryContext(c=torch.tensor([[-0.7, 0.5, -0.2]])),
    )
    assert not torch.allclose(front.points, other.points, atol=1e-6)


def test_a_freed_cap_reloads_as_the_same_shape(tmp_path):
    """Resume and evaluate rebuild the model from the checkpoint; the cap net has
    to come back with it or the reloaded bubble is a different one."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, TINY_CAP)
    train(cfg, paths)
    model, data, _ = load_model(cfg, paths)

    x = torch.rand(128, 3)
    x[:, 2] = data.domain.t_min + x[:, 2] * (data.domain.t_max - data.domain.t_min)
    before = model.nets["phi"](x)

    again, _, _ = load_model(cfg, paths)
    assert torch.equal(before, again.nets["phi"](x))
