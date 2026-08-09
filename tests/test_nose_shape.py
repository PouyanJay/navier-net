"""The nose-shape journey (`.claude/plans/nose-shape-plan.md`).

T0: the nose measurement. With the root freed and supervised (PR #53), the
NOSE cap is the one remaining piece of the interface that is a hard-coded
perfect circle -- and the measured head is genuinely non-circular, and only
late: the nose-window bluntness drifts 0.61 -> 1.83 through the Series-1
event, crossing 1.0, which means the head OVERHANGS (wider a quarter-window
in than at the window edge -- a mushroomed head no circle can draw at any
radius). These tests pin the measurement itself: exact on shapes whose nose
geometry is known analytically, then the measured table on the real masks.

The machinery is the root's own (`fit_root_window` mirrored), so the nose
module is thin wrappers, and these tests focus on what is NEW: orientation
(the nose is the extent edge FARTHER from the pin), the overhang being
measurable at all, and the `physics.nose` block.
"""

from __future__ import annotations

import numpy as np
import pytest

from naviernet.physics.nose import (
    measured_nose_fit,
    model_nose_window_rms,
)
from naviernet.physics.root import ROOT_WINDOW_FRACTION, Raster, fit_root_window
from tests.conftest import make_config

# --- Analytic masks -----------------------------------------------------------
#
# The same high-resolution grid the root's analytic tests use: pixelization
# error stays far under the effects the assertions are about.

GRID_H, GRID_W = 240, 300
X_MAX, Y_MAX = 1.5, 1.2
XS = np.linspace(0.0, X_MAX, GRID_W)
YS = np.linspace(0.0, Y_MAX, GRID_H)

DISC_CENTER, DISC_RADIUS = (0.7, 0.6), 0.3


def disc_mask() -> np.ndarray:
    """A disc: its nose window lies entirely on a true circle, so the fit must
    read the circle's own signature from the RIGHT edge."""
    gx, gy = np.meshgrid(XS, YS)
    return np.hypot(gx - DISC_CENTER[0], gy - DISC_CENTER[1]) < DISC_RADIUS


def mushroom_mask() -> np.ndarray:
    """A capsule whose head is a wider, shallow bulge: the analytic overhang.

    Body: a horizontal capsule of radius 0.12 from x = 0.2 to x = 0.9. Head:
    the front half of a disc of radius 0.2 about (0.9, 0.6). The widest point
    of the head (0.2) sits INTERIOR to the nose edge while the window's far
    end reads the 0.12 body -- the mushroomed-head regime frames 9-11 measure,
    as a shape whose window profile is known exactly.
    """
    gx, gy = np.meshgrid(XS, YS)
    body_x = np.clip(gx, 0.2, 0.9)
    capsule = np.hypot(gx - body_x, gy - 0.6) < 0.12
    bulge = (np.hypot(gx - 0.9, gy - 0.6) < 0.2) & (gx > 0.86)
    return capsule | bulge


def test_the_disc_nose_reads_as_its_own_circle():
    """The mirror of the root's disc test: measured from the RIGHT edge, the
    same circle signature -- taper ~1/2, tiny residual, sub-1 bluntness."""
    fit = fit_root_window(
        Raster(disc_mask(), XS, YS), y_root=DISC_CENTER[1], root_at_left=False
    )
    assert 0.40 <= fit.taper_exponent <= 0.55
    assert fit.circle_rms < 0.03
    depth = ROOT_WINDOW_FRACTION * 2 * DISC_RADIUS
    expected = np.sqrt((0.25 * depth) * (2 * DISC_RADIUS - 0.25 * depth)) / np.sqrt(
        depth * (2 * DISC_RADIUS - depth)
    )
    assert fit.bluntness == pytest.approx(expected, abs=0.03)


def test_the_mushroom_head_reads_an_overhang():
    """Bluntness ABOVE 1 is the fact no circle can express: the head is wider
    a quarter-window in than at the window's full depth. The measurement must
    see it -- nothing in the fit may assume ratios <= 1. (The whole-window
    taper exponent stays positive here -- the log-log fit spans the tip's
    circular rise too -- which is exactly why the bluntness ratio, not the
    taper, is the overhang's own measure and the supervised scalar.)"""
    fit = fit_root_window(Raster(mushroom_mask(), XS, YS), y_root=0.6, root_at_left=False)
    assert fit.bluntness > 1.2
    # The disc, by contrast, stays safely below 1.
    disc = fit_root_window(
        Raster(disc_mask(), XS, YS), y_root=DISC_CENTER[1], root_at_left=False
    )
    assert disc.bluntness < 0.9


# --- The mushroom fixture and the dataset wrapper -----------------------------

# The mushroom capsule: a THIN body (so the head's overhang has room inside the
# window) closed by a circular root and a bulged nose -- the front slice of a
# wider disc riding on the advancing tip. The analytic mushroomed-head dataset:
# its measured nose bluntness sits WELL above 1 at every frame while its root
# stays circular, which is what makes it both the orientation proof here and
# the T2 mechanism fixture (does the supervision actually move w_nose?).
MUSHROOM_BODY_RADIUS = 0.06
MUSHROOM_BULGE_RADIUS = 0.14
# The bulge keeps only its front slice: everything behind s + this ledge offset
# is cut away, so the head is wide but SHALLOW and the window's full-depth
# station lands on the thin body behind the ledge -- the overhang's signature.
# 0.05 keeps the ledge clear of the earliest frame's full-depth station (at a
# smaller offset that station falls ON the ledge and the ratio flips regime).
MUSHROOM_LEDGE_OFFSET = 0.05
# The fixture's own grid: twice the plain capsule's resolution, because the
# bulge's features (0.06-0.14) are a few pixels at 96x48 and the interpolated
# bluntness turns noisy exactly where the assertions are.
MUSHROOM_W_PX, MUSHROOM_H_PX = 192, 96


def _write_thin_capsule(path, n_frames: int, bulged_nose: bool) -> None:
    """The thin-capsule dataset at the fixture's own fine grid, with or
    without the mushroom head: SDF-exact (away from the corner rings, the
    blunt-root fixture's own accepted caveat) via min/max of primitive SDFs.
    ONE writer for the mechanism pair, so its control differs from its
    treatment by exactly the bulge and nothing else -- resolution included
    (the coarse 96x48 grid quantises the window onto ~6 columns and the
    'measured' targets become pixel noise the loss then chases)."""
    import json

    from tests.conftest import (
        CAPSULE_DT,
        CAPSULE_GROWTH_PER_FRAME,
        CAPSULE_S0,
        CAPSULE_T_REF_MS,
        CAPSULE_X_MAX,
        CAPSULE_X_ROOT,
        CAPSULE_Y_CENTER,
        CAPSULE_Y_MAX,
    )

    xs = np.linspace(0.0, CAPSULE_X_MAX, MUSHROOM_W_PX, dtype=np.float32)
    ys = np.linspace(0.0, CAPSULE_Y_MAX, MUSHROOM_H_PX, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)

    def frame_sdf(nose: float) -> np.ndarray:
        nearest_x = np.clip(gx, CAPSULE_X_ROOT, nose)
        body = np.hypot(gx - nearest_x, gy - CAPSULE_Y_CENTER) - MUSHROOM_BODY_RADIUS
        if not bulged_nose:
            return body
        disc = np.hypot(gx - nose, gy - CAPSULE_Y_CENTER) - MUSHROOM_BULGE_RADIUS
        bulge = np.maximum(disc, (nose + MUSHROOM_LEDGE_OFFSET) - gx)
        return np.minimum(body, bulge)

    sdf = np.stack(
        [frame_sdf(CAPSULE_S0 + k * CAPSULE_GROWTH_PER_FRAME) for k in range(n_frames)]
    )
    alpha = (sdf < 0).astype(np.float32)
    meta = {
        "x_pin_star": CAPSULE_X_ROOT - MUSHROOM_BODY_RADIUS,
        "t_ref_ms": CAPSULE_T_REF_MS,
        "n_frames_usable": n_frames,
        "n_frames_event": n_frames,
        "frame_numbers": list(range(1, n_frames + 1)),
    }
    np.savez_compressed(
        path,
        alpha=alpha,
        sdf=sdf.astype(np.float32),
        valid=np.ones_like(alpha, dtype=np.uint8),
        masks_camera=(alpha > 0.5).astype(np.uint8),
        x_star=xs,
        y_star=ys,
        t_star=(np.arange(n_frames) * CAPSULE_DT).astype(np.float32),
        meta=json.dumps(meta),
    )


def mushroom_capsule_writer(path, n_frames: int = 6) -> None:
    """The mechanism treatment: the thin capsule with the mushroomed head."""
    _write_thin_capsule(path, n_frames, bulged_nose=True)


def thin_capsule_writer(path, n_frames: int = 6) -> None:
    """The mechanism control: the same capsule, same grid, circular nose."""
    _write_thin_capsule(path, n_frames, bulged_nose=False)


def test_measured_nose_fit_reads_the_far_edge_from_the_pin(tmp_path):
    """Orientation: the nose is the x-extent edge FARTHER from the measured pin
    anchor -- the exact complement of the root's rule. On the mushroom fixture
    only the NOSE overhangs, so reading bluntness > 1 at the nose and a
    circular sub-1 root proves the wrapper measures its actual end."""
    from naviernet.data.dataset import BubbleDataset
    from naviernet.physics.nose import measured_nose_edge
    from naviernet.physics.root import measured_root_fit
    from naviernet.utils.paths import RunPaths
    from tests.conftest import staged_run

    cfg, paths = staged_run(tmp_path, write=mushroom_capsule_writer)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    nose = measured_nose_fit(data, row=0)
    root = measured_root_fit(data, row=0)
    assert nose.bluntness > 1.1
    assert root.bluntness < 0.9
    # And the window's anchor ADVANCES with the head, unlike the root's fixed pin.
    edges = [measured_nose_edge(data, row) for row in range(data.n_event)]
    assert all(b > a for a, b in zip(edges, edges[1:], strict=False))


def test_physics_report_carries_the_nose_block(tmp_path):
    """The nose sibling of physics.root: any front-geometry run's metrics now
    measure the nose window, model against masks, per frame."""
    import torch

    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.physics.diagnostics import physics_report
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(tmp_path, ["model.front_geometry=true"])
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    torch.manual_seed(0)
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    report = physics_report(model, data)

    nose = report["nose"]
    assert nose["window_fraction"] == ROOT_WINDOW_FRACTION
    assert len(nose["per_frame"]) == len(data.t)
    frame = nose["per_frame"][0]
    for key in (
        "taper_exponent",
        "circle_rms",
        "bluntness",
        "model_bluntness",
        "model_nose_rms",
    ):
        assert np.isfinite(frame[key]), key
    assert nose["model_nose_rms_last"] >= 0.0
    # The summary carries the drift's endpoints -- the bench gate's numbers.
    assert np.isfinite(nose["bluntness_first"])
    assert np.isfinite(nose["bluntness_last"])


def test_model_nose_rms_reads_the_nose_cap_not_the_root(tmp_path):
    """The distance is normalised by and sampled at the NOSE cap: a model whose
    nose sits exactly on the measured interface reads ~0 whatever the root
    does. The capsule fixture's first frame IS the init capsule's target, so
    the number is finite and bounded by the init mismatch."""
    import torch

    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(tmp_path, ["model.front_geometry=true"])
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    torch.manual_seed(0)
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    rms = model_nose_window_rms(model, data, row=0)
    assert np.isfinite(rms)
    assert 0.0 <= rms < 1.0


# --- Series-1: the measurement this journey is built on -----------------------


@pytest.mark.needs_data
class TestSeries1NoseDrift:
    """Plan §0 fact 1, re-measured by the shipped module: the head starts
    near-circular, blunts through the event, and OVERHANGS by the last frames
    (bluntness crossing 1.0) -- and the facts survive threshold and window
    perturbation (plan §8)."""

    @pytest.fixture()
    def data(self):
        from naviernet.data.dataset import BubbleDataset
        from naviernet.utils.paths import RunPaths

        cfg = make_config(
            ["dataset=Series-1", "training.holdout_frame=-1", "training.val_fraction=0"]
        )
        paths = RunPaths.from_config(cfg)
        if not paths.tensors.exists():
            pytest.skip("Series-1 tensors not preprocessed")
        return BubbleDataset(cfg, paths, device="cpu")

    def test_the_measured_table_reproduces(self, data):
        """The plan's own numbers, by frame: bluntness 0.605 (f1) -> 1.833
        (f11), circle RMS 0.022 -> 0.189."""
        by_frame = {
            int(frame): measured_nose_fit(data, row)
            for row, frame in enumerate(data.frame_numbers)
        }
        assert by_frame[1].bluntness == pytest.approx(0.605, abs=0.05)
        assert by_frame[5].bluntness == pytest.approx(0.982, abs=0.05)
        assert by_frame[11].bluntness == pytest.approx(1.833, abs=0.08)
        assert by_frame[1].circle_rms == pytest.approx(0.022, abs=0.02)
        assert by_frame[11].circle_rms == pytest.approx(0.189, abs=0.03)

    def test_the_overhang_appears_and_the_drift_is_late(self, data):
        """The head is near-circular EARLY and overhangs LATE -- the two facts
        that scope the feature (frames 9-11 are where the model's error is)."""
        fits = [measured_nose_fit(data, row) for row in range(data.n_event)]
        assert fits[0].bluntness < 0.75
        assert fits[-1].bluntness > 1.0  # the overhang no circle can draw
        assert fits[-1].circle_rms > fits[0].circle_rms + 0.05

    def test_the_drift_survives_threshold_and_window_perturbation(self, data):
        for level in (0.4, 0.5, 0.6):
            for window in (0.20, 0.25, 0.30):
                fits = [
                    measured_nose_fit(data, row, window=window, alpha_level=level)
                    for row in range(data.n_event)
                ]
                assert fits[-1].bluntness > fits[0].bluntness + 0.3, (level, window)
                assert fits[-1].bluntness > 1.0, (level, window)


# --- T1: the Hugelschaffer nose cap (`model.nose_shape`) ----------------------
#
# The nose cap stops asserting a circle: the same Hugelschaffer half-oval that
# freed the root, mirrored (sign +1, centre bx), with its OWN trained, bounded,
# time-varying bluntness w_nose(t). The family already expresses the overhang:
# for w > 0 the width's maximum sits INTERIOR to the seam, so the rendered
# bluntness exceeds 1 -- precisely the measured f9-f11 regime. No new
# mathematics; the root/nose difference is a sign and a centre.

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
    import torch

    from naviernet.models.geometry import GeometricInterface, GeometryPriors

    torch.manual_seed(seed)
    return GeometricInterface(GeometryPriors(**PRIORS), **kwargs)


def _force_nose_bluntness(geometry, raw: float) -> None:
    """Drive the w_nose(t) net to a constant raw output, so the rendered
    bluntness is ``nose_delta * tanh(raw)`` at every time -- the structural
    guarantees must hold at ARBITRARY bluntness, not just the trained one."""
    import torch

    with torch.no_grad():
        for p in geometry.nose_net.parameters():
            p.zero_()
        geometry.nose_net[-1].bias.fill_(raw)


def _nose_cap_front(geometry, t: float = 0.4, n_cap: int = 65):
    # n_cap ODD, so the angle grid contains psi = 0 -- the apex itself -- and
    # the exact-apex assertion tests the apex, not a neighbour.
    import torch

    front = geometry.front(torch.tensor([[t]]), n_body=16, n_cap=n_cap)
    on_nose = (front.on_cap.squeeze(1) > 0) & (front.u.squeeze(1) > 0.5)
    return front, on_nose


def test_nose_shape_is_off_by_default(cfg):
    assert cfg.model.nose_shape is False
    assert 0.0 <= cfg.model.nose_delta < 1.0


def test_flag_off_leaves_every_net_bit_identical():
    """The RNG-ordering guarantee (the film's and the root's lesson): the nose
    net is built LAST -- after the root net -- so a seeded flag-off run starts
    from bit-identical weights, INCLUDING a rooted run gaining the nose."""
    import torch

    plain = _geo(7)
    nosed = _geo(7, nose_shape=True)
    plain_state = dict(plain.state_dict())
    nosed_state = dict(nosed.state_dict())
    nose_keys = {k for k in nosed_state if k.startswith("nose_net")}
    assert nose_keys, "the flag must add the w_nose(t) net"
    assert set(nosed_state) - nose_keys == set(plain_state)
    for key in plain_state:
        assert torch.equal(plain_state[key], nosed_state[key]), key

    # And the intended composition: root + nose adds ONLY the nose net on top
    # of the rooted construction, every shared weight bit-identical.
    rooted = _geo(7, nucleation_root=True)
    both = _geo(7, nucleation_root=True, nose_shape=True)
    rooted_state, both_state = dict(rooted.state_dict()), dict(both.state_dict())
    assert set(both_state) - {k for k in both_state if k.startswith("nose_net")} == set(
        rooted_state
    )
    for key in rooted_state:
        assert torch.equal(rooted_state[key], both_state[key]), key


def test_nose_shape_requires_the_front_geometry(tmp_path):
    from naviernet.models.pinn import BubblePINN
    from tests.conftest import staged_run

    cfg, _ = staged_run(tmp_path, ["model.nose_shape=true"])
    with pytest.raises(ValueError, match="front_geometry"):
        BubblePINN(cfg)


def test_nose_shape_rejects_cap_freedom():
    """Two shape systems on one cap: no bench could attribute a result to
    either. cap_freedom itself is untouched -- it stays, exclusive with both
    root and nose."""
    with pytest.raises(ValueError, match="cap_freedom"):
        _geo(0, nose_shape=True, cap_freedom=True)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 2.5])
def test_nose_delta_outside_its_bound_is_rejected(bad):
    """At |w| >= r the Hugelschaffer denominator can reach zero and the apex
    curvature diverges -- checked whatever the flag says, like root_delta."""
    with pytest.raises(ValueError, match="nose_delta"):
        _geo(0, nose_delta=bad)


def test_the_construction_opens_as_the_circle():
    """Zero-bias net -> w_nose = 0 -> the nose cap is the circle it replaces,
    on both routes: the front samples and the field."""
    import torch

    plain = _geo(3)
    nosed = _geo(3, nose_shape=True)
    _force_nose_bluntness(nosed, 0.0)  # exactly w = 0, not the init's ~0.01 hair
    front_p, on_nose = _nose_cap_front(plain)
    front_n, _ = _nose_cap_front(nosed)
    assert torch.allclose(front_p.points[on_nose], front_n.points[on_nose], atol=1e-5)
    assert torch.allclose(front_p.kappa_par[on_nose], front_n.kappa_par[on_nose], atol=1e-3)
    phi = nosed(front_n.points[on_nose]).detach()
    assert float(phi.abs().max()) < 1e-4


@pytest.mark.parametrize("raw", [-2.0, 2.0])
def test_forced_nose_bluntness_keeps_apex_seams_and_finite_curvature(raw):
    """The structural guarantees survive ANY bluntness inside the bound: the
    apex stays EXACTLY the monotone s(t) (which is why fv_apex composes
    unchanged), seams keep meeting the body, curvature stays finite, and the
    field and the front describe ONE shape."""
    import torch

    geometry = _geo(5, nose_shape=True)
    _force_nose_bluntness(geometry, raw)
    t = 0.4
    front, on_nose = _nose_cap_front(geometry, t=t)

    # The apex sample (angle 0) sits exactly on the monotone nose point.
    nose_pts = front.points[on_nose]
    angles = front.angle[on_nose].squeeze(1)
    apex = nose_pts[angles.abs().argmin()]
    anchor = geometry.nose_point(t)
    assert float((apex - anchor).detach().abs().max()) < 1e-6

    # Curvature is finite on the whole nose cap, apex included.
    assert torch.isfinite(front.kappa_par[on_nose]).all()

    # The seam samples meet the body: the cap's half-width at |angle| -> pi/2
    # approaches the body's half-width at u = 1.
    tt = torch.tensor([[t]])
    r_nose = geometry.frame(tt).r_nose.item()
    seam = nose_pts[angles.argmax()]
    centre = geometry.centerline(torch.ones(1, 1), tt).item()
    assert seam[1].item() - centre == pytest.approx(r_nose, abs=1e-2)

    # Field and front describe ONE shape.
    phi = geometry(front.points[on_nose]).detach()
    assert float(phi.abs().max()) < 1e-4


def _rendered_nose_bluntness(geometry, t: float = 0.4) -> float:
    """The RENDERED mask's nose bluntness through T0's own measurement -- the
    negative-w lesson: every gate reads the shape, never the parameter."""
    import torch

    xs = np.linspace(0.0, 1.0, 400)
    ys = np.linspace(0.0, 0.5, 200)
    gx, gy = np.meshgrid(xs, ys)
    pts = torch.tensor(
        np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, t)]),
        dtype=torch.float32,
    )
    with torch.no_grad():
        phi = geometry(pts).reshape(gy.shape[0], gx.shape[1]).numpy()
    fit = fit_root_window(Raster(phi > 0, xs, ys), y_root=PRIORS["y_root"], root_at_left=False)
    return fit.bluntness


def test_positive_w_nose_renders_the_overhang():
    """The reachable regime the journey exists for: forced w_nose > 0 makes
    the RENDERED nose bluntness cross 1 -- the mushroomed head -- through T0's
    own fit, while the circle stays below it."""
    blunt = _geo(11, nose_shape=True)
    _force_nose_bluntness(blunt, 2.0)
    circle = _geo(11)
    b_blunt, b_circle = _rendered_nose_bluntness(blunt), _rendered_nose_bluntness(circle)
    assert b_circle < 1.0
    assert b_blunt > 1.0, f"forced w > 0 must render an overhang, got {b_blunt}"
    assert b_blunt > b_circle + 0.05


def test_root_and_nose_compose_on_their_own_caps():
    """The intended recipe: nucleation_root and nose_shape own DIFFERENT caps,
    so forcing both leaves the root apex pinned, the nose apex monotone, and
    the field agreeing with the front on both caps at once."""
    import torch

    geometry = _geo(9, nucleation_root=True, nose_shape=True)
    with torch.no_grad():
        for p in geometry.root_net.parameters():
            p.zero_()
        geometry.root_net[-1].bias.fill_(2.0)
    _force_nose_bluntness(geometry, 2.0)

    t = 0.4
    front, on_nose = _nose_cap_front(geometry, t=t)
    on_root = (front.on_cap.squeeze(1) > 0) & (front.u.squeeze(1) < 0.5)

    root_angles = front.angle[on_root].squeeze(1)
    root_apex = front.points[on_root][root_angles.abs().argmin()]
    assert float((root_apex - geometry.root_point(t)).detach().abs().max()) < 1e-6
    nose_angles = front.angle[on_nose].squeeze(1)
    nose_apex = front.points[on_nose][nose_angles.abs().argmin()]
    assert float((nose_apex - geometry.nose_point(t)).detach().abs().max()) < 1e-6

    on_caps = front.on_cap.squeeze(1) > 0
    phi = geometry(front.points[on_caps]).detach()
    assert float(phi.abs().max()) < 1e-4
    assert torch.isfinite(front.kappa_par).all()


def test_a_training_step_leaves_the_nose_net_finite(tmp_path):
    """Regression, mirrored from the root: the field's width comparison must
    floor INSIDE its one sqrt -- an intermediate sqrt(width_sq) is exactly
    zero at the apex column, where its gradient is infinite, and 0 * inf fed
    NaN into the bluntness net on the first training step."""
    import torch

    from naviernet.training import load_model, train
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(
        tmp_path, ["model.front_geometry=true", "model.nose_shape=true"]
    )
    train(cfg, paths)
    model, _, _ = load_model(cfg, paths)
    w = model.nets["phi"].nose_bluntness(torch.tensor([[0.0], [0.5]]))
    assert torch.isfinite(w).all()


# --- T2: the supervision channel ----------------------------------------------
#
# The DOF ships WITH its channel, in one task: this exact nose is where
# cap_freedom starved twice on pixel losses (mismatch-AREA scaling against a
# cap-sized knob), and the root journey's distance + ratio channel measurably
# fixed the same failure at the other end. The mechanism test is the bench's
# question in miniature: does a mushroomed-head dataset actually move w_nose?

NOSE_TRAIN = [
    "model.front_geometry=true",
    "model.nose_shape=true",
    "training.nose_supervision=true",
    "training.steps=120",
    "training.lr=0.01",
]


def test_nose_supervision_requires_the_nose_shape(tmp_path):
    from naviernet.training import _validate_training_config
    from tests.conftest import staged_run

    cfg, _ = staged_run(
        tmp_path, ["model.front_geometry=true", "training.nose_supervision=true"]
    )
    with pytest.raises(ValueError, match="nose_shape"):
        _validate_training_config(cfg)


def test_zero_weight_nose_supervision_is_rejected(tmp_path):
    """Both weights at zero measures the nose and then ignores it -- an
    undriven knob wearing a supervision flag, the exact trap this journey's
    design rule exists to forbid."""
    from naviernet.training import _validate_training_config
    from tests.conftest import staged_run

    cfg, _ = staged_run(
        tmp_path,
        [
            "model.front_geometry=true",
            "model.nose_shape=true",
            "training.nose_supervision=true",
            "training.nose_sdf_weight=0",
            "training.nose_bluntness_weight=0",
        ],
    )
    with pytest.raises(ValueError, match="supervises nothing"):
        _validate_training_config(cfg)


def test_the_nose_plan_excludes_held_out_frames(tmp_path):
    from naviernet.data.dataset import BubbleDataset
    from naviernet.physics import nose_supervision
    from naviernet.utils.paths import RunPaths
    from tests.conftest import staged_run

    cfg, _ = staged_run(
        tmp_path,
        ["training.val_fraction=0.2", "training.val_strategy=tail"],
        write=mushroom_capsule_writer,
    )
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    plan = nose_supervision.build_plan(data, cfg, "cpu")
    assert set(plan.rows).isdisjoint(data.validation_rows)
    assert bool(np.isfinite(plan.bluntness.numpy()).all())
    assert bool((plan.depth > 0).all())
    # The mushroom's measured targets sit ABOVE 1 -- the plan must carry the
    # overhang as-is, never clamp it to a circle-legal ratio.
    assert float(plan.bluntness.max()) > 1.2


def _trained_station_ratios(cfg, model) -> list[float]:
    """The model's own quarter-vs-full half-width ratios at the loss's own
    stations -- the RENDERED quantity the mechanism gates read. Never the
    parameter: the root bench's w went negative while rendering a BLUNTER
    shape (the cap radius overshoots the mask's local scale and w compensates
    -- a coordinate, not a contradiction), so a gate on w's value or sign
    would misread exactly the mechanism it exists to check."""
    import torch

    from naviernet.data.dataset import BubbleDataset
    from naviernet.physics import nose_supervision
    from naviernet.physics.root import BLUNTNESS_DEPTH_FRACTION
    from naviernet.utils.paths import RunPaths

    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    plan = nose_supervision.build_plan(data, cfg, "cpu")
    geometry = model.nets["phi"]
    with torch.no_grad():
        anchor = geometry.nose(plan.times)
        shallow = geometry.nose_half_width(
            anchor - BLUNTNESS_DEPTH_FRACTION * plan.depth, plan.times
        )
        deep = geometry.nose_half_width(anchor - plan.depth, plan.times)
        return (shallow / deep).squeeze(1).tolist()


@pytest.mark.slow  # trains 120 Adam steps (~4s)
def test_the_supervision_drives_the_overhang_on_the_mushroom_head(tmp_path):
    """The mechanism question in miniature: a dataset whose head overhangs
    must move w_nose(t) off its zero init AND render the overhang -- the
    undriven-knob trap's regression test at the nose."""
    import torch

    from naviernet.training import load_model, train
    from tests.conftest import staged_run

    cfg, paths = staged_run(tmp_path, NOSE_TRAIN, write=mushroom_capsule_writer)
    train(cfg, paths)
    model, _, _ = load_model(cfg, paths)
    with torch.no_grad():
        w = model.nets["phi"].nose_bluntness(torch.tensor([[0.25]]))
    assert abs(float(w)) > 0.05, f"the supervision left w_nose(t) undriven: {float(w)}"

    ratios = _trained_station_ratios(cfg, model)
    mean_ratio = sum(ratios) / len(ratios)
    assert mean_ratio > 1.0, f"the trained head must overhang, ratios={ratios}"


@pytest.mark.slow  # trains 120 Adam steps (~4s)
def test_a_circular_nose_does_not_learn_an_overhang(tmp_path):
    """The control: on data whose nose IS a circle the same supervision must
    not invent an overhang. The mushroom's own writer minus the bulge -- same
    grid, same body -- so the pair isolates the head's shape. Gated on the
    RENDERED ratio, not the parameter: on this fixture the trained w settles
    near -0.6 while the rendered head faithfully tracks the measured circle
    (r_nose overshoots and w compensates -- the root bench's negative-w
    coordinate story reproduced in miniature, which is exactly why parameter
    gates are forbidden)."""
    from naviernet.training import load_model, train
    from tests.conftest import staged_run

    cfg, paths = staged_run(tmp_path, NOSE_TRAIN, write=thin_capsule_writer)
    train(cfg, paths)
    model, _, _ = load_model(cfg, paths)
    ratios = _trained_station_ratios(cfg, model)
    mean_ratio = sum(ratios) / len(ratios)
    assert mean_ratio < 1.0, f"supervision invented an overhang on a circle: {ratios}"
    assert max(ratios) < 1.08, f"supervision invented an overhang on a circle: {ratios}"


# --- T3: the artifact, end-to-end ---------------------------------------------


def test_evaluate_writes_the_nose_block_for_a_nose_shape_run(tmp_path):
    """A trained nose-shape run's metrics.json carries physics.nose with the
    trained w_nose(t), its budget use, and the per-frame model RMS -- exactly
    what the bench gates read."""
    import json

    from naviernet.evaluation import evaluate
    from naviernet.training import train
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            "model.front_geometry=true",
            "model.nose_shape=true",
            "training.nose_supervision=true",
        ],
    )
    model, data, _ = train(cfg, paths)
    evaluate(cfg, model, data, paths)
    metrics = json.loads(paths.metrics_json.read_text())
    nose = metrics["physics"]["nose"]
    assert "w" in nose["per_frame"][0]
    assert "w_budget_use" in nose
    assert nose["nose_delta"] == cfg.model.nose_delta
    assert np.isfinite(nose["per_frame"][0]["model_nose_rms"])


# --- T4: variant coverage -----------------------------------------------------


def test_checkpoint_refuses_a_nose_shape_mismatch(tmp_path):
    """The flag changes what the weights MEAN (a nose net exists and the cap
    is a different family), so a mismatched invocation must fail loudly."""
    from naviernet.training import load_model, train
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(
        tmp_path, ["model.front_geometry=true", "model.nose_shape=true"]
    )
    train(cfg, paths)
    plain, _ = staged_capsule_run(tmp_path, ["model.front_geometry=true"])
    with pytest.raises(ValueError, match="nose_shape"):
        load_model(plain, paths)


def test_nose_delta_change_is_refused_on_reload(tmp_path):
    """The bound scales the bluntness net's output: the same weights describe
    a different head through a different delta -- like root_delta."""
    from naviernet.training import load_model, train
    from tests.conftest import staged_capsule_run

    on = ["model.front_geometry=true", "model.nose_shape=true"]
    cfg, paths = staged_capsule_run(tmp_path, [*on, "model.nose_delta=0.6"])
    train(cfg, paths)
    other, _ = staged_capsule_run(tmp_path, [*on, "model.nose_delta=0.3"])
    with pytest.raises(ValueError, match="nose_delta"):
        load_model(other, paths)


def test_joint_runs_reject_the_nose_shape_for_now(tmp_path):
    """The supervision plan is single-dataset; a joint w_nose(t) with no
    channel would be exactly the undriven knob. Explicit rejection until it
    is threaded through the joint contexts -- the root's rule, mirrored."""
    from naviernet.training import train
    from tests.conftest import staged_joint_run

    cfg, paths = staged_joint_run(
        tmp_path, ["model.front_geometry=true", "model.nose_shape=true"]
    )
    with pytest.raises(NotImplementedError, match="nose_shape"):
        train(cfg, paths)


def test_a_pre_nose_config_snapshot_still_composes(tmp_path):
    """Run snapshots written before the nose fields existed must keep
    composing (the molar_mass lesson): merging over the structured schema
    fills the defaults, the flag reads off, and the model builds."""
    from omegaconf import OmegaConf

    from naviernet.config.schema import Config

    snapshot = OmegaConf.to_container(make_config([]), resolve=True)
    for key in ("nose_shape", "nose_delta"):
        del snapshot["model"][key]
    for key in (
        "nose_supervision",
        "nose_sdf_weight",
        "nose_bluntness_weight",
        "nose_cap_samples",
    ):
        del snapshot["training"][key]
    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.create(snapshot))
    assert cfg.model.nose_shape is False
    assert cfg.training.nose_supervision is False


@pytest.mark.slow  # trains 2 x 120 Adam steps (~8s)
def test_a_nose_shape_run_resumes_cleanly(tmp_path):
    """Resume must rebuild the same construction and keep training: the plan
    is deterministic, so the resumed run supervises against the identical
    measurement."""
    from naviernet.training import train
    from tests.conftest import staged_run

    cfg, paths = staged_run(tmp_path, NOSE_TRAIN, write=mushroom_capsule_writer)
    train(cfg, paths)
    _, _, state = train(cfg, paths)  # resumes from the checkpoint
    assert state["done"] == 2 * cfg.training.steps


def test_root_and_nose_train_together(tmp_path):
    """The intended recipe: both cap DOFs and both channels in one run. Two
    steps engage everything; both nets come out finite and the metrics carry
    both physics blocks."""
    import torch

    from naviernet.physics.diagnostics import physics_report
    from naviernet.training import train
    from tests.conftest import staged_run

    cfg, paths = staged_run(
        tmp_path,
        [
            "model.front_geometry=true",
            "model.nucleation_root=true",
            "model.nose_shape=true",
            "training.root_supervision=true",
            "training.nose_supervision=true",
        ],
        write=mushroom_capsule_writer,
    )
    model, data, _ = train(cfg, paths)
    geometry = model.nets["phi"]
    times = torch.tensor([[0.0], [0.5]])
    assert torch.isfinite(geometry.root_bluntness(times)).all()
    assert torch.isfinite(geometry.nose_bluntness(times)).all()
    report = physics_report(model, data)
    assert "w" in report["root"]["per_frame"][0]
    assert "w" in report["nose"]["per_frame"][0]


def test_nose_shape_composes_with_the_front_velocity_terms(tmp_path):
    """Plan decision 6, stated as a test rather than assumed: fv_apex
    supervises the apex's 2-D displacement and the Hugelschaffer apex crossing
    is exact for every w, so the fv terms train unchanged beside the shaped
    nose (off_nose_cap keeps excluding the nose from the normal profile)."""
    import torch

    from naviernet.training import train
    from tests.conftest import staged_run

    cfg, paths = staged_run(
        tmp_path,
        [
            "model.front_geometry=true",
            "model.nose_shape=true",
            "training.nose_supervision=true",
            "training.front_velocity=true",
        ],
        write=mushroom_capsule_writer,
    )
    model, _, _ = train(cfg, paths)
    w = model.nets["phi"].nose_bluntness(torch.tensor([[0.25]]))
    assert torch.isfinite(w).all()


# --- T5: the five-fluid code gate ---------------------------------------------
#
# The bench runs on Series-1/FC-72 (the only data), but future series may use
# any of the five fluids, so the CODE must be correct for all of them from day
# one -- the full recipe (root ladder + nose) composes, validates, and takes a
# 2-step training run per fluid, with the wetting provenance intact.

TINY_SHARP_ROOT_NOSE = [
    "model=stage_b",
    "training=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.nucleation_root=true",
    "model.root_attachment=true",
    "training.root_supervision=true",
    "model.nose_shape=true",
    "training.nose_supervision=true",
]


@pytest.mark.parametrize(
    ("fluid_id", "source"),
    [
        ("fc72", "measured"),
        ("fc77", "extrapolated"),
        ("hfe7100", "partial"),
        ("novec649", "extrapolated"),
        ("water", "hysteresis"),
    ],
)
def test_the_five_fluid_code_gate_with_the_nose(tmp_path, fluid_id, source):
    from naviernet.physics.diagnostics import physics_report
    from naviernet.physics.groups import compute_groups
    from naviernet.training import train
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(tmp_path, [*TINY_SHARP_ROOT_NOSE, f"fluid={fluid_id}"])
    model, data, _ = train(cfg, paths)  # 2 steps: the whole ladder engages
    report = physics_report(model, data, compute_groups(cfg))
    assert report["root"]["wetting_source"] == source
    assert np.isfinite(report["root"]["per_frame"][0]["w"])
    assert np.isfinite(report["nose"]["per_frame"][0]["w"])
    assert 0.0 <= report["nose"]["w_budget_use"] <= 1.0
