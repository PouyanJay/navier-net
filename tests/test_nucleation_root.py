"""The nucleation root journey (`.claude/plans/nucleation-root-plan.md`).

T0: the root measurement module. The diagnostics' profile stations deliberately
exclude the caps, so nothing existing can see the measured fact this journey is
built on -- the masks disagree with a circle at the root MORE AND MORE as the
event runs (taper exponent p drifting 0.5 -> 0.38, circle residual 12% -> 32%).
These tests pin the measurement itself: exact on shapes whose root geometry is
known analytically, then directional on the real Series-1 masks.
"""

from __future__ import annotations

import numpy as np
import pytest

from naviernet.physics.root import (
    ROOT_WINDOW_FRACTION,
    fit_root_window,
    measured_root_fit,
)
from tests.conftest import make_config

# --- Analytic masks -----------------------------------------------------------
#
# High-resolution rasters, so pixelization error stays far under the effects the
# assertions are about (a 300-px-wide grid puts it at the percent level).

GRID_H, GRID_W = 240, 300
X_MAX, Y_MAX = 1.5, 1.2
XS = np.linspace(0.0, X_MAX, GRID_W)
YS = np.linspace(0.0, Y_MAX, GRID_H)

DISC_CENTER, DISC_RADIUS = (0.7, 0.6), 0.3


def disc_mask() -> np.ndarray:
    """A disc: its root window lies entirely on a true circle, so the fit must
    read the circle's own signature -- taper exponent 1/2, residual ~0."""
    gx, gy = np.meshgrid(XS, YS)
    return np.hypot(gx - DISC_CENTER[0], gy - DISC_CENTER[1]) < DISC_RADIUS


def slab_mask() -> np.ndarray:
    """A rectangle spanning the disc's extent: the fully squared-off limit.
    Constant width means no taper at all -- the far pole of the bluntness axis."""
    gx, gy = np.meshgrid(XS, YS)
    inside_x = (gx > DISC_CENTER[0] - DISC_RADIUS) & (gx < DISC_CENTER[0] + DISC_RADIUS)
    inside_y = (gy > DISC_CENTER[1] - DISC_RADIUS) & (gy < DISC_CENTER[1] + DISC_RADIUS)
    return inside_x & inside_y


def test_the_disc_reads_as_its_own_circle():
    fit = fit_root_window(disc_mask(), XS, YS, y_root=DISC_CENTER[1])
    # W(d) = sqrt(d (2r - d)) ~ d^{1/2} near the apex. The log-log fit runs over
    # the whole window, where the exponent is slightly below 1/2 on average.
    assert 0.40 <= fit.taper_exponent <= 0.55
    # A circle fit to circle points: residual is pixelization only.
    assert fit.circle_rms < 0.03
    # The analytic half-width ratio at quarter-window vs full-window depth.
    depth = ROOT_WINDOW_FRACTION * 2 * DISC_RADIUS
    expected = np.sqrt((0.25 * depth) * (2 * DISC_RADIUS - 0.25 * depth)) / np.sqrt(
        depth * (2 * DISC_RADIUS - depth)
    )
    assert fit.bluntness == pytest.approx(expected, abs=0.03)
    # A symmetric shape carries no bias and no tilt.
    assert abs(fit.updown_bias) < 0.01
    assert abs(fit.tilt) < 0.05


def test_the_slab_reads_blunter_than_the_disc_on_the_taper_measures():
    """Squared-off = lower taper exponent, higher half-width ratio. The circle
    residual is deliberately NOT asserted here: a window short relative to the
    half-width is radially near-equidistant from a distant centre whatever the
    shape (measured: the slab fits a circle to 0.9% radially), so the taper
    measures are the discriminators and the circle fit is kept as the
    plan-§1-comparable diagnostic only."""
    disc = fit_root_window(disc_mask(), XS, YS, y_root=DISC_CENTER[1])
    slab = fit_root_window(slab_mask(), XS, YS, y_root=DISC_CENTER[1])
    assert slab.taper_exponent < 0.15 < disc.taper_exponent
    assert slab.bluntness > 0.9 > disc.bluntness


def test_the_fit_is_orientation_agnostic():
    """A mirrored mask with the root on the RIGHT must measure the same root:
    the root side is chosen by the anchor, not by an assumed orientation."""
    mirrored = disc_mask()[:, ::-1]
    left = fit_root_window(disc_mask(), XS, YS, y_root=DISC_CENTER[1])
    right = fit_root_window(mirrored, XS, YS, y_root=DISC_CENTER[1], root_at_left=False)
    assert right.taper_exponent == pytest.approx(left.taper_exponent, abs=1e-6)
    assert right.circle_rms == pytest.approx(left.circle_rms, abs=1e-6)
    assert right.bluntness == pytest.approx(left.bluntness, abs=1e-6)


def test_a_vapour_free_mask_fails_loudly():
    with pytest.raises(ValueError, match="no vapour"):
        fit_root_window(np.zeros((GRID_H, GRID_W), dtype=bool), XS, YS, y_root=0.5)


def test_a_window_too_thin_to_fit_reads_nan_not_garbage():
    """Two usable columns cannot support a log-log fit or a circle; the fit says
    so with NaN rather than inventing an exponent."""
    mask = np.zeros((GRID_H, GRID_W), dtype=bool)
    mask[100:140, 10:13] = True  # 3 columns: apex + 2 -- under the minimum
    fit = fit_root_window(mask, XS, YS, y_root=float(YS[120]))
    assert np.isnan(fit.taper_exponent)
    assert np.isnan(fit.circle_rms)


# --- The dataset wrapper and the metrics block --------------------------------


def test_measured_root_fit_reads_the_capsule_fixture(tmp_path):
    """End-to-end over the staged dataset: the analytic capsule's root cap is a
    true circle, and the wrapper must find it through BubbleDataset."""
    from naviernet.data.dataset import BubbleDataset
    from naviernet.utils.paths import RunPaths
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(tmp_path)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    fit = measured_root_fit(data, row=0)
    # The default window (25% of extent) overshoots this short capsule's cap
    # onto its constant-width body, so the log-log slope reads between the
    # circle's 1/2 and the slab's 0 -- the honest full-window value.
    assert 0.15 <= fit.taper_exponent <= 0.45
    assert fit.circle_rms < 0.12
    assert 0.55 <= fit.bluntness <= 0.9


def test_physics_report_carries_the_root_block(tmp_path):
    """The blind spot closed: any front-geometry run's metrics now measure the
    root window, model against masks, per frame."""
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

    root = report["root"]
    assert root["window_fraction"] == ROOT_WINDOW_FRACTION
    assert len(root["per_frame"]) == len(data.t)
    frame = root["per_frame"][0]
    for key in (
        "taper_exponent",
        "circle_rms",
        "bluntness",
        "model_root_rms",
    ):
        assert np.isfinite(frame[key]), key
    # The capsule model opens circular AND the capsule data is circular at the
    # root, so the model-vs-mask distance is small but honest (init mismatch).
    assert root["model_root_rms_last"] >= 0.0


# --- T1: per-fluid wetting conditions (plan §2.9) -----------------------------
#
# Wetting is where the five offered fluids genuinely diverge -- their kinematic
# viscosities nearly coincide, their contact-angle physics does not, and water
# is a different REGIME, not a parameter change. Every closure reads per-fluid
# config; nothing may be an FC-72 literal in code.

# (fluid id, wetting_source, has an evaporating-angle law)
WETTING_EXPECTATIONS = [
    ("fc72", "measured", True),
    ("fc77", "extrapolated", True),
    ("hfe7100", "partial", True),
    ("novec649", "extrapolated", True),
    ("water", "hysteresis", False),
]


@pytest.mark.parametrize(("fluid_id", "source", "has_law"), WETTING_EXPECTATIONS)
def test_every_fluid_carries_cited_or_flagged_wetting_conditions(fluid_id, source, has_law):
    cfg = make_config([f"fluid={fluid_id}"])
    fluid = cfg.fluid
    assert fluid.wetting_source == source
    assert 0.0 <= fluid.theta_rec_deg <= fluid.theta_adv_deg < 91.0
    assert 0.0 <= fluid.theta_e_deg < 91.0
    if has_law:
        assert fluid.theta_ev_coeff > 0
        assert fluid.theta_ev_exp >= 0
    else:
        # Water's path: no power law exists; the hysteresis window carries it.
        assert fluid.theta_ev_coeff is None
        assert fluid.theta_ev_exp is None


@pytest.mark.parametrize(("fluid_id", "source", "has_law"), WETTING_EXPECTATIONS)
def test_wetting_groups_are_finite_ordered_and_in_range_for_every_fluid(
    fluid_id, source, has_law
):
    from naviernet.physics.groups import compute_groups

    groups = compute_groups(make_config([f"fluid={fluid_id}"]))
    lo = groups["theta_app_min_deg"]
    init = groups["theta_app_init_deg"]
    hi = groups["theta_app_max_deg"]
    assert np.isfinite([lo, init, hi]).all(), fluid_id
    assert 0.0 <= lo <= init <= hi <= 90.0, (fluid_id, lo, init, hi)
    # The bounds must leave the unknown ROOM: a collapsed window cannot train.
    assert hi - lo > 1.0, (fluid_id, lo, hi)
    assert groups["wetting_ca_crit"] > 0.0
    assert np.isfinite(groups["wetting_ca_crit"])


def test_water_is_a_regime_change_not_a_parameter_change():
    """Wide hysteresis window and an order-of-magnitude lower Ca: the two facts
    that make water's root physics categorically different (plan §2.9)."""
    from naviernet.physics.groups import compute_groups

    water = make_config(["fluid=water"])
    fc72 = make_config(["fluid=fc72"])
    # The dielectrics' hysteresis window is degrees-wide at most; water's is
    # tens of degrees.
    assert water.fluid.theta_adv_deg - water.fluid.theta_rec_deg > 20.0
    assert fc72.fluid.theta_adv_deg - fc72.fluid.theta_rec_deg < 5.0
    assert compute_groups(water)["Ca"] < 0.2 * compute_groups(fc72)["Ca"]


def test_wetting_groups_tolerate_a_pre_wetting_config_snapshot():
    """Run snapshots written before the wetting fields existed must keep
    composing (the molar_mass lesson, PR #52): the wetting groups are OMITTED,
    never defaulted, and nothing else in the groups is disturbed."""
    from omegaconf import OmegaConf

    from naviernet.config.schema import Config
    from naviernet.physics.groups import compute_groups

    # Composed the way the API composes an OLD run's snapshot: recorded values
    # merged over the structured schema, wetting fields absent -> MISSING.
    snapshot = OmegaConf.to_container(make_config([]), resolve=True)
    for key in (
        "theta_e_deg",
        "theta_adv_deg",
        "theta_rec_deg",
        "theta_ev_coeff",
        "theta_ev_exp",
        "wetting_source",
    ):
        del snapshot["fluid"][key]
    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.create(snapshot))
    groups = compute_groups(cfg)
    assert "theta_app_init_deg" not in groups
    assert "wetting_ca_crit" not in groups
    assert groups["Ca"] > 0  # the rest of the groups are untouched


# --- T2 (R0): the receding Bretherton branch ----------------------------------
#
# A rear/receding cap is FLATTER than static -- kappa R = 1 + beta (3Ca)^{2/3}
# with beta_rear = -1.13 (Bretherton 1961; Balestra, Zhu & Gallaire 2018; the
# coefficient as quoted by Lu et al. 2007) -- and until now the model clamped
# every receding section to the static 2/H*. At our Ca that is an O(10%)
# curvature overstatement on exactly the receding root body.


def _groups_fc72() -> dict:
    from naviernet.physics.groups import compute_groups

    return compute_groups(make_config(["dataset=Series-1"]))


def test_receding_cap_is_off_by_default(cfg):
    assert cfg.model.receding_cap is False


def test_receding_sections_read_below_static_advancing_unchanged():
    import torch

    from naviernet.physics.residuals import gap_curvature

    groups = _groups_fc72()
    static = 2.0 / groups["H_star"]
    speeds = torch.tensor([[-1.0], [-0.3], [0.0], [0.5], [1.0]])

    off = gap_curvature(speeds, groups)
    on = gap_curvature(speeds, groups, receding=True)

    # Advancing and stationary samples: bit-identical to the flag-off branch.
    assert torch.equal(on[2:], off[2:])
    # Receding samples: strictly below static, monotone in the receding speed.
    assert float(off[0]) == pytest.approx(static)  # the old clamp-to-static
    assert float(on[0]) < float(on[1]) < static
    # The predicted magnitude at our regime: ~10% below static at v_n = -1
    # (1.13 * (3 * 0.0107)^{2/3} = 0.114).
    assert 0.05 < 1.0 - float(on[0]) / static < 0.20


def test_a_fast_receding_section_floors_at_zero_not_negative():
    import torch

    from naviernet.physics.residuals import gap_curvature

    groups = dict(_groups_fc72(), Ca=5.0)  # absurd Ca to drive the bracket down
    kappa = gap_curvature(torch.tensor([[-10.0]]), groups, receding=True)
    assert float(kappa) == 0.0


def test_receding_cap_requires_the_sharp_interface(tmp_path):
    from naviernet.models.pinn import BubblePINN

    cfg = make_config(["model=stage_b", "model.receding_cap=true"])
    with pytest.raises(ValueError, match="receding_cap"):
        BubblePINN(cfg)


# --- T3: the Hugelschaffer root cap (`model.nucleation_root`) -----------------
#
# The capsule's root cap stops asserting a circle: it becomes the Hugelschaffer
# half-oval with ONE trained, bounded, time-varying bluntness parameter w(t).
# w = 0 IS the circle, the apex crossing is exact for every w (the pin survives
# structurally), and the seams keep meeting the body -- so freedom is spent
# exactly where the masks demand it and nowhere else.

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


def _force_bluntness(geometry, raw: float) -> None:
    """Drive the w(t) net to a constant raw output, so the rendered bluntness
    is ``root_delta * tanh(raw)`` at every time -- the structural guarantees
    must hold at ARBITRARY bluntness, not just at the trained one."""
    import torch

    with torch.no_grad():
        for p in geometry.root_net.parameters():
            p.zero_()
        geometry.root_net[-1].bias.fill_(raw)


def _root_cap_front(geometry, t: float = 0.3, n_cap: int = 65):
    # n_cap ODD, so the angle grid contains psi = 0 -- the apex itself -- and
    # the exact-pin assertion tests the apex, not a neighbour.
    import torch

    front = geometry.front(torch.tensor([[t]]), n_body=16, n_cap=n_cap)
    on_root = (front.on_cap.squeeze(1) > 0) & (front.u.squeeze(1) < 0.5)
    return front, on_root


def test_nucleation_root_is_off_by_default(cfg):
    assert cfg.model.nucleation_root is False
    assert 0.0 <= cfg.model.root_delta < 1.0


def test_flag_off_leaves_every_net_bit_identical():
    """The RNG-ordering guarantee (the film's lesson): the root net is built
    LAST, so a seeded flag-off run starts from bit-identical weights."""
    import torch

    plain = _geo(7)
    rooted = _geo(7, nucleation_root=True)
    plain_state = dict(plain.state_dict())
    rooted_state = dict(rooted.state_dict())
    root_keys = {k for k in rooted_state if k.startswith("root_net")}
    assert root_keys, "the flag must add the w(t) net"
    assert set(rooted_state) - root_keys == set(plain_state)
    for key in plain_state:
        assert torch.equal(plain_state[key], rooted_state[key]), key


def test_nucleation_root_requires_the_front_geometry(tmp_path):
    from naviernet.models.pinn import BubblePINN
    from tests.conftest import staged_run

    cfg, _ = staged_run(tmp_path, ["model.nucleation_root=true"])
    with pytest.raises(ValueError, match="front_geometry"):
        BubblePINN(cfg)


def test_nucleation_root_rejects_cap_freedom():
    """Two shape systems on one cap would fight over the same boundary with no
    way to attribute a bench result to either."""
    with pytest.raises(ValueError, match="cap_freedom"):
        _geo(0, nucleation_root=True, cap_freedom=True)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 2.5])
def test_root_delta_outside_its_bound_is_rejected(bad):
    """At |w| >= r the Hugelschaffer denominator can reach zero and the apex
    curvature diverges -- checked whatever the flag says, like cap_delta."""
    with pytest.raises(ValueError, match="root_delta"):
        _geo(0, root_delta=bad)


def test_the_construction_opens_as_the_circle():
    """Zero-bias net -> w = 0 -> the root cap is the circle it replaces, on
    both routes: the front samples and the field."""
    import torch

    plain = _geo(3)
    rooted = _geo(3, nucleation_root=True)
    _force_bluntness(rooted, 0.0)  # exactly w = 0, not the init's ~0.01 hair
    front_p, on_root = _root_cap_front(plain)
    front_r, _ = _root_cap_front(rooted)
    assert torch.allclose(front_p.points[on_root], front_r.points[on_root], atol=1e-5)
    assert torch.allclose(front_p.kappa_par[on_root], front_r.kappa_par[on_root], atol=1e-3)
    # And the field agrees with its own front: every sample sits on phi = 0.
    phi = rooted(front_r.points[on_root])
    assert float(phi.abs().max()) < 1e-4


@pytest.mark.parametrize("raw", [-2.0, 2.0])
def test_forced_bluntness_keeps_pin_seams_and_finite_curvature(raw):
    """The structural guarantees survive ANY bluntness inside the bound: exact
    apex, seams meeting the body, finite curvature everywhere, and one shape
    described by both the field and the front."""
    import torch

    geometry = _geo(5, nucleation_root=True)
    _force_bluntness(geometry, raw)
    t = 0.4
    front, on_root = _root_cap_front(geometry, t=t)

    # The apex sample (angle 0) sits exactly on the pinned root point.
    root_pts = front.points[on_root]
    angles = front.angle[on_root].squeeze(1)
    apex = root_pts[angles.abs().argmin()]
    anchor = geometry.root_point(t)
    assert float((apex - anchor).abs().max()) < 1e-6

    # Curvature is finite on the whole root cap, apex included.
    kappa = front.kappa_par[on_root]
    assert torch.isfinite(kappa).all()

    # The seam samples meet the body: the cap's width at |angle| -> pi/2
    # approaches the body's half-width at u = 0.
    tt = torch.tensor([[t]])
    r_root = geometry.frame(tt).r_root.item()
    seam = root_pts[angles.argmax()]
    centre = geometry.centerline(torch.zeros(1, 1), tt).item()
    assert seam[1].item() - centre == pytest.approx(r_root, abs=1e-2)

    # Field and front describe ONE shape.
    phi = geometry(front.points[on_root])
    assert float(phi.abs().max()) < 1e-4


def test_positive_w_blunts_the_root_and_lowers_the_apex_curvature():
    """The signed direction the attachment physics predicts: w > 0 inflates the
    width near the apex (blunter, the measured drift) and lowers the apex
    curvature below the circle's 1/r; w < 0 sharpens."""
    import torch

    t = 0.4
    blunt = _geo(11, nucleation_root=True)
    _force_bluntness(blunt, 2.0)
    sharp = _geo(11, nucleation_root=True)
    _force_bluntness(sharp, -2.0)
    circle = _geo(11)

    def apex_kappa(geometry):
        front, on_root = _root_cap_front(geometry, t=t)
        angles = front.angle[on_root].squeeze(1)
        return float(front.kappa_par[on_root][angles.abs().argmin()])

    tt = torch.tensor([[t]])
    inverse_r = 1.0 / float(circle.frame(tt).r_root)
    assert apex_kappa(blunt) < inverse_r < apex_kappa(sharp)

    # And the RENDERED mask reads blunter through T0's own measurement.
    def rendered_bluntness(geometry):
        xs = np.linspace(0.0, 1.0, 400)
        ys = np.linspace(0.0, 0.5, 200)
        gx, gy = np.meshgrid(xs, ys)
        pts = torch.tensor(
            np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, t)]),
            dtype=torch.float32,
        )
        with torch.no_grad():
            phi = geometry(pts).reshape(gy.shape[0], gx.shape[1]).numpy()
        fit = fit_root_window(phi > 0, xs, ys, y_root=PRIORS["y_root"])
        return fit.bluntness

    assert rendered_bluntness(blunt) > rendered_bluntness(circle) + 0.02


def test_a_training_step_leaves_the_bluntness_net_finite(tmp_path):
    """Regression: the field's width comparison must floor INSIDE its one sqrt.
    An intermediate sqrt(width_sq) reaches exactly zero at the apex column,
    where its gradient is infinite, and 0 * inf fed NaN into the bluntness net
    on the very first training step -- the matched-floor lesson, relearned."""
    import torch

    from naviernet.training import load_model, train
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(
        tmp_path, ["model.front_geometry=true", "model.nucleation_root=true"]
    )
    train(cfg, paths)
    model, _, _ = load_model(cfg, paths)
    w = model.nets["phi"].root_bluntness(torch.tensor([[0.0], [0.5]]))
    assert torch.isfinite(w).all()


# --- T4 (R1): the supervision channel -----------------------------------------
#
# The DOF ships WITH its channel, in one task: cap_freedom was benched undriven
# twice, and the diagnosis was that pixel losses scale with mismatch AREA and
# starve a cap-sized knob. These terms scale with DISTANCE (the mask's signed
# distance transform at the predicted root contour) and with a per-frame SCALAR
# (the measured bluntness ratio) -- and the mechanism test here is the R1
# question in miniature: does a blunt-rooted dataset actually move w(t)?


def _blunt_capsule_writer(cut_fraction: float):
    """A capsule whose root cap is CUT FLAT at ``cut_fraction`` of the cap
    depth: the analytic blunt-root fixture. SDF of the intersection of two
    convex sets, exact away from the corner ring."""

    def write(path, n_frames: int = 6) -> None:
        import json

        from tests.conftest import (
            CAPSULE_DT,
            CAPSULE_GROWTH_PER_FRAME,
            CAPSULE_H_PX,
            CAPSULE_RADIUS,
            CAPSULE_S0,
            CAPSULE_T_REF_MS,
            CAPSULE_W_PX,
            CAPSULE_X_MAX,
            CAPSULE_X_ROOT,
            CAPSULE_Y_MAX,
            capsule_sdf,
        )

        xs = np.linspace(0.0, CAPSULE_X_MAX, CAPSULE_W_PX, dtype=np.float32)
        ys = np.linspace(0.0, CAPSULE_Y_MAX, CAPSULE_H_PX, dtype=np.float32)
        cut = CAPSULE_X_ROOT - (1.0 - cut_fraction) * CAPSULE_RADIUS
        gx, _ = np.meshgrid(xs, ys)
        sdf = np.stack(
            [
                np.maximum(
                    capsule_sdf(xs, ys, CAPSULE_S0 + k * CAPSULE_GROWTH_PER_FRAME),
                    cut - gx,
                )
                for k in range(n_frames)
            ]
        )
        alpha = (sdf < 0).astype(np.float32)
        meta = {
            "x_pin_star": cut,
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

    return write


ROOT_TRAIN = [
    "model.front_geometry=true",
    "model.nucleation_root=true",
    "training.root_supervision=true",
    "training.steps=120",
    "training.lr=0.01",
]


def test_root_supervision_requires_the_nucleation_root(tmp_path):
    from naviernet.models.pinn import BubblePINN
    from tests.conftest import staged_run

    cfg, _ = staged_run(
        tmp_path, ["model.front_geometry=true", "training.root_supervision=true"]
    )
    from naviernet.training import _validate_training_config

    with pytest.raises(ValueError, match="nucleation_root"):
        _validate_training_config(cfg)
    del BubblePINN  # imported for parity with sibling tests; the check is config-level


def test_the_root_plan_excludes_held_out_frames(tmp_path):
    from naviernet.data.dataset import BubbleDataset
    from naviernet.physics import root_supervision
    from naviernet.utils.paths import RunPaths
    from tests.conftest import staged_capsule_run

    cfg, _ = staged_capsule_run(
        tmp_path, ["training.val_fraction=0.2", "training.val_strategy=tail"]
    )
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    plan = root_supervision.build_plan(data, cfg, "cpu")
    assert set(plan.rows).isdisjoint(data.validation_rows)
    assert bool(np.isfinite(plan.bluntness.numpy()).all())
    assert bool((plan.depth > 0).all())


def test_the_supervision_drives_w_toward_a_blunt_root(tmp_path):
    """The R1 mechanism question in miniature: a dataset whose root is cut flat
    must pull w(t) POSITIVE (blunter), while the circular capsule leaves it
    near zero. This is the undriven-knob trap's regression test."""
    import torch

    from naviernet.training import load_model, train
    from tests.conftest import staged_run

    def trained_w(root, overrides):
        cfg, paths = staged_run(root, overrides, write=_blunt_capsule_writer(1.0))
        train(cfg, paths)
        model, _, _ = load_model(cfg, paths)
        with torch.no_grad():
            w = model.nets["phi"].root_bluntness(torch.tensor([[0.25]]))
        return float(w)

    w = trained_w(tmp_path / "blunt", ROOT_TRAIN)
    assert w > 0.05, f"the supervision left w(t) undriven: {w}"


def test_a_circular_root_leaves_w_near_zero(tmp_path):
    """The control: on data whose root IS a circle the same supervision must
    not invent bluntness."""
    import torch

    from naviernet.training import load_model, train
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(tmp_path, ROOT_TRAIN)
    train(cfg, paths)
    model, _, _ = load_model(cfg, paths)
    with torch.no_grad():
        w = model.nets["phi"].root_bluntness(torch.tensor([[0.25]]))
    assert abs(float(w)) < 0.08, f"supervision invented bluntness on a circle: {float(w)}"


def test_joint_runs_reject_the_nucleation_root_for_now(tmp_path):
    """The supervision plan is single-dataset; a joint w(t) with no channel
    would be exactly the undriven knob. Explicit rejection until it is
    threaded through the joint contexts."""
    from naviernet.training import train
    from tests.conftest import staged_joint_run

    cfg, paths = staged_joint_run(
        tmp_path, ["model.front_geometry=true", "model.nucleation_root=true"]
    )
    with pytest.raises(NotImplementedError, match="nucleation_root"):
        train(cfg, paths)


# --- T5 (R2): the attachment pressure -----------------------------------------
#
# Over the dry contact patch the gap-direction capillary term is NOT the
# both-plates-wetted 2/H*: contact-angle cosines add across the gap, so the
# attached side carries (1 + cos theta_app)/H* -- LOWER pressure, demanding a
# BLUNTER root, the measured sign. theta_app is one trained scalar whose init
# and bounds come from the per-fluid wetting config (T1); the patch footprint
# a(x,t) is a trained monotone extent about the anchor.

TINY_SHARP_ROOT = [
    "model=stage_b",
    "training=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.nucleation_root=true",
    "model.root_attachment=true",
    "training.root_supervision=true",
]


def test_root_attachment_is_off_by_default(cfg):
    assert cfg.model.root_attachment is False


def test_root_attachment_requires_its_parents(tmp_path):
    from naviernet.models.geometry import GeometryPriors
    from naviernet.models.pinn import BubblePINN
    from tests.conftest import staged_run

    priors = GeometryPriors(**PRIORS)
    cfg, _ = staged_run(
        tmp_path,
        [
            "model=stage_b",
            "model.front_geometry=true",
            "model.sharp_interface=true",
            "model.root_attachment=true",
        ],
    )
    with pytest.raises(ValueError, match="nucleation_root"):
        BubblePINN(cfg, geometry=priors)

    cfg2, _ = staged_run(
        tmp_path,
        [
            "model.front_geometry=true",
            "model.nucleation_root=true",
            "model.root_attachment=true",
        ],
    )
    with pytest.raises(ValueError, match="sharp_interface"):
        BubblePINN(cfg2, geometry=priors)


@pytest.mark.parametrize(("fluid_id", "source", "has_law"), WETTING_EXPECTATIONS)
def test_theta_app_initialises_at_the_fluids_own_law(tmp_path, fluid_id, source, has_law):
    """The apparent angle is an inverse unknown like r_int_star, but its START
    is the fluid's own physics: the evaporating-angle law at this run's
    superheat scale (or the advancing angle on water's hysteresis path)."""
    import torch

    from naviernet.models.pinn import BubblePINN
    from naviernet.physics.groups import compute_groups
    from tests.conftest import staged_capsule_run

    cfg, _ = staged_capsule_run(tmp_path, [*TINY_SHARP_ROOT, f"fluid={fluid_id}"])
    from naviernet.data.dataset import BubbleDataset
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths

    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    groups = compute_groups(cfg)
    theta = float(model.theta_app_deg(groups))
    assert theta == pytest.approx(groups["theta_app_init_deg"], abs=0.5)
    assert groups["theta_app_min_deg"] <= theta <= groups["theta_app_max_deg"]
    del torch


def test_theta_app_names_the_fix_on_a_pre_wetting_snapshot(tmp_path):
    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.physics.groups import compute_groups
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths
    from tests.conftest import staged_capsule_run

    cfg, _ = staged_capsule_run(tmp_path, TINY_SHARP_ROOT)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    groups = compute_groups(cfg)
    for key in ("theta_app_min_deg", "theta_app_max_deg", "theta_app_init_deg"):
        groups.pop(key, None)
    with pytest.raises(ValueError, match="wetting"):
        model.theta_app_deg(groups)


def test_the_attachment_lowers_the_gap_term_over_the_patch_only(tmp_path):
    """The blend reads (1 + cos theta)/H* over the patch and the free
    Bretherton branch beyond it: root-adjacent samples BELOW the free value,
    the nose untouched."""
    import torch

    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.physics.groups import compute_groups
    from naviernet.physics.residuals import gap_curvature, jump_gap_curvature
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths
    from tests.conftest import staged_capsule_run

    cfg, _ = staged_capsule_run(tmp_path, TINY_SHARP_ROOT)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    torch.manual_seed(0)
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    groups = compute_groups(cfg)

    front = model.front(torch.tensor([[0.2]]), n_body=32, n_cap=17)
    free = gap_curvature(front.normal_speed, groups)
    blended = jump_gap_curvature(model, front, groups)

    on_root = (front.on_cap.squeeze(1) > 0) & (front.u.squeeze(1) < 0.5)
    on_nose = (front.on_cap.squeeze(1) > 0) & (front.u.squeeze(1) > 0.5)
    apex = front.angle[on_root].squeeze(1).abs().argmin()
    assert float(blended[on_root][apex]) < float(free[on_root][apex])
    assert torch.allclose(blended[on_nose], free[on_nose], atol=1e-6)


def test_attachment_unknowns_move_off_init(tmp_path):
    """The T5 mechanism gate in miniature: after a short sharp-interface run,
    theta_app and the patch extent have both moved off their inits, and the
    patch extent never shrinks in time (the nose-rate treatment)."""
    import torch

    from naviernet.training import load_model, train
    from tests.conftest import staged_run

    cfg, paths = staged_run(
        tmp_path,
        [*TINY_SHARP_ROOT, "training.steps=80", "training.lr=0.01"],
        write=_blunt_capsule_writer(1.0),
    )
    train(cfg, paths)
    model, _, _ = load_model(cfg, paths)
    assert float(model._theta_app_raw.abs()) > 1e-4
    assert (
        float(model._patch_frac_raw.abs()) > 1e-4 or float(model._patch_rate_raw.abs()) > 1e-4
    )
    with torch.no_grad():
        extents = model.patch_extent(torch.tensor([[0.0], [0.25], [0.5]]))
    assert float(extents[1]) >= float(extents[0])
    assert float(extents[2]) >= float(extents[1])


# --- T6: the extended physics.root block --------------------------------------


def test_the_root_block_reports_the_trained_dof(tmp_path):
    """A nucleation-root run's metrics carry the mechanism gate's own numbers:
    w(t) per frame, its budget use, the patch extent, theta_app, and which
    wetting provenance the closures ran on."""
    import torch

    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.physics.diagnostics import physics_report
    from naviernet.physics.groups import compute_groups
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths
    from tests.conftest import staged_capsule_run

    cfg, _ = staged_capsule_run(tmp_path, TINY_SHARP_ROOT)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    torch.manual_seed(0)
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    root = physics_report(model, data, compute_groups(cfg))["root"]

    assert root["root_delta"] == pytest.approx(cfg.model.root_delta)
    assert 0.0 <= root["w_budget_use"] <= 1.0
    assert root["wetting_source"] == "measured"  # the default fluid is FC-72
    assert np.isfinite(root["theta_app_deg"])
    frame = root["per_frame"][0]
    assert np.isfinite(frame["w"])
    assert frame["patch_extent"] > 0


# --- T7's code gate: every offered fluid composes, validates and trains -------
#
# The bench runs on Series-1/FC-72 (the only data), but future series may use
# any of the five fluids, so the CODE must be correct for all of them from day
# one -- and a run on extrapolated wetting constants must say so in its output.


@pytest.mark.parametrize(("fluid_id", "source", "has_law"), WETTING_EXPECTATIONS)
def test_the_five_fluid_code_gate(tmp_path, fluid_id, source, has_law):
    from naviernet.physics.diagnostics import physics_report
    from naviernet.physics.groups import compute_groups
    from naviernet.training import train
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(tmp_path, [*TINY_SHARP_ROOT, f"fluid={fluid_id}"])
    model, data, _ = train(cfg, paths)  # 2 steps: the whole ladder engages
    root = physics_report(model, data, compute_groups(cfg))["root"]
    assert root["wetting_source"] == source
    assert np.isfinite(root["theta_app_deg"])
    assert np.isfinite(root["per_frame"][0]["w"])


# --- T8: variant coverage -----------------------------------------------------


def test_checkpoint_refuses_a_nucleation_root_mismatch(tmp_path):
    """The flag changes what the weights MEAN (a root net exists and the cap
    is a different family), so a mismatched invocation must fail loudly."""
    from naviernet.training import load_model, train
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(
        tmp_path, ["model.front_geometry=true", "model.nucleation_root=true"]
    )
    train(cfg, paths)
    plain, _ = staged_capsule_run(tmp_path, ["model.front_geometry=true"])
    with pytest.raises(ValueError, match="nucleation_root"):
        load_model(plain, paths)


def test_root_delta_change_is_refused_on_reload(tmp_path):
    """The bound scales the bluntness net's output: the same weights describe a
    different root through a different delta -- like cap_delta."""
    from naviernet.training import load_model, train
    from tests.conftest import staged_capsule_run

    on = ["model.front_geometry=true", "model.nucleation_root=true"]
    cfg, paths = staged_capsule_run(tmp_path, [*on, "model.root_delta=0.6"])
    train(cfg, paths)
    other, _ = staged_capsule_run(tmp_path, [*on, "model.root_delta=0.3"])
    with pytest.raises(ValueError, match="root_delta"):
        load_model(other, paths)


def test_a_nucleation_root_run_resumes_cleanly(tmp_path):
    """Resume must rebuild the same construction and keep training: the plan
    is deterministic, so the resumed run supervises against the identical
    measurement."""
    from naviernet.training import train
    from tests.conftest import staged_run

    cfg, paths = staged_run(tmp_path, ROOT_TRAIN, write=_blunt_capsule_writer(1.0))
    train(cfg, paths)
    _, _, state = train(cfg, paths)  # resumes from the checkpoint
    assert state["done"] == 2 * cfg.training.steps


def test_evaluate_writes_the_root_block_for_a_rooted_run(tmp_path):
    """The artifact, end-to-end: a trained nucleation-root run's metrics.json
    carries physics.root with the trained w(t) -- what the bench reads."""
    import json

    from naviernet.evaluation import evaluate
    from naviernet.training import train
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            "model.front_geometry=true",
            "model.nucleation_root=true",
            "training.root_supervision=true",
        ],
    )
    model, data, _ = train(cfg, paths)
    evaluate(cfg, model, data, paths)
    metrics = json.loads(paths.metrics_json.read_text())
    root = metrics["physics"]["root"]
    assert "w" in root["per_frame"][0]
    assert "w_budget_use" in root
    assert root["wetting_source"] == "measured"


# --- Series-1: the measurement this journey is built on -----------------------


@pytest.mark.needs_data
class TestSeries1RootDrift:
    """Plan §1, re-measured by the shipped module: the deviation from a circle
    is radial bluntness, it GROWS through the event, and both facts survive
    segmentation-threshold and window-size perturbation (plan §8.2)."""

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

    def test_bluntness_grows_and_the_circle_fit_degrades(self, data):
        rows = list(range(data.n_event))
        fits = [measured_root_fit(data, row) for row in rows]
        early = np.nanmean([f.taper_exponent for f in fits[:2]])
        late = np.nanmean([f.taper_exponent for f in fits[-2:]])
        assert late < early - 0.05, (early, late)
        assert fits[-1].circle_rms > fits[0].circle_rms + 0.05
        # Up/down symmetric to first order (plan §1.1): tiny bias, tiny tilt.
        assert all(abs(f.updown_bias) < 0.02 for f in fits)

    def test_the_drift_survives_threshold_and_window_perturbation(self, data):
        for level in (0.4, 0.5, 0.6):
            for window in (0.20, 0.25, 0.30):
                fits = [
                    measured_root_fit(data, row, window=window, alpha_level=level)
                    for row in range(data.n_event)
                ]
                early = np.nanmean([f.taper_exponent for f in fits[:2]])
                late = np.nanmean([f.taper_exponent for f in fits[-2:]])
                assert late < early, (level, window, early, late)
