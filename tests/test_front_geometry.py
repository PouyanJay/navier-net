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


# --- Structural guarantees on RANDOM (untrained) nets -------------------------

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


def _random_geo(seed: int):
    from naviernet.models.geometry import GeometricInterface, GeometryPriors

    torch.manual_seed(seed)
    return GeometricInterface(GeometryPriors(**PRIORS))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nose_is_monotone_even_beyond_the_grid(seed):
    """Monotone for TRAINED-scale nets, not just the flat init: near-constant
    rates hide interpolation bugs (the reviewed nose bug produced negative node
    jumps only once the rate net had real temporal structure), so the rate net
    is scaled 10x and the sweep includes the exact grid nodes."""
    geo = _random_geo(seed)
    with torch.no_grad():
        for p in geo.rate_net.parameters():
            p.mul_(10.0)
    dense = torch.linspace(0.0, 5.0, 30_000)
    times = torch.cat([dense, geo.time_grid()]).sort().values.unsqueeze(1)

    with torch.no_grad():
        s = geo.nose(times).squeeze(1)

    assert torch.all(s[1:] >= s[:-1] - 1e-7), "the nose retreated"
    assert float(s[0]) == pytest.approx(PRIORS["x_root"] + 0.3, abs=1e-5), (
        "s(t_min) != measured s0"
    )


@pytest.mark.parametrize("degenerate", [{"w0": 0.0}, {"rate0": 0.0}, {"w0": 0.0, "rate0": 0.0}])
def test_degenerate_priors_stay_finite_and_monotone(degenerate):
    from naviernet.models.geometry import GeometricInterface, GeometryPriors

    torch.manual_seed(0)
    geo = GeometricInterface(GeometryPriors(**{**PRIORS, **degenerate}))
    times = torch.linspace(0.0, 3.0, 500).unsqueeze(1)

    with torch.no_grad():
        s = geo.nose(times).squeeze(1)
        phi = geo(torch.tensor([[0.35, 0.25, 1.0]]))

    assert torch.isfinite(s).all() and torch.isfinite(phi).all()
    assert torch.all(s[1:] >= s[:-1] - 1e-7)


def test_interface_closes_exactly_at_root_and_nose():
    """Guarantee #4 measured at both ends: alpha is exactly 0.5 at the root
    point AND the nose point, for a random net at an extrapolated time."""
    geo = _random_geo(1)
    for point in (geo.root_point(2.5), geo.nose_point(2.5)):
        with torch.no_grad():
            alpha = torch.sigmoid(geo(point.unsqueeze(0)) / 0.05)
        assert torch.allclose(alpha, torch.tensor([[0.5]]), atol=1e-6), (
            f"interface does not close at {point.tolist()}: {float(alpha)}"
        )


def test_short_bubble_keeps_both_apexes_exact_and_vapor_inside():
    """Review-reproduced regression: a just-nucleated bubble shorter than its
    cap radii used to push the root-cap center past the nose (alpha 0.55 at the
    nose point, vapor overshooting the tracked nose by half the bubble length).
    The joint radius rescale must keep both apexes exact and the vapor inside
    [x_root, s]."""
    from naviernet.models.geometry import GeometricInterface, GeometryPriors

    torch.manual_seed(0)
    geo = GeometricInterface(GeometryPriors(**{**PRIORS, "s0": 0.21, "rate0": 0.0}))

    for point in (geo.root_point(0.0), geo.nose_point(0.0)):
        with torch.no_grad():
            alpha = torch.sigmoid(geo(point.unsqueeze(0)) / 0.05)
        assert torch.allclose(alpha, torch.tensor([[0.5]]), atol=1e-6), (
            f"apex lost on the short bubble at {point.tolist()}: {float(alpha)}"
        )

    with torch.no_grad():
        s = float(geo.nose(torch.tensor([[0.0]])))
        xs = torch.linspace(0.15, 0.5, 400)
        pts = torch.stack([xs, torch.full_like(xs, PRIORS["y_root"]), torch.zeros_like(xs)], 1)
        vapor = (torch.sigmoid(geo(pts) / 0.05) > 0.5).squeeze(1)
    reach = xs[vapor]
    assert reach.numel() > 0 and float(reach.max()) <= s + 1e-3, (
        f"vapor overshoots the tracked nose: {float(reach.max()):.4f} > s={s:.4f}"
    )
    # And the degenerate spine must not spike the VOF-facing gradient.
    probe = torch.tensor(
        [[0.5 * (PRIORS["x_root"] + s), PRIORS["y_root"], 0.0]], requires_grad=True
    )
    alpha = torch.sigmoid(geo(probe) / 0.05)
    a_x = torch.autograd.grad(alpha.sum(), probe)[0][0, 0]
    assert torch.isfinite(a_x) and abs(float(a_x)) < 5e3, f"alpha_x spiked: {float(a_x):.3e}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_geometry_is_one_capsule_inside_the_channel(seed):
    from scipy import ndimage

    geo = _random_geo(seed)
    xs = torch.linspace(0.0, 1.1, 120)
    ys = torch.linspace(0.0, 0.5, 60)
    for t in (0.0, 1.0, 3.0):
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        pts = torch.stack([gx.ravel(), gy.ravel(), torch.full_like(gx.ravel(), t)], dim=1)
        with torch.no_grad():
            alpha = torch.sigmoid(geo(pts) / 0.05).reshape(120, 60)
        mask = (alpha > 0.5).numpy()
        _, n = ndimage.label(mask)
        assert n == 1, f"seed {seed}, t={t}: {n} components"
        cols = mask.any(axis=1).nonzero()[0]
        assert xs[cols[0]] >= PRIORS["x_root"] - 0.02, "vapour upstream of the root"
        with torch.no_grad():
            nose = float(geo.nose(torch.tensor([[t]])))
        assert xs[cols[-1]] <= nose + 0.02, "vapour beyond the nose"


@pytest.mark.parametrize("seed", [0, 1])
def test_random_geometry_phi_is_negative_outside_the_slab(seed):
    geo = _random_geo(seed)
    outside = torch.tensor(
        [[0.05, 0.25, 0.5], [0.19, 0.1, 2.0], [50.0, 0.25, 0.5], [0.0, 0.0, 0.0]]
    )
    with torch.no_grad():
        phi = geo(outside)
    assert torch.all(phi < 0), f"phi >= 0 outside the capsule slab: {phi.ravel()}"


def test_geometry_keeps_the_surface_tension_term_bounded():
    """The KAPPA lesson, applied to the new construction: kappa * grad(alpha)
    must stay bounded at the delicate places -- the root cap, the nose cap, and
    the (smoothed) centerline."""
    from naviernet.physics.residuals import curvature

    geo = _random_geo(0)
    with torch.no_grad():
        nose = float(geo.nose(torch.tensor([[0.5]])))
        root = geo.root_point(0.5)
        t_col = torch.full((1, 1), 0.5)
        r0 = float(geo._radius(torch.zeros(1, 1), t_col))
        r1 = float(geo._radius(torch.ones(1, 1), t_col))
    y0 = float(root[1])
    delicate = torch.tensor(
        [
            [PRIORS["x_root"] + 1e-4, y0, 0.5],  # at the root cap apex
            [nose - 1e-4, y0, 0.5],  # at the nose cap apex
            [0.35, y0, 0.5],  # on the centerline mid-capsule
            # The cap-body SEAMS (x = cap-center planes, on the interface
            # flank): the u-clamp makes phi C0-but-not-C1 there -- a measured,
            # accepted trade-off (kappa*a_x ~ O(10-50), far under the bound;
            # a C1 blend would require tying R'(0)=0 and cost expressivity).
            [PRIORS["x_root"] + r0, y0 + r0, 0.5],
            [nose - r1, y0 + r1, 0.5],
        ],
        requires_grad=True,
    )
    alpha = torch.sigmoid(geo(delicate) / 0.05)
    kappa = curvature(alpha, delicate)
    a_x = torch.autograd.grad(alpha.sum(), delicate, create_graph=True)[0][:, 0:1]
    product = (kappa * a_x).detach()
    assert torch.isfinite(product).all() and product.abs().max() < 1e4, (
        f"surface-tension term unbounded at the delicate points: {product.ravel()}"
    )


# --- Fit, resume, compat ------------------------------------------------------


def test_front_geometry_learns_a_representable_target(tmp_path):
    """The representation must be able to LEARN, not just constrain. The shared
    rectangle fixture is adversarial to a capsule (full width to sharp edges),
    so this is the well-specified inverse problem instead: supervision generated
    FROM a teacher GeometricInterface (exactly representable by construction,
    ``sdf := -phi_teacher`` so the smoothed targets reproduce exactly), and a
    differently-seeded student must recover it. Physics weights zeroed: the SUT
    is the representation's learnability, not the full-objective dynamics."""
    import numpy as np

    from naviernet.training import train

    cfg, paths = _staged_run(
        tmp_path,
        [
            *TINY_GEO,
            "training.steps=1000",
            "training.n_data=128",
            "training.log_every=1",
            "training.weights.vof=0",
            "training.weights.div=0",
            "training.weights.src=0",
            "training.weights.bc=0",
        ],
    )
    archive = dict(np.load(paths.tensors))
    meta = archive.pop("meta")
    n_t, n_y, n_x = archive["alpha"].shape
    xs, ys, ts = archive["x_star"], archive["y_star"], archive["t_star"]

    teacher = _random_geo(seed=7)
    # Nudge the teacher off the data-anchored init so the student has real work.
    with torch.no_grad():
        teacher.width_net[-1].bias += 0.8
        teacher.center_net[-1].bias += 0.4
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    for k in range(n_t):
        pts = torch.tensor(
            np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, ts[k])], axis=1),
            dtype=torch.float32,
        )
        with torch.no_grad():
            phi = teacher(pts).numpy().reshape(n_y, n_x)
        archive["sdf"][k] = -phi.astype(np.float32)
        archive["alpha"][k] = (phi > 0).astype(np.float32)
    np.savez_compressed(paths.tensors, **archive, meta=meta)

    _, _, state = train(cfg, paths)

    hist = state["hist"]
    # Mean over the last 50 steps de-noises the tiny-batch loss (measured
    # trajectory: 0.046 -> ~0.01 by step 800-1000).
    tail = sum(r["data"] for r in hist[-50:]) / 50
    # A coarse convergence gate only -- the strong claim is the direct
    # reconstruction below (the capsule-form landscape sits near 0.40 exactly).
    assert tail < 0.45 * hist[0]["data"], (
        f"student failed to recover the teacher: {hist[0]['data']:.4f} -> tail mean {tail:.4f}"
    )

    # Direct reconstruction, not just a loss ratio: the student's alpha must
    # track the teacher's on a probe grid (loss can plateau at a mediocre
    # minimum and still clear a relative bar).
    from naviernet.training import load_model

    student, _, _ = load_model(cfg, paths)
    probe_x = torch.linspace(0.05, 1.05, 40)
    probe_y = torch.linspace(0.02, 0.48, 20)
    gx2, gy2 = torch.meshgrid(probe_x, probe_y, indexing="ij")
    for t in (0.05, 0.25):
        probe = torch.stack([gx2.ravel(), gy2.ravel(), torch.full_like(gx2.ravel(), t)], dim=1)
        with torch.no_grad():
            student_alpha = student.alpha(probe)
            teacher_alpha = torch.sigmoid(teacher(probe) / student.eps)
        gap = float((student_alpha - teacher_alpha).abs().mean())
        assert gap < 0.08, f"student/teacher mean alpha gap {gap:.3f} at t={t}"


def test_front_geometry_resumes_cleanly(tmp_path):
    """Resume accumulates steps AND the reloaded model reproduces the trained
    model's predictions exactly at off-root points (the root-pin invariant alone
    is true by construction for ANY weights, so it cannot prove the round-trip)."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, [*TINY_GEO, "training.steps=1"])
    train(cfg, paths)
    trained, _, state = train(cfg, paths)  # resume

    assert state["done"] == 2
    probe = torch.tensor(
        [[0.4, 0.3, 0.15], [0.8, 0.2, 0.35], [0.6, 0.25, 0.9]], dtype=torch.float32
    )
    with torch.no_grad():
        before = trained.alpha(probe)
    reloaded, _, _ = load_model(cfg, paths)
    with torch.no_grad():
        after = reloaded.alpha(probe)
    assert torch.equal(before, after), "checkpoint round-trip changed the predictions"


def test_checkpoint_refuses_a_front_geometry_mismatch(tmp_path):
    from naviernet.training import load_model, train
    from tests.conftest import make_config

    cfg_on, paths = _staged_run(tmp_path, TINY_GEO)
    train(cfg_on, paths)
    cfg_off = make_config([f"paths.root={tmp_path}", "training.holdout_frame=-1"])

    with pytest.raises(ValueError, match="front_geometry"):
        load_model(cfg_off, paths)


def test_front_geometry_trains_a_joint_run_and_pins_each_dataset_to_its_own_root(tmp_path):
    """Transfer, the point of the whole conditioning apparatus: one shared
    construction, each condition landing on ITS OWN measured anchors.

    The two staged datasets have deliberately different roots, so a construction
    that ignored the binding -- or bound the primary's anchors to everything --
    would put both interfaces at the same x, and this would catch it.
    """
    from naviernet.models.geometry import GeometryContext
    from naviernet.training import load_joint, train
    from tests.conftest import staged_joint_run

    cfg, paths = staged_joint_run(tmp_path, TINY_GEO)
    train(cfg, paths)
    model, contexts, _ = load_joint(cfg, paths)

    roots = {cx.name: cx.data.pin_anchor[0] for cx in contexts}
    assert len(set(roots.values())) == len(roots), (
        "the fixture must give the datasets distinct roots or this proves nothing"
    )

    geo = model.nets["phi"]
    for cx in contexts:
        for t in (cx.data.domain.t_min, cx.data.domain.t_max, 2.0 * cx.data.domain.t_max):
            point = geo.root_point(t, GeometryContext(cx.c, cx.geometry))
            assert float(point[0]) == pytest.approx(roots[cx.name], abs=1e-6), (
                f"{cx.name} is anchored at {float(point[0])}, not its own root {roots[cx.name]}"
            )
            alpha = model.bound(cx.c, pin=cx.pin, geometry=cx.geometry).alpha(
                point.unsqueeze(0)
            )
            assert torch.allclose(alpha, torch.tensor([[0.5]]), atol=1e-6), (
                f"{cx.name}'s interface does not pass through its root at t={t}"
            )


def test_joint_front_geometry_starts_each_dataset_at_its_own_front(tmp_path):
    """The nose too: the datasets grow at different rates from different fronts,
    and the shared rate/gap are scaled to each condition's measured values."""
    from naviernet.models.geometry import GeometryContext
    from naviernet.models.pinn import BubblePINN
    from naviernet.physics.groups import N_COND
    from naviernet.training import _load_joint_datasets
    from tests.conftest import staged_joint_run

    cfg, paths = staged_joint_run(tmp_path, TINY_GEO)
    contexts = _load_joint_datasets(cfg, paths, torch.device("cpu"))
    model = BubblePINN(cfg, n_cond=N_COND, geometry=contexts[0].geometry)

    geo = model.nets["phi"]
    for cx in contexts:
        t0 = torch.tensor([[cx.data.domain.t_min]])
        with torch.no_grad():
            start = float(geo.nose(t0, GeometryContext(cx.c, cx.geometry)))
        assert start == pytest.approx(cx.data.initial_front, abs=1e-4), (
            f"{cx.name} starts its nose at {start}, not its measured front "
            f"{cx.data.initial_front}"
        )


def test_single_dataset_geometry_is_untouched_by_the_conditioning_support(tmp_path):
    """Every per-dataset rescaling is a ratio against the reference dataset, so
    for a single-dataset run each one is exactly 1 and the construction is what
    it always was. Asserted, not assumed."""
    from naviernet.models.geometry import (
        GeometricInterface,
        GeometryContext,
        GeometryPriors,
    )

    torch.manual_seed(0)
    priors = GeometryPriors(**PRIORS)
    geo = GeometricInterface(priors)
    t = torch.linspace(0.0, 2.0, 64).reshape(-1, 1)

    with torch.no_grad():
        implicit = geo.nose(t)
        explicit = geo.nose(t, GeometryContext(priors=priors))  # same anchors, passed in
    assert torch.equal(implicit, explicit)
    assert float(implicit[0]) == pytest.approx(PRIORS["x_root"] + 0.3, abs=1e-5)


def test_front_geometry_composes_with_causal_and_kinematics(tmp_path):
    """The bench stack compositions, each asserted by an effect only the
    composed technique produces (a bare checkpoint-exists would pass a silent
    no-op -- the anti-pattern the kinematics review already banned)."""
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path / "plain", [*TINY_GEO, "training.log_every=1"])
    _, _, plain_state = train(cfg, paths)

    kin_extra = [
        "training.kinematics=true",
        "training.kin_grid=6",
        "training.kin_times=3",
        "training.kin_weight_evap=0",
        "training.log_every=1",
    ]
    cfg, paths = _staged_run(tmp_path / "kin", [*TINY_GEO, *kin_extra])
    _, _, kin_state = train(cfg, paths)
    assert "kin_mono" in kin_state["hist"][-1], "kinematics silently dropped"

    cfg, paths = _staged_run(tmp_path / "rba", [*TINY_GEO, "training.weighting=rba"])
    _, _, rba_state = train(cfg, paths)
    assert "attention" in rba_state, "RBA attention never engaged"

    cfg, paths = _staged_run(
        tmp_path / "causal",
        [*TINY_GEO, "training.causal_weighting=true", "training.log_every=1"],
    )
    _, _, causal_state = train(cfg, paths)
    plain_last, causal_last = plain_state["hist"][-1], causal_state["hist"][-1]
    assert any(
        plain_last[k] != causal_last[k] for k in plain_last if k not in ("step", "lr")
    ), "causal weighting had no effect on the trajectory"
