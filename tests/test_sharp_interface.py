"""Sharp-interface physics (R4): interface conditions on the explicit front.

With ``model.sharp_interface`` on, the shape stops being driven by a bulk
residual that a free pressure field can absorb, and starts being driven by the
conditions that actually hold at a fluid interface -- the Young-Laplace jump and
the kinematic condition -- sampled ON the front the front geometry already
parameterizes. These tests drive the real layers: synthetic tensors -> priors ->
geometric phi -> front samples -> trainer -> checkpoint -> reloaded model.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import staged_run as _staged_run

# Sharp-interface physics needs the explicit front (front geometry) and the
# pressure field (Stage B), so every run here composes all three.
TINY_SHARP = ["model=stage_b", "model.front_geometry=true", "model.sharp_interface=true"]


def test_sharp_interface_walking_skeleton(tmp_path):
    """The whole path in one tiny real run: the flag composes, the registry adds
    the Laplace equation, the trainer samples the front and evaluates it, the
    vapour-pressure unknown trains, and the checkpoint records the architecture."""
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, TINY_SHARP)
    train(cfg, paths)

    ckpt = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    assert ckpt["sharp_interface"] is True, "the architecture record must carry the flag"

    record = ckpt["state"]["hist"][-1]
    assert "laplace" in record, f"the Laplace term must be trained and logged, got {record}"
    assert record["laplace"] == pytest.approx(record["laplace"]), "laplace must not be NaN"

    vapor = [k for k in ckpt["model"] if k.startswith("vapor_pressure")]
    assert vapor, f"p_v(t) must be a trained parameter, got fields {list(ckpt['model'])}"


def test_sharp_interface_defaults_off_and_leaves_the_equation_set_alone(tmp_path):
    """Opt-in: without the flag the composed physics is exactly what it was."""
    from naviernet.physics import registry

    cfg, _ = _staged_run(tmp_path, ["model=stage_b"])
    assert cfg.model.sharp_interface is False, "sharp_interface must default off"

    ids = [e.id for e in registry.enabled_equations(cfg.model.fields)]
    assert ids == ["vof", "div", "src", "bc", "mom", "energy", "evap"]


def test_sharp_interface_adds_the_laplace_equation_to_the_registry(tmp_path):
    """With the flag on, the Laplace jump joins the active set -- declaratively,
    so the trainer, the API and the UI all see one source of truth."""
    from naviernet.physics import registry

    cfg, _ = _staged_run(tmp_path, TINY_SHARP)
    ids = [e.id for e in registry.enabled_equations(cfg.model.fields, sharp_interface=True)]
    assert "laplace" in ids


def test_sharp_interface_requires_the_explicit_front(tmp_path):
    """There is no front to sample without the front geometry: fail loudly at
    construction rather than silently training a different objective."""
    from naviernet.models.pinn import BubblePINN

    cfg, _ = _staged_run(tmp_path, ["model=stage_b", "model.sharp_interface=true"])
    with pytest.raises(ValueError, match="front_geometry"):
        BubblePINN(cfg)


def test_sharp_interface_requires_the_pressure_field(tmp_path):
    """The jump condition reads the liquid pressure; a Stage-A field set cannot
    express it, so refuse the combination instead of silently dropping the term."""
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths

    cfg, paths = _staged_run(
        tmp_path, ["model.front_geometry=true", "model.sharp_interface=true"]
    )
    from naviernet.data.dataset import BubbleDataset

    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    with pytest.raises(ValueError, match="'p'"):
        BubblePINN(cfg, geometry=_geometry_priors(cfg, data))


def test_checkpoint_refuses_a_sharp_interface_mismatch(tmp_path):
    """Like the hard pin and the front geometry: the flag changes the objective
    and adds parameters, so a mismatched invocation must not consume the weights."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, TINY_SHARP)
    train(cfg, paths)

    plain, plain_paths = _staged_run(tmp_path, ["model=stage_b", "model.front_geometry=true"])
    with pytest.raises(ValueError, match="sharp_interface"):
        load_model(plain, plain_paths)


def test_front_samples_lie_exactly_on_the_interface(tmp_path):
    """The front sampler is only useful if its points really are the interface:
    alpha must be 0.5 at every sample, on the body and on both caps."""
    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths

    cfg, paths = _staged_run(tmp_path, TINY_SHARP)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))

    times = torch.tensor([[0.0], [0.5], [1.0]]) * data.domain.t_max
    front = model.nets["phi"].front(times, n_body=16, n_cap=8)
    alpha = model.alpha(front.points)
    assert torch.allclose(alpha, torch.full_like(alpha, 0.5), atol=1e-5), (
        f"front samples must sit on alpha=0.5; worst |alpha-0.5| = "
        f"{float((alpha - 0.5).abs().max())}"
    )


# --- T1: the interface curvature, taken from the parameterization -------------


def _geometry(tmp_path, overrides=None):
    """A model's geometric interface, built from the staged dataset's priors."""
    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths

    cfg, paths = _staged_run(tmp_path, overrides or TINY_SHARP)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    return model.nets["phi"], data


def _straighten(geo):
    """Zero the profile nets' output layers so the radius and centerline are
    exactly constant: a straight-sided capsule whose body curvature is known to
    be zero. Not a mock -- the nets keep their real forward pass, they are just
    put in a state whose geometry is analytically known."""
    for net in (geo.width_net, geo.center_net):
        torch.nn.init.zeros_(net[-1].weight)


def test_body_curvature_vanishes_on_a_straight_sided_capsule(tmp_path):
    """A constant radius about a flat centerline is two straight lines: the
    in-plane curvature of the body must be exactly zero, with no epsilon."""
    geo, data = _geometry(tmp_path)
    _straighten(geo)

    front = geo.front(torch.tensor([[0.0], [data.domain.t_max]]), n_body=32, n_cap=8)
    body = front.on_cap.squeeze() == 0
    kappa = front.kappa_par[body]
    assert torch.allclose(kappa, torch.zeros_like(kappa), atol=1e-5), (
        f"a straight body must read zero curvature, got max |kappa| = "
        f"{float(kappa.abs().max())}"
    )


def test_cap_curvature_is_the_caps_own_reciprocal_radius(tmp_path):
    """The caps are circles by construction, so their curvature is exactly 1/r --
    positive, because a vapour bubble is convex (the Laplace jump is positive)."""
    geo, data = _geometry(tmp_path)
    times = torch.tensor([[0.0], [data.domain.t_max]])
    front = geo.front(times, n_body=8, n_cap=8)
    frame = geo.frame(front.points[:, 2:3])

    cap = front.on_cap.squeeze() == 1
    expected = torch.where(front.u < 0.5, 1.0 / frame.r_root, 1.0 / frame.r_nose)
    assert torch.allclose(front.kappa_par[cap], expected[cap], rtol=1e-5)
    assert (front.kappa_par[cap] > 0).all(), "a convex vapour cap reads positive"


def test_analytic_curvature_matches_a_numerical_curvature_of_the_same_front(tmp_path):
    """The independent check: the closed-form curvature must agree with central
    differences of the very points the sampler returned. This is what licenses
    dropping the diffuse kappa, whose |kappa| spikes to ~1e4 on the spine."""
    geo, data = _geometry(tmp_path)
    # Perturb the profile nets off their near-constant init so the body has real,
    # varying curvature to get wrong.
    with torch.no_grad():
        for net in (geo.width_net, geo.center_net):
            net[-1].weight.add_(torch.randn_like(net[-1].weight) * 0.5)

    # 65 samples, not thousands: the second difference divides float32 rounding
    # by h^2, so too fine a grid measures roundoff rather than curvature.
    front = geo.front(torch.tensor([[0.5 * data.domain.t_max]]), n_body=65, n_cap=4)
    body = (front.on_cap.squeeze() == 0) & (front.side.squeeze() > 0)
    x = front.points[body, 0].detach().double()
    y = front.points[body, 1].detach().double()

    h = x[1] - x[0]
    y_x = (y[2:] - y[:-2]) / (2 * h)
    y_xx = (y[2:] - 2 * y[1:-1] + y[:-2]) / h**2
    numerical = -y_xx / (1.0 + y_x**2) ** 1.5

    analytic = front.kappa_par[body].squeeze()[1:-1].detach().double()
    assert torch.allclose(analytic, numerical, atol=5e-3, rtol=2e-2), (
        f"analytic vs numerical curvature disagree by "
        f"{float((analytic - numerical).abs().max())}"
    )


def test_front_curvature_carries_gradient_to_the_shape(tmp_path):
    """The curvature must be differentiable w.r.t. the profile nets, or the
    Laplace jump cannot move the shape -- which is the entire point."""
    geo, _ = _geometry(tmp_path)
    front = geo.front(torch.tensor([[0.4]]), n_body=8, n_cap=4)
    front.kappa_par.sum().backward()

    grad = geo.width_net[-1].weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0, (
        "curvature must backpropagate into the width profile"
    )


# --- T3: Darcy replaces the 2-D momentum residual -----------------------------


def test_darcy_replaces_momentum_in_sharp_mode(tmp_path):
    """One momentum-family equation at a time: the depth-averaged balance in
    sharp mode, the 2-D residual in diffuse mode. Never both."""
    from naviernet.physics import registry

    cfg, _ = _staged_run(tmp_path, TINY_SHARP)
    sharp = [e.id for e in registry.enabled_equations(cfg.model.fields, sharp_interface=True)]
    diffuse = [e.id for e in registry.enabled_equations(cfg.model.fields)]

    assert "darcy" in sharp and "mom" not in sharp
    assert "mom" in diffuse and "darcy" not in diffuse


def test_darcy_residual_is_the_depth_averaged_balance(tmp_path):
    """grad p = -C_HS mu*(alpha) u, term for term -- nothing else."""
    from naviernet.physics.groups import compute_groups
    from naviernet.physics.residuals import darcy_residuals, gradients, mixture

    model, data, cfg = _model_and_data(tmp_path)
    groups = compute_groups(cfg)
    x = _collocation(data)

    res = darcy_residuals(model, x, groups)
    u, v = model.velocity(x)
    p_x, p_y, _ = gradients(model.pressure(x), x)
    drag = groups["hele_shaw"] * mixture(model.alpha(x), groups["mu_ratio"])

    assert torch.allclose(res.mom_x, p_x + drag * u, atol=1e-6)
    assert torch.allclose(res.mom_y, p_y + drag * v, atol=1e-6)


def test_darcy_carries_no_inertia_no_in_plane_viscosity_no_body_force(tmp_path):
    """The three terms the regime argument removes. Each is detected by the group
    it would have entered through: Re for the in-plane Laplacian, We for the CSF
    surface-tension force. If either changes the residual, the term is still in."""
    from naviernet.physics.groups import compute_groups
    from naviernet.physics.residuals import darcy_residuals

    model, data, cfg = _model_and_data(tmp_path)
    groups = compute_groups(cfg)
    x = _collocation(data)
    base = darcy_residuals(model, x, groups)

    for key in ("Re", "We"):
        perturbed = darcy_residuals(model, x, {**groups, key: groups[key] * 1000.0})
        assert torch.allclose(perturbed.mom_x, base.mom_x, atol=1e-9), (
            f"the Darcy residual still depends on {key} -- a term that should be gone"
        )


def test_sharp_run_trains_darcy_and_never_momentum(tmp_path):
    """End to end: the trained history carries darcy, not mom."""
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, TINY_SHARP)
    train(cfg, paths)

    record = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    last = record["state"]["hist"][-1]
    assert "darcy" in last and "mom" not in last, last


def _model_and_data(tmp_path, overrides=None):
    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths

    cfg, paths = _staged_run(tmp_path, overrides or TINY_SHARP)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    return BubblePINN(cfg, geometry=_geometry_priors(cfg, data)), data, cfg


def _collocation(data, n: int = 64):
    import numpy as np

    return data.sample_collocation(n, np.random.default_rng(0))


# --- T4/T5: the front's normal, its own speed, and the gap curvature ----------


def test_front_normal_is_the_outward_unit_normal(tmp_path):
    """Unit length everywhere, and pointing OUT of the vapour: stepping a hair
    along it must lower alpha (leave the bubble) on every segment."""
    geo, data = _geometry(tmp_path)
    front = geo.front(torch.tensor([[0.3 * data.domain.t_max]]), n_body=24, n_cap=12)

    norm = front.normal.norm(dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    from naviernet.data.dataset import BubbleDataset  # noqa: F401  (fixture parity)

    step = 1e-3
    outside = front.points.clone()
    outside[:, :2] = outside[:, :2] + step * front.normal
    inside = front.points.clone()
    inside[:, :2] = inside[:, :2] - step * front.normal
    phi_out, phi_in = geo(outside), geo(inside)
    assert (phi_out < phi_in).all(), "the normal must point out of the vapour everywhere"


def test_front_normal_speed_matches_the_nose_rate_at_the_nose_apex(tmp_path):
    """The apex of the nose cap moves at exactly ds/dt: the one point on the
    front whose speed has an independent closed form to check against."""
    geo, data = _geometry(tmp_path)
    t = torch.tensor([[0.4 * data.domain.t_max]], requires_grad=True)
    nose = geo.nose(t)
    (rate,) = torch.autograd.grad(nose.sum(), t)

    front = geo.front(t.detach(), n_body=8, n_cap=9)
    apex = (front.on_cap.squeeze(1) == 1) & (front.u.squeeze(1) == 1.0)
    # The apex is the cap sample at angle 0, i.e. the one with side == 0.
    apex = apex & (front.side.squeeze(1) == 0)
    assert int(apex.sum()) == 1

    assert front.normal_speed[apex].item() == pytest.approx(float(rate), rel=1e-3)


def test_gap_curvature_grows_with_the_local_front_speed(tmp_path):
    """The Bretherton correction is the whole mechanism: a faster-advancing
    section of front carries MORE capillary pressure than a slow one, which is
    what inflates the nose and drains the mid-body."""
    from naviernet.physics.residuals import gap_curvature

    groups = {"H_star": 0.5, "Ca": 0.0107}
    slow = gap_curvature(torch.tensor([[0.05]]), groups)
    fast = gap_curvature(torch.tensor([[1.50]]), groups)

    assert float(fast) > float(slow) > 2.0 / groups["H_star"] * 0.999
    assert float(slow) == pytest.approx(2.0 / groups["H_star"], rel=0.05), (
        "a nearly stationary front reads the static gap curvature"
    )


def test_gap_curvature_ignores_a_receding_front(tmp_path):
    """The lubrication film is deposited by an ADVANCING meniscus; a receding one
    gets no thickening, and a negative capillary number has no 2/3 power."""
    from naviernet.physics.residuals import gap_curvature

    groups = {"H_star": 0.5, "Ca": 0.0107}
    receding = gap_curvature(torch.tensor([[-1.0]]), groups)
    assert float(receding) == pytest.approx(2.0 / groups["H_star"], rel=1e-6)


def test_h_star_is_the_channel_gap_over_the_reference_length(tmp_path):
    """The gap curvature is a property of the channel, computed like every other
    group -- never a literal."""
    from naviernet.physics.groups import compute_groups

    cfg, _ = _staged_run(tmp_path, TINY_SHARP)
    groups = compute_groups(cfg)
    expected = cfg.experiment.channel_height_um / cfg.scales.L_ref_um
    assert groups["H_star"] == pytest.approx(expected)


# --- T5: the kinematic condition, imposed on the front ------------------------


def test_kinematic_condition_equates_the_front_speed_to_the_normal_velocity(tmp_path):
    """v_n = u.n at the interface, term for term. Only the NORMAL component of
    the flow can move a front; a tangential slip along it moves nothing."""
    from naviernet.physics.residuals import kinematic_residual

    model, data, _ = _model_and_data(tmp_path)
    front = model.nets["phi"].front(torch.tensor([[0.3], [0.9]]), n_body=16, n_cap=8)

    u, v = model.velocity(front.points)
    advected = u * front.normal[:, 0:1] + v * front.normal[:, 1:2]
    assert torch.allclose(kinematic_residual(model, front), front.normal_speed - advected)


def test_kinematic_condition_joins_the_sharp_equation_set(tmp_path):
    """It is a boundary condition on the front, so -- like `bc` and `laplace` --
    it is not a collocation term and stays out of the causal/RBA reweighting."""
    from naviernet.physics import registry

    cfg, _ = _staged_run(tmp_path, TINY_SHARP)
    equations = registry.enabled_equations(cfg.model.fields, sharp_interface=True)
    kinematic = next(e for e in equations if e.id == "kinematic")
    assert kinematic.on_collocation is False
    assert "kinematic" not in [e.id for e in registry.enabled_equations(cfg.model.fields)]


@pytest.mark.slow  # trains the velocity nets for 600 Adam steps (~3 s)
def test_kinematic_condition_is_satisfiable_by_the_velocity_field(tmp_path):
    """Training the velocity nets against it alone must drive it down -- if it
    cannot be satisfied, adding it to the objective only fights the other terms."""
    from naviernet.physics.residuals import kinematic_residual

    torch.manual_seed(0)
    model, data, _ = _model_and_data(
        tmp_path, [*TINY_SHARP, "model.hidden=64", "model.layers=3", "model.fourier_feats=32"]
    )
    times = torch.linspace(data.domain.t_min, data.domain.t_max, 8).reshape(-1, 1)
    params = [p for name in ("u", "v") for p in model.nets[name].parameters()]
    opt = torch.optim.Adam(params, lr=5e-3)

    def loss():
        front = model.nets["phi"].front(times, n_body=32, n_cap=8)
        return (kinematic_residual(model, front) ** 2).mean()

    before = float(loss().detach())
    for _ in range(600):
        opt.zero_grad()
        loss().backward()
        opt.step()
    after = float(loss().detach())
    assert after < 0.05 * before, f"kinematic residual {before:.4g} -> {after:.4g}"


def test_sharp_run_trains_the_kinematic_condition(tmp_path):
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, TINY_SHARP)
    train(cfg, paths)
    last = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)["state"][
        "hist"
    ][-1]
    assert "kinematic" in last, last


# --- T13: variants and edge cases ---------------------------------------------


def test_a_sharp_run_resumes_and_keeps_training_the_same_objective(tmp_path):
    """Two chunks must equal one run: the front times are a deterministic grid
    and p_v is an ordinary parameter, so resume has to pick both up unchanged."""
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, [*TINY_SHARP, "training.steps=2"])
    train(cfg, paths)
    train(cfg, paths)

    ckpt = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    assert ckpt["state"]["done"] == 4
    assert "laplace" in ckpt["state"]["hist"][-1]


def test_a_degenerate_front_stays_finite(tmp_path):
    """A just-nucleated bubble is shorter than its own cap radii, so the spine
    collapses and every per-sample quantity divides by something small. Curvature,
    normals and speeds must all stay finite -- a NaN here poisons the whole step."""
    from naviernet.models.geometry import GeometricInterface, GeometryPriors

    torch.manual_seed(0)
    geo = GeometricInterface(
        GeometryPriors(
            x_root=0.2,
            y_root=0.25,
            s0=0.201,
            w0=0.06,
            rate0=0.0,
            y_min=0.0,
            y_max=0.5,
            t_min=0.0,
            t_max=1.0,
        )
    )
    front = geo.front(torch.tensor([[0.0], [1.0]]), n_body=16, n_cap=8)
    for name, tensor in front._asdict().items():
        assert torch.isfinite(tensor).all(), f"{name} went non-finite on a degenerate front"


def test_front_sampling_rejects_a_cap_that_is_not_an_arc(tmp_path):
    """A one-point cap is a point, not an arc; refuse it rather than silently
    sampling a degenerate contour."""
    geo, _ = _geometry(tmp_path)
    with pytest.raises(ValueError, match="n_cap"):
        geo.front(torch.tensor([[0.0]]), n_body=8, n_cap=1)


def test_sharp_interface_composes_with_causal_weighting(tmp_path):
    """The interface conditions are boundary terms, so they sit outside the
    causal collocation reweighting -- the two must compose without exploding."""
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, [*TINY_SHARP, "training.causal_weighting=true"])
    train(cfg, paths)

    record = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    last = record["state"]["hist"][-1]
    assert all(v == v for v in last.values()), f"a term went NaN: {last}"


def test_pinching_and_sharp_interface_compose(tmp_path):
    """The recipe a detachment study would actually run."""
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, [*TINY_SHARP, "model.allow_pinch=true"])
    train(cfg, paths)
    last = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)["state"][
        "hist"
    ][-1]
    assert all(v == v for v in last.values()), f"a term went NaN: {last}"


# --- joint (multi-dataset) sharp-interface runs -------------------------------


def test_joint_sharp_run_samples_each_datasets_own_front(tmp_path):
    """Each dataset's interface conditions must be imposed on ITS OWN front.

    The front is reached through the dataset's bound view; a view that fell
    through to the raw model would hand every condition the same front -- the
    same one, at the same place -- and every jump residual after the first would
    be scored against another dataset's interface. The staged datasets have
    deliberately different roots, so identical fronts are detectable.
    """
    from naviernet.physics.groups import N_COND
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _load_joint_datasets
    from tests.conftest import staged_joint_run

    cfg, paths = _staged_run_joint(tmp_path)
    contexts = _load_joint_datasets(cfg, paths, torch.device("cpu"))
    model = BubblePINN(cfg, n_cond=N_COND, geometry=contexts[0].geometry)

    fronts = {}
    for cx in contexts:
        view = model.bound(cx.c, pin=cx.pin, geometry=cx.geometry)
        times = torch.tensor([[cx.data.domain.t_min]])
        # An ODD cap count so the sweep includes angle 0 -- the apex, the only
        # cap sample that sits exactly on the root.
        fronts[cx.name] = view.front(times, n_body=8, n_cap=5).points

    (first, second) = fronts.values()
    assert not torch.allclose(first, second), (
        "every dataset sampled the same front -- the bound view is not carrying "
        "its own anchors"
    )
    for cx in contexts:
        root_x = cx.data.pin_anchor[0]
        assert float(fronts[cx.name][:, 0].min()) == pytest.approx(root_x, abs=1e-3), (
            f"{cx.name}'s front does not start at its own root"
        )


def test_a_joint_sharp_run_trains(tmp_path):
    """The combination end to end -- it used to raise from deep inside the first
    step, blaming a mechanism the joint trainer never wired."""
    from naviernet.training import train

    cfg, paths = _staged_run_joint(tmp_path)
    train(cfg, paths)

    last = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)["state"]["hist"][-1]
    assert "laplace" in last and "kinematic" in last, last
    assert all(v == v for v in last.values()), f"a term went NaN: {last}"


def _staged_run_joint(tmp_path):
    from tests.conftest import staged_joint_run

    return staged_joint_run(tmp_path, TINY_SHARP)
