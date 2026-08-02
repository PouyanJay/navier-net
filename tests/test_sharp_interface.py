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
