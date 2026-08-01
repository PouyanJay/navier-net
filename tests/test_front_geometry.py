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
    geo = _random_geo(seed)
    times = torch.linspace(0.0, 5.0, 200).unsqueeze(1)  # grid ends at 1.5

    with torch.no_grad():
        s = geo.nose(times).squeeze(1)

    assert torch.all(s[1:] >= s[:-1] - 1e-7), "the nose retreated"
    assert float(s[0]) == pytest.approx(PRIORS["x_root"] + 0.3, abs=1e-5), (
        "s(t_min) != measured s0"
    )


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
    delicate = torch.tensor(
        [
            [PRIORS["x_root"] + 1e-4, float(root[1]), 0.5],  # at the root cap
            [nose - 1e-4, float(root[1]), 0.5],  # at the nose cap
            [0.35, float(root[1]), 0.5],  # on the centerline mid-capsule
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
    assert tail < 0.4 * hist[0]["data"], (
        f"student failed to recover the teacher: {hist[0]['data']:.4f} -> tail mean {tail:.4f}"
    )


def test_front_geometry_resumes_cleanly(tmp_path):
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, [*TINY_GEO, "training.steps=1"])
    train(cfg, paths)
    _, _, state = train(cfg, paths)  # resume

    assert state["done"] == 2
    model, data, _ = load_model(cfg, paths)
    root = model.nets["phi"].root_point(2.0 * data.domain.t_max)
    assert torch.allclose(model.alpha(root.unsqueeze(0)), torch.tensor([[0.5]]), atol=1e-6)


def test_checkpoint_refuses_a_front_geometry_mismatch(tmp_path):
    from naviernet.training import load_model, train
    from tests.conftest import make_config

    cfg_on, paths = _staged_run(tmp_path, TINY_GEO)
    train(cfg_on, paths)
    cfg_off = make_config([f"paths.root={tmp_path}", "training.holdout_frame=-1"])

    with pytest.raises(ValueError, match="front_geometry"):
        load_model(cfg_off, paths)


def test_front_geometry_rejects_joint_runs(tmp_path):
    from naviernet.training import train
    from tests.conftest import staged_joint_run

    cfg, paths = staged_joint_run(tmp_path, TINY_GEO)

    with pytest.raises(NotImplementedError, match="front_geometry"):
        train(cfg, paths)


def test_front_geometry_composes_with_causal_and_kinematics(tmp_path):
    """The bench stack: geometry + causal trains; kinematics' volume terms read
    alpha and compose untouched."""
    from naviernet.training import train

    for name, extra in (
        ("causal", ["training.causal_weighting=true"]),
        (
            "kin",
            [
                "training.kinematics=true",
                "training.kin_grid=6",
                "training.kin_times=3",
                "training.kin_weight_evap=0",
            ],
        ),
        ("rba", ["training.weighting=rba"]),
    ):
        cfg, paths = _staged_run(tmp_path / name, [*TINY_GEO, *extra])
        train(cfg, paths)
        assert paths.checkpoint.exists(), f"{name} composition failed"
