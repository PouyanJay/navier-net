"""Hard root pin: the level-set output transform anchoring the bubble root.

The pin is architectural -- ``phi = tanh(dist/d_ref) * N`` -- so the interface
(alpha = 0.5) passes through the dataset's measured root anchor at every time,
including times beyond the training window. These tests drive the real layers:
synthetic tensors -> anchor measurement -> model transform -> trainer ->
checkpoint -> reloaded model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tests.conftest import (
    GROWING_FRAMES,
    GROWING_H,
    GROWING_W,
    JOINT_SPECS,
    ROOT_COL,
    VAPOR_ROWS,
    make_config,
)
from tests.conftest import (
    staged_joint_run as _staged_joint_run,
)
from tests.conftest import (
    staged_run as _staged_run,
)


def test_hard_pin_holds_the_interface_at_the_anchor_for_all_t(tmp_path):
    """The walking skeleton: train with the pin on, reload the checkpoint, and the
    interface sits exactly on the measured root anchor -- at trained times AND at
    times far beyond the training window (the extrapolation guarantee)."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, ["model.hard_pin=true"])
    train(cfg, paths)
    model, data, _ = load_model(cfg, paths)

    x0, y0 = data.pin_anchor
    times = [data.domain.t_min, data.domain.t_max, 3.0 * data.domain.t_max + 1.0]
    points = torch.tensor([[x0, y0, t] for t in times], dtype=torch.float32)

    alpha = model.alpha(points)

    # The gate is exactly tanh(0) = 0 at the anchor, so alpha is bit-exact 0.5 in
    # float32 -- atol here guards gross regressions, not a loose numeric budget.
    assert torch.allclose(alpha, torch.full_like(alpha, 0.5), atol=1e-6), (
        f"pinned alpha at the anchor should be exactly 0.5 at every t, got {alpha.ravel()}"
    )


def test_hard_pin_off_leaves_phi_raw(tmp_path):
    """Flag off (the default): phi is the raw network output -- no transform."""
    from naviernet.models.pinn import BubblePINN

    cfg, _ = _staged_run(tmp_path)
    assert cfg.model.hard_pin is False, "hard_pin must default off"

    model = BubblePINN(cfg)
    x = torch.tensor([[0.3, 0.2, 0.05], [0.9, 0.4, 0.35]], dtype=torch.float32)

    assert torch.equal(model.phi(x), model.nets["phi"](x))


# --- Anchor measurement (dataset layer) --------------------------------------


def test_anchor_sits_on_the_stationary_edge(tmp_path):
    """The root anchor is the stationary bubble edge's x* and the centre of its
    vapour column in y* -- for the synthetic event, known in closed form."""
    from naviernet.data.dataset import BubbleDataset

    cfg, paths = _staged_run(tmp_path)
    data = BubbleDataset(cfg, paths)

    x_star = np.linspace(0, 1.1, GROWING_W, dtype=np.float32)
    y_star = np.linspace(0, 0.5, GROWING_H, dtype=np.float32)
    x0, y0 = data.pin_anchor

    assert x0 == pytest.approx(float(x_star[ROOT_COL]))
    assert y0 == pytest.approx(float(y_star[VAPOR_ROWS].mean()))


def test_anchor_is_orientation_agnostic(tmp_path):
    """A mirrored event (root on the high-x side, front growing toward low x)
    anchors on the high-x edge -- no orientation is assumed."""
    from naviernet.data.dataset import BubbleDataset

    cfg, paths = _staged_run(tmp_path)
    archive = dict(np.load(paths.tensors))
    meta = archive.pop("meta")
    fields = ("alpha", "sdf", "valid", "masks_camera")  # mirror fields, not coordinates
    flipped = {
        k: np.ascontiguousarray(v[..., ::-1]) if k in fields else v for k, v in archive.items()
    }
    np.savez_compressed(paths.tensors, **flipped, meta=meta)

    data = BubbleDataset(cfg, paths)
    x_star = np.linspace(0, 1.1, GROWING_W, dtype=np.float32)
    y_star = np.linspace(0, 0.5, GROWING_H, dtype=np.float32)
    mirrored_root = GROWING_W - 1 - ROOT_COL

    assert data.pin_anchor[0] == pytest.approx(float(x_star[mirrored_root]))
    assert data.pin_anchor[1] == pytest.approx(float(y_star[VAPOR_ROWS].mean()))


def test_anchor_tie_breaks_to_the_low_edge(tmp_path):
    """A static bubble (both edges equally stationary) anchors deterministically
    on the low-x edge -- pinned deliberately, not left as an accident."""
    from naviernet.data.dataset import BubbleDataset

    cfg, paths = _staged_run(tmp_path)
    archive = dict(np.load(paths.tensors))
    meta = archive.pop("meta")
    static = np.zeros_like(archive["alpha"])
    static[:, VAPOR_ROWS, 3:7] = 1.0  # same extent in every frame
    archive["alpha"] = static
    np.savez_compressed(paths.tensors, **archive, meta=meta)

    data = BubbleDataset(cfg, paths)
    x_star = np.linspace(0, 1.1, GROWING_W, dtype=np.float32)

    assert data.pin_anchor[0] == pytest.approx(float(x_star[3]))


def test_anchor_never_reads_held_out_frames(tmp_path):
    """Leakage guard: corrupting the root in the held-out tail frame must not
    move the anchor, because the anchor is measured on training frames only."""
    from naviernet.data.dataset import BubbleDataset

    split = ["training.val_fraction=0.25", "training.val_strategy=tail"]
    cfg, paths = _staged_run(tmp_path, split)
    clean_anchor = BubbleDataset(cfg, paths).pin_anchor

    archive = dict(np.load(paths.tensors))
    meta = archive.pop("meta")
    # Shift the last (held-out) frame's bubble far downstream: a would-be leak.
    last = np.zeros_like(archive["alpha"][-1])
    last[:, -3:] = 1.0
    archive["alpha"][-1] = last
    np.savez_compressed(paths.tensors, **archive, meta=meta)
    corrupted = BubbleDataset(cfg, paths)

    assert corrupted.split_rows == [GROWING_FRAMES - 1], "tail split must hold the last frame"
    assert corrupted.pin_anchor == clean_anchor


def test_anchor_fails_loud_when_no_frame_has_vapor(tmp_path):
    from naviernet.data.dataset import BubbleDataset

    cfg, paths = _staged_run(tmp_path)
    archive = dict(np.load(paths.tensors))
    meta = archive.pop("meta")
    archive["alpha"] = np.zeros_like(archive["alpha"])
    np.savez_compressed(paths.tensors, **archive, meta=meta)

    with pytest.raises(ValueError, match="no training frame"):
        _ = BubbleDataset(cfg, paths).pin_anchor


# --- The pin gate (model layer) ----------------------------------------------


def test_pin_gate_is_transparent_far_from_the_anchor(tmp_path):
    """Beyond a few d_ref the gate saturates to ~1: the far field is untouched."""
    from naviernet.models.pinn import BubblePINN

    cfg, paths = _staged_run(tmp_path, ["model.hard_pin=true"])
    model = BubblePINN(cfg, pin=(0.2, 0.25))
    far = torch.tensor([[0.9, 0.25, 0.1], [0.2, 0.25 + 0.7, 0.3]], dtype=torch.float32)

    ratio = model.phi(far) / model.nets["phi"](far)

    assert torch.all(ratio > 0.999), f"far-field gate should be ~1, got {ratio.ravel()}"


def test_pin_gradients_are_finite_at_the_anchor(tmp_path):
    """A collocation point exactly on the anchor must not NaN the backward pass
    (the raw distance has no gradient at zero; the floored form does)."""
    from naviernet.models.pinn import BubblePINN

    cfg, _ = _staged_run(tmp_path, ["model.hard_pin=true"])
    model = BubblePINN(cfg, pin=(0.2, 0.25))
    x = torch.tensor(
        [[0.2, 0.25, 0.1], [0.4, 0.3, 0.2]], dtype=torch.float32, requires_grad=True
    )

    model.alpha(x).sum().backward()

    assert torch.isfinite(x.grad).all(), f"NaN/inf gradient at the anchor: {x.grad}"
    grads = [p.grad for p in model.nets["phi"].parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("d_ref", [0.0, -0.1])
def test_pin_rejects_a_nonpositive_d_ref(tmp_path, d_ref):
    from naviernet.models.pinn import BubblePINN

    cfg, _ = _staged_run(tmp_path, ["model.hard_pin=true", f"model.pin_d_ref={d_ref}"])

    with pytest.raises(ValueError, match="pin_d_ref"):
        BubblePINN(cfg, pin=(0.2, 0.25))


def test_pin_keeps_the_stage_b_surface_tension_term_bounded_near_the_anchor(tmp_path):
    """Regression for the gate's smoothness: the momentum surface-tension quantity
    kappa * grad(alpha) must stay bounded arbitrarily close to the anchor. A gate
    built on raw Euclidean distance has a cusp there whose curvature diverges like
    1/d (measured: ~3e6 at d=1e-6); the shipped quadratic-argument gate keeps the
    product bounded because grad(alpha) vanishes as fast as kappa grows."""
    from naviernet.models.pinn import BubblePINN
    from naviernet.physics.residuals import curvature

    cfg, _ = _staged_run(tmp_path, ["model=stage_b", "model.hard_pin=true"])
    anchor = (0.75, 0.25)
    torch.manual_seed(0)
    model = BubblePINN(cfg, pin=anchor)

    for d in (0.0, 1e-6, 1e-4, 1e-3):
        x = torch.tensor([[anchor[0] + d, anchor[1], 0.2]], requires_grad=True)
        alpha = model.alpha(x)
        kappa = curvature(alpha, x)
        a_x = torch.autograd.grad(alpha.sum(), x, create_graph=True)[0][0, 0]
        product = float((kappa * a_x).detach())
        assert math.isfinite(product) and abs(product) < 1e3, (
            f"kappa*a_x = {product:.3e} at d={d:g} -- the pin gate has a cusp"
        )


def test_pin_requires_an_anchor(tmp_path):
    from naviernet.models.pinn import BubblePINN

    cfg, _ = _staged_run(tmp_path, ["model.hard_pin=true"])

    with pytest.raises(ValueError, match="anchor"):
        BubblePINN(cfg)


def test_training_with_the_pin_diverges_from_baseline(tmp_path):
    """The transform must actually reach the training graph: identical seeds with
    the pin on vs off produce different fitted models."""
    from naviernet.training import train

    cfg_off, paths_off = _staged_run(tmp_path / "off")
    train(cfg_off, paths_off)
    cfg_on, paths_on = _staged_run(tmp_path / "on", ["model.hard_pin=true"])
    train(cfg_on, paths_on)

    off = torch.load(paths_off.checkpoint, weights_only=False)["model"]
    on = torch.load(paths_on.checkpoint, weights_only=False)["model"]

    assert any(not torch.equal(off[k], on[k]) for k in off), (
        "pin on vs off trained identically -- the gate never entered the loss"
    )


def test_resumed_pin_run_keeps_the_anchor(tmp_path):
    """Chunked training (naviernet's default execution model): a hard-pin run
    resumed from its checkpoint re-derives the same anchor and keeps the
    interface exactly on it."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, ["model.hard_pin=true", "training.steps=1"])
    train(cfg, paths)
    train(cfg, paths)  # resume: continues from the checkpoint
    model, data, state = load_model(cfg, paths)

    assert state["done"] == 2, "the second call must resume, not restart"
    x0, y0 = data.pin_anchor
    point = torch.tensor([[x0, y0, 2.0 * data.domain.t_max]], dtype=torch.float32)
    assert torch.allclose(model.alpha(point), torch.tensor([[0.5]]), atol=1e-6)


def test_checkpoint_refuses_a_pin_flag_mismatch(tmp_path):
    """The pin gate adds no parameters, so load_state_dict cannot catch a config
    mismatch -- the checkpoint records the pin architecture and any consumer
    composed differently fails loudly instead of silently unpinning the fields."""
    from naviernet.training import load_model, train

    cfg_on, paths = _staged_run(tmp_path, ["model.hard_pin=true"])
    train(cfg_on, paths)

    cfg_off = make_config([f"paths.root={tmp_path}", "training.holdout_frame=-1"])

    with pytest.raises(ValueError, match="hard_pin"):
        load_model(cfg_off, paths)
    with pytest.raises(ValueError, match="hard_pin"):
        train(cfg_off, paths)  # resuming with the flag dropped is the same mismatch


def test_checkpoint_refuses_a_d_ref_mismatch(tmp_path):
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, ["model.hard_pin=true", "model.pin_d_ref=0.1"])
    train(cfg, paths)
    drifted = make_config(
        [
            f"paths.root={tmp_path}",
            "training.holdout_frame=-1",
            "model.hard_pin=true",
            "model.pin_d_ref=0.2",
        ]
    )

    with pytest.raises(ValueError, match="pin_d_ref"):
        load_model(drifted, paths)


def test_pin_composes_with_rba_weighting(tmp_path):
    """The pin is applied inside phi(), so it must reach RBA's fixed-pool loss
    path too: a 2-step RBA + pin run trains and holds the anchor."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, ["model.hard_pin=true", "training.weighting=rba"])
    train(cfg, paths)
    model, data, _ = load_model(cfg, paths)

    x0, y0 = data.pin_anchor
    point = torch.tensor([[x0, y0, 2.0 * data.domain.t_max]], dtype=torch.float32)
    assert torch.allclose(model.alpha(point), torch.tensor([[0.5]]), atol=1e-6)


# --- Root-position metric (bench) --------------------------------------------


def test_root_position_picks_the_edge_nearer_the_anchor():
    """Of a mask's two x-extent edges, the root is the one nearer the dataset's
    measured anchor -- in either orientation; an empty mask is NaN."""
    from naviernet.evaluation import root_position

    xs = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    mask = np.zeros((3, 11), dtype=bool)
    mask[1, 2:8] = True  # bubble spanning x* 0.2 .. 0.7

    assert root_position(mask, xs, x_anchor=0.15) == pytest.approx(0.2)
    assert root_position(mask, xs, x_anchor=0.75) == pytest.approx(0.7)
    assert math.isnan(root_position(np.zeros((3, 11), dtype=bool), xs, x_anchor=0.5))


# --- Joint (multi-dataset) runs ----------------------------------------------


def test_joint_hard_pin_anchors_each_dataset_at_its_own_root(tmp_path):
    """One conditioned model, two datasets with different roots: each dataset's
    bound view holds alpha = 0.5 at ITS anchor for all t, anchors distinct."""
    from naviernet.training import load_joint, train

    cfg, paths = _staged_joint_run(tmp_path, ["model.hard_pin=true"])
    train(cfg, paths)
    model, contexts, _ = load_joint(cfg, paths)

    anchors = [cx.pin for cx in contexts]
    assert anchors[0][0] != anchors[1][0], "distinct roots must give distinct anchors"

    for cx in contexts:
        bound = model.bound(cx.c, pin=cx.pin)
        x0, y0 = cx.pin
        t_beyond = 3.0 * cx.data.domain.t_max + 1.0
        points = torch.tensor([[x0, y0, 0.0], [x0, y0, t_beyond]], dtype=torch.float32)
        alpha = bound.alpha(points)
        assert torch.allclose(alpha, torch.full_like(alpha, 0.5), atol=1e-6), (
            f"{cx.name}: pinned alpha at its anchor should be 0.5, got {alpha.ravel()}"
        )


def test_joint_hard_pin_model_refuses_unbound_calls(tmp_path):
    """Zero silent failures: a joint hard-pin model evaluated without a bound
    anchor raises instead of quietly predicting unpinned fields."""
    from naviernet.training import load_joint, train

    cfg, paths = _staged_joint_run(tmp_path, ["model.hard_pin=true"])
    train(cfg, paths)
    model, contexts, _ = load_joint(cfg, paths)

    points = torch.tensor([[0.2, 0.25, 0.1]], dtype=torch.float32)
    c = contexts[0].c.expand(1, -1)

    with pytest.raises(RuntimeError, match="anchor"):
        model.alpha(points, c)


def test_joint_hard_pin_composes_with_causal_weighting(tmp_path):
    """The bench composition: joint + pin + causal trains, and the causal path's
    per-bin contexts (rebuilt from the collocation batches) kept each dataset's
    anchor -- verified by the pinned interface surviving the fitted model."""
    from naviernet.training import load_joint, train

    cfg, paths = _staged_joint_run(
        tmp_path, ["model.hard_pin=true", "training.causal_weighting=true"]
    )
    train(cfg, paths)
    model, contexts, _ = load_joint(cfg, paths)

    for cx in contexts:
        x0, y0 = cx.pin
        point = torch.tensor([[x0, y0, 2.0 * cx.data.domain.t_max]], dtype=torch.float32)
        alpha = model.bound(cx.c, pin=cx.pin).alpha(point)
        assert torch.allclose(alpha, torch.tensor([[0.5]]), atol=1e-6), (
            f"{cx.name}: anchor lost through the causal collocation path"
        )


def test_joint_hard_pin_scores_a_heldout_transfer_dataset(tmp_path):
    """Axis-B transfer: a dataset held out of training entirely still evaluates
    through its own pin-bound view (its anchor comes from its own data, which the
    model never saw gradients from)."""
    from naviernet.evaluation import evaluate_joint
    from naviernet.training import load_joint, train

    cfg, paths = _staged_joint_run(tmp_path, ["model.hard_pin=true", "heldout_datasets=[ds_b]"])
    train(cfg, paths)
    model, contexts, heldout = load_joint(cfg, paths)

    report = evaluate_joint(cfg, model, contexts, paths, heldout_datasets=heldout)

    assert heldout == ["ds_b"]
    assert list(report["per_dataset"]) == ["ds_a"]
    assert "ds_b" in report["transfer"]["per_dataset"]


def test_joint_hard_pin_evaluates(tmp_path):
    """evaluate_joint runs the pinned views end-to-end and writes metrics."""
    import json as json_mod

    from naviernet.evaluation import evaluate_joint
    from naviernet.training import load_joint, train

    cfg, paths = _staged_joint_run(tmp_path, ["model.hard_pin=true"])
    train(cfg, paths)
    model, contexts, heldout = load_joint(cfg, paths)

    report = evaluate_joint(cfg, model, contexts, paths, heldout_datasets=heldout)

    assert set(report["per_dataset"]) == {spec[0] for spec in JOINT_SPECS}
    assert paths.metrics_json.exists()
    assert json_mod.loads(paths.metrics_json.read_text())["datasets"]
