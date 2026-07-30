"""Hard root pin: the level-set output transform anchoring the bubble root.

The pin is architectural -- ``phi = tanh(dist/d_ref) * N`` -- so the interface
(alpha = 0.5) passes through the dataset's measured root anchor at every time,
including times beyond the training window. These tests drive the real layers:
synthetic tensors -> anchor measurement -> model transform -> trainer ->
checkpoint -> reloaded model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from naviernet.utils.paths import RunPaths
from tests.conftest import make_config

GROWING_H, GROWING_W, GROWING_FRAMES = 6, 12, 4
ROOT_COL = 2  # the stationary (nucleation-side) bubble edge, all frames
VAPOR_ROWS = slice(2, 4)  # the bubble's y-extent at the root


def _write_growing_bubble(
    path, n_frames: int = GROWING_FRAMES, root_col: int = ROOT_COL, groups=None
) -> None:
    """A synthetic growth event: the root edge fixed at ``root_col``, the front
    advancing one column per frame -- the geometry the anchor measurement is for.
    ``groups`` (when given) is recorded as preprocess would, so the dataset can
    join a conditioned multi-dataset run."""
    alpha = np.zeros((n_frames, GROWING_H, GROWING_W), dtype=np.float32)
    for k in range(n_frames):
        alpha[k, VAPOR_ROWS, root_col : root_col + 3 + k] = 1.0
    meta = {
        "x_pin_star": 0.2,
        "t_ref_ms": 1.5,
        "n_frames_usable": n_frames,
        "n_frames_event": n_frames,
        "frame_numbers": list(range(1, n_frames + 1)),
    }
    if groups is not None:
        meta["groups"] = groups
    np.savez_compressed(
        path,
        alpha=alpha,
        sdf=((0.5 - alpha) * 0.1).astype(np.float32),
        valid=np.ones_like(alpha),
        masks_camera=(alpha > 0.5).astype(np.uint8),
        x_star=np.linspace(0, 1.1, GROWING_W, dtype=np.float32),
        y_star=np.linspace(0, 0.5, GROWING_H, dtype=np.float32),
        t_star=(np.arange(n_frames) * 0.1).astype(np.float32),
        meta=json.dumps(meta),
    )


def _staged_run(tmp_path, overrides=None):
    """Compose a tiny run over the synthetic growth event, as the CLI would."""
    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            "model.hidden=8",
            "model.layers=2",
            "model.fourier_feats=4",
            "training.steps=2",
            "training.n_data=16",
            "training.n_coll=16",
            "training.n_bc=8",
            "training.holdout_frame=-1",
            *(overrides or []),
        ]
    )
    paths = RunPaths.from_config(cfg)
    paths.ensure()
    paths.tensors.parent.mkdir(parents=True, exist_ok=True)
    _write_growing_bubble(paths.tensors)
    return cfg, paths


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
    mirrored_root = GROWING_W - 1 - ROOT_COL

    assert data.pin_anchor[0] == pytest.approx(float(x_star[mirrored_root]))


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


def test_pin_rejects_a_nonpositive_d_ref(tmp_path):
    from naviernet.models.pinn import BubblePINN

    cfg, _ = _staged_run(tmp_path, ["model.hard_pin=true", "model.pin_d_ref=0"])

    with pytest.raises(ValueError, match="pin_d_ref"):
        BubblePINN(cfg, pin=(0.2, 0.25))


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


# --- Joint (multi-dataset) runs ----------------------------------------------

JOINT_SPECS = (("ds_a", 2.0, 2), ("ds_b", 5.0, 6))  # name, q_wall, root column


def _staged_joint_run(tmp_path, overrides=None):
    """Two synthetic datasets with distinct regimes AND distinct root columns,
    staged for a conditioned joint run, as preprocess would have left them."""
    from naviernet.physics.groups import compute_groups

    names = ",".join(name for name, _, _ in JOINT_SPECS)
    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            f"datasets=[{names}]",
            "model.hidden=8",
            "model.layers=2",
            "model.fourier_feats=4",
            "training.steps=2",
            "training.n_data=16",
            "training.n_coll=16",
            "training.n_bc=8",
            "training.holdout_frame=-1",
            *(overrides or []),
        ]
    )
    paths = RunPaths.from_config(cfg)
    paths.ensure()
    for name, q_wall, root_col in JOINT_SPECS:
        ds_paths = paths.for_dataset(name)
        ds_paths.processed_dir.mkdir(parents=True, exist_ok=True)
        groups = compute_groups(make_config([f"experiment.q_wall_W_cm2={q_wall}"]))
        _write_growing_bubble(ds_paths.tensors, root_col=root_col, groups=groups)
    return cfg, paths


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
    """The bench composition: joint + pin + causal must train (the causal path
    rebuilds per-bin contexts from the collocation batches -- each must keep its
    dataset's anchor)."""
    from naviernet.training import train

    cfg, paths = _staged_joint_run(
        tmp_path, ["model.hard_pin=true", "training.causal_weighting=true"]
    )
    train(cfg, paths)

    assert paths.checkpoint.exists()


def test_joint_hard_pin_evaluates(tmp_path):
    """evaluate_joint runs the pinned views end-to-end and writes metrics."""
    import json as json_mod

    from naviernet.evaluation import evaluate_joint
    from naviernet.training import load_joint, train

    cfg, paths = _staged_joint_run(tmp_path, ["model.hard_pin=true"])
    train(cfg, paths)
    model, contexts, heldout = load_joint(cfg, paths)

    report = evaluate_joint(cfg, model, contexts, paths, heldout_datasets=heldout)

    assert set(report["per_dataset"]) == {name for name, _, _ in JOINT_SPECS}
    assert paths.metrics_json.exists()
    assert json_mod.loads(paths.metrics_json.read_text())["datasets"]
