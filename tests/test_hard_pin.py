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
import torch

from naviernet.utils.paths import RunPaths
from tests.conftest import make_config

GROWING_H, GROWING_W, GROWING_FRAMES = 6, 12, 4
ROOT_COL = 2  # the stationary (nucleation-side) bubble edge, all frames
VAPOR_ROWS = slice(2, 4)  # the bubble's y-extent at the root


def _write_growing_bubble(path, n_frames: int = GROWING_FRAMES) -> None:
    """A synthetic growth event: the root edge fixed at ROOT_COL, the front
    advancing one column per frame -- the geometry the anchor measurement is for."""
    alpha = np.zeros((n_frames, GROWING_H, GROWING_W), dtype=np.float32)
    for k in range(n_frames):
        alpha[k, VAPOR_ROWS, ROOT_COL : ROOT_COL + 3 + k] = 1.0
    meta = {
        "x_pin_star": 0.2,
        "t_ref_ms": 1.5,
        "n_frames_usable": n_frames,
        "n_frames_event": n_frames,
        "frame_numbers": list(range(1, n_frames + 1)),
    }
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
