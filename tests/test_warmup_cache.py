"""The shared warm-up (``training.warmup_cache``): trained once per outline,
settings and code, reused by every run that agrees on them."""

from __future__ import annotations

import numpy as np
import torch

from naviernet import training, warmup_cache

from .conftest import staged_capsule_run

STAGED = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.momentum_closure=brinkman",
    "training.stage_b_warmup_steps=2",
    "training.warmup_cache=true",
    "training.log_every=1",
]


def _run(tmp_path, name: str, extra=(), steps: int = 4, edit_archive=None):
    cfg, paths = staged_capsule_run(tmp_path, [*STAGED, f"run_name={name}", *extra])
    if edit_archive is not None:
        archive = dict(np.load(paths.tensors))
        edit_archive(archive)
        np.savez(paths.tensors, **archive)
    model, _, state = training.train(cfg, paths, steps=steps)
    return cfg, paths, model, state


def _entries(cfg, paths) -> list:
    root = warmup_cache.cache_dir(cfg, paths)
    return sorted(p.name for p in root.iterdir()) if root.exists() else []


def test_a_run_that_reuses_the_warm_up_ends_where_one_that_trained_it_does(tmp_path):
    cfg, paths, first, state = _run(tmp_path, "first")
    _, _, second, second_state = _run(tmp_path, "second")

    assert len(_entries(cfg, paths)) == 1, "one warm-up, trained once"
    assert state["done"] == second_state["done"] == 4
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key


def test_a_physics_only_change_shares_the_warm_up(tmp_path):
    cfg, paths, *_ = _run(tmp_path, "base")
    _run(
        tmp_path,
        "variant",
        [
            "training.weights.brinkman=7.0",
            "model.momentum_inertia=true",
            "model.hard_kinematics=true",
        ],
    )

    assert len(_entries(cfg, paths)) == 1


def test_a_warm_up_setting_or_new_outlines_train_a_new_warm_up(tmp_path):
    cfg, paths, *_ = _run(tmp_path, "base")
    _run(tmp_path, "data_weight", ["training.weights.data=3.0"])
    assert len(_entries(cfg, paths)) == 2

    def retrace(archive):
        archive["alpha"] = archive["alpha"][:, ::-1].copy()

    _run(tmp_path, "new_outline", edit_archive=retrace)
    assert len(_entries(cfg, paths)) == 3


def test_a_run_too_short_to_finish_its_warm_up_is_not_cached(tmp_path):
    cfg, paths, *_ = _run(tmp_path, "short", steps=1)

    assert _entries(cfg, paths) == []


def test_a_resumed_run_continues_its_own_checkpoint(tmp_path):
    cfg, paths, _, state = _run(tmp_path, "resumed", steps=4)
    _, _, _, again = _run(tmp_path, "resumed", steps=2)

    assert again["done"] == state["done"] + 2
