"""The learning-rate schedule, and its faster clock for a compressed physics
stage (``training.physics_lr_halflife``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from naviernet.training import learning_rate


def _tcfg(**overrides):
    base = dict(
        lr=2e-3,
        lr_halflife=2500,
        stage_b_warmup_steps=3000,
        physics_lr_halflife=0,
        temperature_stage_from=0,
    )
    return SimpleNamespace(**{**base, **overrides})


def test_one_clock_halves_every_lr_halflife_steps():
    tcfg = _tcfg()

    assert learning_rate(2499, tcfg) == pytest.approx(2e-3)
    assert learning_rate(2500, tcfg) == pytest.approx(1e-3)
    assert learning_rate(12000, tcfg) == pytest.approx(2e-3 / 16)


def test_a_compressed_physics_stage_sweeps_the_full_length_rates():
    # 2250 physics steps at a quarter of the halving period cover the rates the
    # full run's 9000 physics steps do, step for scaled step.
    full, short = _tcfg(), _tcfg(physics_lr_halflife=625)

    assert learning_rate(3000, short) == learning_rate(3000, full), "warm-up untouched"
    for physics_step in (1, 500, 1000, 1749, 1750, 2250):
        assert learning_rate(3000 + physics_step, short) == pytest.approx(
            learning_rate(3000 + 4 * physics_step, full)
        )
