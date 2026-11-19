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


def test_a_physics_floor_holds_the_rate_once_the_schedule_reaches_it():
    # ghost10's schedule: 1.25e-4 from step 7200, halved again at 8700.
    falling = _tcfg(physics_lr_halflife=1500)
    held = _tcfg(physics_lr_halflife=1500, physics_lr_floor=1.25e-4)

    assert learning_rate(8700, falling) == pytest.approx(6.25e-5)
    for step in (8700, 10000, 12000):
        assert learning_rate(step, held) == pytest.approx(1.25e-4)


def test_a_physics_floor_leaves_the_rates_above_it_alone():
    falling = _tcfg(physics_lr_halflife=1500)
    held = _tcfg(physics_lr_halflife=1500, physics_lr_floor=1.25e-4)

    for step in (1, 2500, 3000, 3001, 4200, 5700, 7200, 8000, 8699):
        assert learning_rate(step, held) == learning_rate(step, falling)


def test_a_physics_floor_does_not_lift_the_warm_up():
    lifted = _tcfg(physics_lr_floor=5e-3)

    assert learning_rate(3000, lifted) == pytest.approx(1e-3), "warm-up keeps its schedule"
    assert learning_rate(3001, lifted) == pytest.approx(5e-3)
