"""Kinematic growth constraints: physics-only late-window functionals.

The R2 terms — monotone bubble volume, volume-vs-source balance, and the
evaporation floor — are scalar functionals on a fixed quadrature grid over the
late time window, normalized by the dataset's supervised-tail growth rate.
These tests drive the real layers: synthetic tensors -> quadrature/references ->
terms -> trainer -> history record.
"""

from __future__ import annotations

import math

from tests.test_hard_pin import _staged_run

TINY_KIN = ["training.kinematics=true", "training.kin_grid=6", "training.kin_times=3"]


def test_kinematics_terms_train_and_are_logged(tmp_path):
    """The walking skeleton: a tiny run with the flag on trains, and the kin
    term values appear in the logged history record."""
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, [*TINY_KIN, "training.log_every=1"])
    _, _, state = train(cfg, paths)

    record = state["hist"][-1]
    assert "kin_mono" in record, f"kin terms missing from the history record: {record.keys()}"
    assert math.isfinite(record["kin_mono"])


def test_kinematics_off_leaves_the_record_clean(tmp_path):
    """Flag off (the default): no kin terms anywhere in the history."""
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, ["training.log_every=1"])
    assert cfg.training.kinematics is False, "kinematics must default off"
    _, _, state = train(cfg, paths)

    assert not any(k.startswith("kin_") for k in state["hist"][-1])
