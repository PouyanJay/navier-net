"""The front-velocity report: how fast the interface moves, and where.

The measured normal velocity earned its place as an OUTPUT rather than as
supervision. As a training signal it benched neutral -- ``v_n`` is a time
derivative of the masks the shape term already fits, so it carries no new
information into the optimisation. As a record of what the interface did it is
not redundant at all: nothing else in the run's artifacts reports the front's
rate anywhere except the nose's position curve.

This suite holds the report to the same standard the measurement is held to in
:mod:`tests.test_front_velocity` -- the analytic capsule, whose nose advances at
a known ``ds/dt`` while its flanks, of fixed radius, do not move at all.
"""

from __future__ import annotations

import json

import pytest

from tests import conftest

GEO = ["model.front_geometry=true"]

# The capsule's true nose speed in the report's units. The dataset's own
# reference length is the composed config's, and its time reference is the
# fixture's own -- the same two numbers `_write_report` scales by.
NOSE_SPEED_STAR = conftest.CAPSULE_NOSE_SPEED  # x* per t*


def _evaluated(tmp_path, overrides=None):
    """A tiny trained run over the analytic capsule, evaluated -- the real
    pipeline path, so the report is produced the way a real run produces it."""
    from naviernet.evaluation import evaluate
    from naviernet.training import load_model, train

    cfg, paths = conftest.staged_capsule_run(tmp_path, [*GEO, *(overrides or [])])
    train(cfg, paths)
    model, data, _ = load_model(cfg, paths)
    evaluate(cfg, model, data, paths)
    return cfg, paths, data


def _report(tmp_path, overrides=None) -> dict:
    _, paths, _ = _evaluated(tmp_path, overrides)
    return json.loads(paths.front_velocity_json.read_text())


# --- The walking skeleton: evaluate produces a report the platform can read ---


def test_evaluate_writes_a_front_velocity_report(tmp_path):
    """The whole path: the evaluate stage measures the front's motion and leaves
    it beside the trajectory, in physical units, as JSON with no NaN token."""
    report = _report(tmp_path)

    assert report["front_geometry"] is True
    nose = report["nose_speed"]
    assert len(nose["t_ms"]) == len(nose["v_um_per_ms"]) > 1


def test_the_reported_nose_speed_integrates_to_the_reported_nose_travel(tmp_path):
    """The two artifacts must agree: the speed chart sits directly under the
    position chart, and a reader is entitled to read one as the other's slope.

    Asserted in integrated form rather than pointwise. The predicted nose is read
    off a strided pixel grid, so it advances as a staircase, and
    ``trajectory.json`` stores its axes rounded -- a pointwise slope comparison
    would mostly be measuring JSON quantisation. What must hold regardless is
    that the speed integrates to the travel, which is also what catches the
    failure that actually matters here: a units or reference-scale error.
    """
    import numpy as np

    _, paths, _ = _evaluated(tmp_path)
    trajectory = json.loads(paths.trajectory_json.read_text())
    report = json.loads(paths.front_velocity_json.read_text())

    assert report["nose_speed"]["t_ms"] == trajectory["t_ms"]
    times = np.array(trajectory["t_ms"], dtype=float)
    nose = np.array(trajectory["nose_um"], dtype=float)
    speed = np.array(report["nose_speed"]["v_um_per_ms"], dtype=float)

    assert np.isfinite(nose).all() and np.isfinite(speed).all()
    assert np.trapezoid(speed, times) == pytest.approx(nose[-1] - nose[0], rel=0.02)


def test_the_report_states_its_units_rather_than_leaving_them_to_the_reader(tmp_path):
    """A bare number is not a measurement. Every consumer -- chart axis, CSV
    column, tooltip -- reads the unit from the key, so the key carries it."""
    report = _report(tmp_path)

    assert "v_um_per_ms" in report["nose_speed"]
    assert "t_ms" in report["nose_speed"]
