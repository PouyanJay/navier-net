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


def _report_without_front_geometry(tmp_path) -> dict:
    """The same run with no explicit front -- every block that needs one is
    absent, and the blocks read off the predicted mask survive."""
    from naviernet.evaluation import evaluate
    from naviernet.training import load_model, train

    cfg, paths = conftest.staged_capsule_run(tmp_path, ["model=stage_b"])
    train(cfg, paths)
    model, data, _ = load_model(cfg, paths)
    evaluate(cfg, model, data, paths)
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


# --- Which pairs a REPORT may difference, versus which supervision may --------


def test_the_report_pairs_every_consecutive_frame_including_the_held_out_one(tmp_path):
    """The loss drops a pair touching a held-out frame -- differencing it would
    make the validation set a training target. Nothing here trains, so no such
    leak exists, and the held-out interval is precisely the one a reader wants
    the model's rate over. It is marked, not dropped.
    """
    from naviernet.data.dataset import BubbleDataset
    from naviernet.utils.paths import RunPaths

    cfg, _ = conftest.staged_capsule_run(tmp_path, [*GEO, "training.holdout_frame=2"])
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")

    assert data.validation_rows == [2]
    assert data.report_pairs == [(k, k + 1) for k in range(conftest.CAPSULE_FRAMES - 1)]
    # The two pairs that touch it are exactly the ones supervision refuses.
    assert set(data.report_pairs) - set(data.supervised_pairs) == {(1, 2), (2, 3)}


def test_a_pair_across_an_excluded_camera_frame_is_never_reported(tmp_path):
    """A gap on the time axis is a real gap. A difference taken across it is a
    coarser estimate of a different interval -- not the same measurement, so not
    the same chart."""
    import numpy as np

    from naviernet.data.dataset import BubbleDataset
    from naviernet.utils.paths import RunPaths

    cfg, paths = conftest.staged_capsule_run(tmp_path, GEO)
    archive = dict(np.load(paths.tensors, allow_pickle=False))
    meta = json.loads(str(archive["meta"]))
    meta["frame_numbers"] = [1, 2, 3, 5, 6, 7]  # camera frame 4 excluded
    archive["meta"] = json.dumps(meta)
    np.savez_compressed(paths.tensors, **archive)

    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    assert (2, 3) not in data.report_pairs
    assert data.report_pairs == [(0, 1), (1, 2), (3, 4), (4, 5)]


# --- What the masks say the nose did -----------------------------------------


def test_the_measured_nose_speed_recovers_the_capsule_s_true_growth(tmp_path):
    """The analytic answer: this capsule's nose advances at a known rate.

    Asserted on the mean over pairs rather than pair by pair. The measured nose
    is the rightmost mask column, so a single pair's difference is quantised to
    the pixel grid -- here ~4.75 pixels of travel, so ±21% on one pair but ~4%
    over the whole span.
    """
    import numpy as np

    cfg, paths, _ = _evaluated(tmp_path)
    report = json.loads(paths.front_velocity_json.read_text())

    measured = report["nose_speed"]["measured"]
    expected = NOSE_SPEED_STAR * cfg.scales.L_ref_um / conftest.CAPSULE_T_REF_MS
    assert len(measured["t_ms"]) == conftest.CAPSULE_FRAMES - 1
    assert np.mean(measured["v_um_per_ms"]) == pytest.approx(expected, rel=0.06)


def test_each_measured_pair_says_whether_it_spans_a_held_out_frame(tmp_path):
    """Marked, per decision: a reader must be able to tell the intervals the
    model was shown from the interval it was not."""
    report = _report(tmp_path, ["training.holdout_frame=2"])

    measured = report["nose_speed"]["measured"]
    assert len(measured["heldout"]) == len(measured["t_ms"])
    # Rows 1-2 and 2-3 touch the held-out frame; the rest do not.
    assert measured["heldout"] == [False, True, True, False, False]


def test_a_measured_pair_is_reported_at_the_instant_it_spans(tmp_path):
    """A finite difference measures the interval, not either endpoint, so it is
    plotted at the midpoint -- otherwise every measured point sits half a frame
    away from the continuous curve it is being compared against."""
    cfg, paths, data = _evaluated(tmp_path)
    report = json.loads(paths.front_velocity_json.read_text())

    t_ref = conftest.CAPSULE_T_REF_MS
    expected = [
        0.5 * (float(data.t[row]) + float(data.t[nxt])) * t_ref
        for row, nxt in data.report_pairs
    ]
    assert report["nose_speed"]["measured"]["t_ms"] == pytest.approx(expected, abs=1e-4)


# --- The apex: the one point with an honest 2-D velocity ---------------------


def test_the_measured_apex_velocity_recovers_the_capsule_s_growth_in_both_axes(tmp_path):
    """The capsule advances along +x at a known rate and never moves in y.

    The apex is the one interface point with a real frame-to-frame
    correspondence, so unlike the normal velocity this is a true ``(vx, vy)`` --
    and both components have an analytic answer here.
    """
    import numpy as np

    cfg, paths, _ = _evaluated(tmp_path)
    measured = json.loads(paths.front_velocity_json.read_text())["apex"]["measured"]

    scale = cfg.scales.L_ref_um / conftest.CAPSULE_T_REF_MS
    assert np.mean(measured["vx_um_per_ms"]) == pytest.approx(NOSE_SPEED_STAR * scale, rel=0.06)
    assert np.allclose(measured["vy_um_per_ms"], 0.0, atol=0.02 * scale)


def test_the_model_apex_velocity_is_the_derivative_of_its_own_apex_path(tmp_path):
    """Taken by autodiff, not differenced: the apex is a differentiable function
    of time under this parameterisation, so the report gives its exact velocity
    rather than an estimate whose error depends on a step size nobody chose.

    Held to that claim by comparing against a central difference of the model's
    OWN apex -- if the reported values were themselves a coarse difference over
    the report's grid, they would not match one taken over a step three orders
    of magnitude smaller.

    The step is chosen, not guessed: the model is float32, so a central
    difference trades truncation (growing with the step) against cancellation
    (~eps*|apex|/step, shrinking with it). At 1e-3 t* both sit near 1e-2 µm/ms
    for this fixture, which is what the tolerance below allows for -- it is the
    REFERENCE's precision, not the report's.
    """
    import numpy as np
    import torch

    from naviernet.training import load_model

    cfg, paths, data = _evaluated(tmp_path)
    model, _, _ = load_model(cfg, paths)
    apex = json.loads(paths.front_velocity_json.read_text())["apex"]

    t_ref = conftest.CAPSULE_T_REF_MS
    scale = cfg.scales.L_ref_um / t_ref
    step = 1e-3
    for index in (1, len(apex["t_ms"]) // 2, len(apex["t_ms"]) - 2):
        t_star = apex["t_ms"][index] / t_ref
        with torch.no_grad():
            ahead = model.apex(torch.tensor([[t_star + step]], dtype=torch.float32))
            behind = model.apex(torch.tensor([[t_star - step]], dtype=torch.float32))
        expected = ((ahead - behind) / (2 * step)).reshape(2).numpy() * scale

        reported = np.array([apex["vx_um_per_ms"][index], apex["vy_um_per_ms"][index]])
        assert reported == pytest.approx(expected, rel=1e-3, abs=1e-4 * scale)


def test_a_run_without_an_explicit_front_reports_no_apex(tmp_path):
    """No parameterised nose to differentiate. The key is present and null, so
    the view says why rather than drawing an empty axis."""
    report = _report_without_front_geometry(tmp_path)

    assert report["front_geometry"] is False
    assert report["apex"] is None
    # The nose speed survives -- it is read off the predicted mask, not the front.
    assert report["nose_speed"]["v_um_per_ms"]


# --- The profile along the front ---------------------------------------------


def test_the_profile_walks_the_closed_front_once_in_order(tmp_path):
    """``s`` is one traversal of the closed contour, so the profile chart has a
    continuous axis and the kymograph a continuous one to stack against.

    Root cap, up the upper body to the nose, around the nose cap, back down the
    lower body -- four contiguous spans covering the whole curve exactly once.
    """
    import numpy as np

    profile = _report(tmp_path)["profile"]

    assert [s["name"] for s in profile["segments"]] == [
        "root_cap",
        "upper_body",
        "nose_cap",
        "lower_body",
    ]
    starts = [s["s_start"] for s in profile["segments"]]
    ends = [s["s_end"] for s in profile["segments"]]
    assert starts[0] == 0.0
    assert ends[-1] == pytest.approx(1.0)
    assert starts[1:] == pytest.approx(ends[:-1])  # contiguous, no overlap or gap
    # Every bin centre falls inside the traversal, strictly increasing.
    assert np.all(np.diff(profile["s"]) > 0)
    assert profile["s"][0] > 0.0 and profile["s"][-1] < 1.0


def test_the_measured_profile_is_suppressed_exactly_on_the_nose_cap(tmp_path):
    """The level-set estimate is first-order in the distance the front travels,
    and the nose is where that distance is largest against the smallest radius --
    it read +0.489 there against a true +0.295 on Series-1. The chart shows a gap
    rather than a curve known to be wrong, and the segment says why.

    The same validity rule the loss follows, from the same function, so the two
    cannot drift apart.
    """
    profile = _report(tmp_path)["profile"]

    named = {s["name"]: s for s in profile["segments"]}
    assert named["nose_cap"]["measured"] is False
    assert all(named[name]["measured"] for name in ("root_cap", "upper_body", "lower_body"))

    nose = range(named["nose_cap"]["bin_start"], named["nose_cap"]["bin_end"])
    for frame in profile["times"]:
        assert all(frame["measured"][b] is None for b in nose)
        # And the suppression is confined to it: the body still measures.
        body = range(named["upper_body"]["bin_start"], named["upper_body"]["bin_end"])
        assert any(frame["measured"][b] is not None for b in body)


def test_there_is_one_profile_per_reportable_frame_pair(tmp_path):
    """Including the pairs supervision refused, marked as such."""
    report = _report(tmp_path, ["training.holdout_frame=2"])

    frames = report["profile"]["times"]
    assert len(frames) == conftest.CAPSULE_FRAMES - 1
    assert [f["heldout"] for f in frames] == [False, True, True, False, False]
    # Each names the camera frames it was measured between.
    assert [f["frames"] for f in frames] == [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]


def test_every_profile_covers_the_whole_front_with_the_model_s_own_speed(tmp_path):
    """The model side is defined everywhere on its own front -- including the
    nose cap, where only the MEASUREMENT is untrustworthy. Blanking the model
    there too would hide the very place the two estimators disagreed."""
    profile = _report(tmp_path)["profile"]

    for frame in profile["times"]:
        assert len(frame["model"]) == len(profile["s"])
        assert all(value is not None for value in frame["model"])


def test_the_profile_is_ordered_along_the_front_root_to_nose(tmp_path):
    """The analytic capsule again: its flanks have a fixed radius and do not
    move, while the front races ahead at the nose. So the measured profile must
    sit at zero over the body and climb only as the traversal approaches the
    nose cap.

    This is what catches a scrambled traversal. If ``s`` mapped the segments in
    the wrong order or ran a cap backwards, nose-speed values would land in
    body bins -- and every other assertion here would still pass.
    """
    import numpy as np

    profile = _report(tmp_path)["profile"]
    named = {s["name"]: s for s in profile["segments"]}
    measured = np.array(
        [np.nan if v is None else v for v in profile["times"][0]["measured"]], dtype=float
    )

    upper = named["upper_body"]
    span = upper["bin_end"] - upper["bin_start"]
    at_root = measured[upper["bin_start"] : upper["bin_start"] + span // 2]
    at_nose = measured[upper["bin_end"] - 2 : upper["bin_end"]]

    assert np.nanmax(np.abs(at_root)) < 1.0  # µm/ms: standing still
    assert np.nanmax(at_nose) > 2 * np.nanmax(np.abs(at_root))


def test_a_run_without_an_explicit_front_reports_no_profile(tmp_path):
    report = _report_without_front_geometry(tmp_path)
    assert report["profile"] is None


# --- The kymograph: the whole history of the front's motion as one image ------


def test_the_kymograph_covers_the_same_axes_the_other_charts_use(tmp_path):
    """One row per continuous instant, one column per profile bin -- the same
    ``s`` the profile chart plots along and the same time axis the speed charts
    run on, so a reader can move between all three."""
    report = _report(tmp_path)
    profile = report["profile"]
    kymograph = profile["kymograph"]

    assert kymograph["t_ms"] == report["nose_speed"]["t_ms"]
    assert len(kymograph["v_um_per_ms"]) == len(kymograph["t_ms"])
    assert all(len(row) == len(profile["s"]) for row in kymograph["v_um_per_ms"])
    assert all(value is not None for row in kymograph["v_um_per_ms"] for value in row)


def test_the_kymograph_is_denser_in_time_than_the_measurement_can_be(tmp_path):
    """The point of it: the parameterisation is continuous in time while the
    measurement exists only at frame pairs. A kymograph limited to the pair times
    would throw away exactly what the model knows and the camera does not."""
    profile = _report(tmp_path)["profile"]

    assert len(profile["kymograph"]["t_ms"]) > 4 * len(profile["times"])


def test_the_model_profile_survives_a_dataset_with_no_measurable_interval(tmp_path):
    """Every camera frame non-adjacent: nothing can be differenced, so there are
    no measured profiles -- but the model's own answer is still well defined and
    is still reported, rather than the whole block vanishing."""
    import numpy as np

    from naviernet.evaluation import evaluate
    from naviernet.training import load_model, train

    cfg, paths = conftest.staged_capsule_run(tmp_path, GEO)
    archive = dict(np.load(paths.tensors, allow_pickle=False))
    meta = json.loads(str(archive["meta"]))
    meta["frame_numbers"] = [1, 3, 5, 7, 9, 11]  # every other camera frame
    archive["meta"] = json.dumps(meta)
    np.savez_compressed(paths.tensors, **archive)

    train(cfg, paths)
    model, data, _ = load_model(cfg, paths)
    evaluate(cfg, model, data, paths)
    profile = json.loads(paths.front_velocity_json.read_text())["profile"]

    assert data.report_pairs == []
    assert profile["times"] == []
    assert len(profile["kymograph"]["v_um_per_ms"]) > 1


# --- Variants ----------------------------------------------------------------


def test_a_joint_run_reports_each_condition_separately(tmp_path):
    """Every dataset a joint run spans gets its own report, scoped by name --
    the same layout the trajectories use. The two conditions grow at different
    rates in the fixture, so identical files would mean one condition's answer
    was written for both."""
    from naviernet import training
    from naviernet.evaluation import evaluate_joint

    cfg, paths = conftest.staged_joint_run(tmp_path, GEO)
    training.train(cfg, paths)
    model, contexts, heldout = training.load_joint(cfg, paths)
    evaluate_joint(cfg, model, contexts, paths, heldout_datasets=heldout)

    reports = {}
    for name, _, _, _ in conftest.JOINT_SPECS:
        path = paths.front_velocity_json_for(name)
        assert path.is_file(), f"no front-velocity report for {name}"
        reports[name] = json.loads(path.read_text())

    speeds = [tuple(r["nose_speed"]["measured"]["v_um_per_ms"]) for r in reports.values()]
    assert speeds[0] != speeds[1], "both conditions reported the same measured speed"


def test_the_report_states_its_units_rather_than_leaving_them_to_the_reader(tmp_path):
    """A bare number is not a measurement. Every consumer -- chart axis, CSV
    column, tooltip -- reads the unit from the key, so the key carries it."""
    report = _report(tmp_path)

    assert "v_um_per_ms" in report["nose_speed"]
    assert "t_ms" in report["nose_speed"]
