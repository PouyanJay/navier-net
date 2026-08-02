"""Physics diagnostics: the measurements IoU could not make.

The R3 run scored IoU 0.929 / 0.866 while its force balance was ~55% violated,
its Young-Laplace jump was 20x off with the wrong sign, and its axial capillary
gradient was identically zero -- so no neck could ever form. An overlap metric
cannot see any of that. These are the numbers that can.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import staged_run as _staged_run

TINY_SHARP = ["model=stage_b", "model.front_geometry=true", "model.sharp_interface=true"]


def _model(tmp_path, overrides=None):
    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths

    cfg, paths = _staged_run(tmp_path, overrides or TINY_SHARP)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    return BubblePINN(cfg, geometry=_geometry_priors(cfg, data)), data, cfg


# --- neck geometry -----------------------------------------------------------


def test_neck_depth_is_zero_for_a_monotone_profile():
    """The R3 failure mode reads exactly zero: a profile that only widens has no
    neck, however steep its ramp."""
    from naviernet.physics.diagnostics import neck_of_profile

    profile = np.array([0.198, 0.198, 0.199, 0.200, 0.203, 0.213, 0.249, 0.316, 0.365])
    neck = neck_of_profile(profile)
    assert neck.depth == pytest.approx(0.0)


def test_neck_depth_and_location_match_the_measured_bubble():
    """The frame-11 truth: a shoulder at 0.213, a neck at 0.112, a head at 0.392.
    Depth is the fractional collapse below the SHALLOWER shoulder, so a deep neck
    beside a huge head is not flattered by the head."""
    from naviernet.physics.diagnostics import neck_of_profile

    profile = np.array([0.207, 0.213, 0.177, 0.136, 0.112, 0.131, 0.191, 0.275, 0.392])
    neck = neck_of_profile(profile)
    assert neck.depth == pytest.approx(1.0 - 0.112 / 0.213, rel=1e-6)
    assert neck.location == pytest.approx(0.5, abs=1e-6)


def test_neck_ignores_a_minimum_with_nothing_above_it_on_one_side():
    """A monotone profile's minimum sits at an end; that is a taper, not a neck,
    and must not be reported as one."""
    from naviernet.physics.diagnostics import neck_of_profile

    assert neck_of_profile(np.array([0.10, 0.20, 0.30, 0.40])).depth == pytest.approx(0.0)


def test_measured_half_width_profile_reads_the_masks(tmp_path):
    """The truth side of the comparison comes from the segmented masks, sampled
    on the same normalised station grid as the model's own front."""
    from naviernet.physics.diagnostics import measured_half_width_profile

    _, data, _ = _model(tmp_path)
    profile = measured_half_width_profile(data, row=0, n_stations=9)
    assert profile.shape == (9,)
    assert (profile > 0).all(), "a training frame with vapour has positive half-width"


# --- interface conditions ----------------------------------------------------


def test_an_unsatisfied_jump_reads_a_large_error(tmp_path):
    """The negative control for `test_a_satisfied_jump_reads_near_zero_error`.
    An untrained pressure field does not satisfy the condition, and the
    diagnostic has to SAY so -- a metric that always returned a small number
    would pass the positive test alone and hide exactly the failure it exists to
    catch."""
    from naviernet.physics.diagnostics import interface_diagnostics

    model, data, _ = _model(tmp_path)
    diag = interface_diagnostics(model, data)

    assert np.isfinite(diag.laplace_error_nose)
    assert np.isfinite(diag.laplace_error_front)
    assert diag.laplace_error_nose > 0.3, (
        f"an untrained model must read far above the 0.10 gate, got "
        f"{diag.laplace_error_nose:.3f}"
    )


@pytest.mark.slow  # fits a real pressure net for 1500 Adam steps (~14 s)
def test_a_satisfied_jump_reads_near_zero_error(tmp_path):
    """Construct the solution the condition asks for -- p_liq = p_v - kappa/We --
    and the diagnostic must agree it is satisfied. Without this the metric could
    report anything and we would not know."""
    from naviernet.physics.diagnostics import interface_diagnostics

    torch.manual_seed(0)
    # A real pressure architecture, not the 8-wide stub the other tests run: this
    # test asks whether the CONDITION can be satisfied, so it must not be
    # answering a question about capacity instead.
    model, data, _ = _model(
        tmp_path,
        [*TINY_SHARP, "model.hidden=64", "model.layers=3", "model.fourier_feats=32"],
    )
    _fit_pressure_to_the_jump(model, data, steps=1500)

    diag = interface_diagnostics(model, data)
    assert diag.laplace_error_nose < 0.10, (
        f"a fitted pressure must satisfy the jump at the nose, got "
        f"{diag.laplace_error_nose:.3f}"
    )
    assert diag.laplace_error_front < 0.20, (
        f"and across the whole front, got {diag.laplace_error_front:.3f}"
    )


def test_front_error_stays_finite_where_the_body_curvature_vanishes(tmp_path):
    """A straight body has kappa_par -> 0, so a POINTWISE relative error there is
    a division by nothing (it read 1.8e3 on an untrained model). The front-wide
    figure normalises by one global capillary scale instead, and must stay a
    sane, finite number."""
    from naviernet.physics.diagnostics import interface_diagnostics

    model, data, _ = _model(tmp_path)
    for net in (model.nets["phi"].width_net, model.nets["phi"].center_net):
        torch.nn.init.zeros_(net[-1].weight)

    diag = interface_diagnostics(model, data)
    assert np.isfinite(diag.laplace_error_front)
    assert diag.laplace_error_front < 1e3


def test_axial_capillary_gradient_vanishes_for_a_straight_sided_capsule(tmp_path):
    """The R3 diagnosis in one number: a straight body has no curvature variation
    along the bubble, hence no capillary pressure gradient, hence nothing that
    could ever drain a neck."""
    from naviernet.physics.diagnostics import interface_diagnostics

    model, data, _ = _model(tmp_path)
    for net in (model.nets["phi"].width_net, model.nets["phi"].center_net):
        torch.nn.init.zeros_(net[-1].weight)

    diag = interface_diagnostics(model, data)
    assert diag.axial_capillary_gradient == pytest.approx(0.0, abs=1e-4)


def test_a_diffuse_run_is_measured_by_the_same_definition(tmp_path):
    """The R3 baseline has no p_v(t) -- but its jump error is exactly the number
    the before/after turns on. It is estimated from the near-isobaric interior
    the physics already justifies, so both runs are scored one way."""
    from naviernet.physics.diagnostics import interface_diagnostics

    model, data, _ = _model(tmp_path, ["model=stage_b", "model.front_geometry=true"])
    assert model.sharp_interface is False

    diag = interface_diagnostics(model, data)
    assert np.isfinite(diag.laplace_error_nose), (
        "a diffuse front-geometry run must still be measurable, or the gate has "
        "no baseline to beat"
    )


def _fit_pressure_to_the_jump(model, data, steps: int):
    """Train ONLY the pressure fields against the Young-Laplace condition, so the
    diagnostic is checked against a solution that really does satisfy it."""
    from naviernet.physics import diagnostics as diag
    from naviernet.physics.groups import compute_groups
    from naviernet.physics.residuals import laplace_jump_residual

    groups = compute_groups(model.cfg)
    # The diagnostic's own times, not a coarser grid: a model fitted on 8 instants
    # and scored on 16 is not a model that satisfies the condition, and the test
    # would be measuring coverage rather than the metric.
    times = diag.diagnostic_times(data)
    params = list(model.nets["p"].parameters()) + list(model.vapor_pressure.parameters())
    opt = torch.optim.Adam(params, lr=5e-3)
    for _ in range(steps):
        opt.zero_grad()
        front = model.nets["phi"].front(
            times, n_body=diag.DIAGNOSTIC_BODY_SAMPLES, n_cap=diag.DIAGNOSTIC_CAP_SAMPLES
        )
        (laplace_jump_residual(model, front, groups) ** 2).mean().backward()
        opt.step()


# --- the physics block that travels in metrics.json ---------------------------


def test_metrics_json_carries_the_physics_block(tmp_path):
    """A front-geometry run's report must carry the physics, per frame and in
    aggregate -- IoU alone is what let a ~55% violated force balance look fine."""
    import json

    from naviernet.evaluation import evaluate
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, TINY_SHARP)
    train(cfg, paths)
    model, data, _ = load_model(cfg, paths)
    report = evaluate(cfg, model, data, paths)

    physics = report["physics"]
    assert set(physics) >= {
        "laplace_error_nose",
        "laplace_error_front",
        "axial_capillary_gradient",
        "neck_depth_model",
        "neck_depth_measured",
        "residual_convergence",
        "per_frame",
    }
    assert len(physics["per_frame"]) == len(data.frame_numbers)
    row = physics["per_frame"][0]
    assert len(row["half_width_model"]) == len(row["half_width_measured"])
    # and it survives the JSON round-trip the API will read it through
    assert json.loads(paths.metrics_json.read_text())["physics"]["per_frame"]


def test_a_run_without_an_explicit_front_reports_no_physics_block(tmp_path):
    """Nothing to measure the interface conditions on, so the key is present and
    null rather than fabricated."""
    from naviernet.evaluation import evaluate
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, ["model=stage_b"])
    train(cfg, paths)
    model, data, _ = load_model(cfg, paths)
    assert evaluate(cfg, model, data, paths)["physics"] is None


def test_residual_convergence_separates_a_descending_term_from_a_stuck_one():
    """The R3 baseline's momentum term sat at ~4.7 -- neither obviously large nor
    small. What it never did was fall. The ratio says which."""
    from naviernet.physics.diagnostics import residual_convergence

    history = [{"step": s, "darcy": 10.0 / s, "mom": 5.0} for s in range(1, 101)]
    report = residual_convergence(history, ("darcy", "mom"), after_step=0)

    assert report["darcy"]["ratio"] < 0.1, "a converging term"
    assert report["mom"]["ratio"] == pytest.approx(1.0), "a stuck one"


def test_residual_convergence_ignores_the_warmup_window():
    """Stage-B terms are logged during the warm-up but held at zero weight;
    averaging across the gate would mix two different objectives."""
    from naviernet.physics.diagnostics import residual_convergence

    history = [{"step": s, "darcy": 100.0} for s in range(1, 51)]
    history += [{"step": s, "darcy": 10.0 / (s - 50)} for s in range(51, 101)]
    report = residual_convergence(history, ("darcy",), after_step=50)
    assert report["darcy"]["first"] < 100.0, "the warm-up plateau must not be counted"


def test_a_stage_a_run_still_reports_the_shape_diagnostics(tmp_path):
    """No pressure field means no jump to score -- but the neck and the drainage
    drive are pure geometry, so a Stage-A front-geometry run is not left with an
    empty physics block (nor with a crash)."""
    from naviernet.physics.diagnostics import physics_report

    model, data, _ = _model(tmp_path, ["model.front_geometry=true"])
    assert "p" not in model.fields

    block = physics_report(model, data)
    assert np.isnan(block["laplace_error_nose"]), "an unmeasurable jump reads nan"
    assert np.isfinite(block["neck_depth_measured"])
    assert np.isfinite(block["axial_capillary_gradient"])


def test_convergence_reads_the_warmup_the_run_used_not_the_one_recomposed(tmp_path):
    """A standalone `stage=evaluate` composes its own config, which need not carry
    the warm-up override the run was launched with. Reading the wrong one averages
    a term's convergence across the window where it was held at zero weight -- and
    reported a 15x RISE for a residual that was in fact descending."""
    from naviernet.evaluation import evaluate
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(
        tmp_path, [*TINY_SHARP, "training.steps=4", "training.stage_b_warmup_steps=2"]
    )
    train(cfg, paths)

    ckpt = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    assert ckpt["state"]["stage_b_warmup_steps"] == 2, "the run must record its own warm-up"

    # Evaluate through a config that forgot the override, as a standalone stage does.
    forgetful, _ = _staged_run(tmp_path, [*TINY_SHARP, "training.stage_b_warmup_steps=0"])
    model, data, _ = load_model(forgetful, paths)
    block = evaluate(forgetful, model, data, paths)["physics"]
    assert block["residual_convergence"] is not None
