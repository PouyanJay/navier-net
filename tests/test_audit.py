"""The acceptance audit and the seed-ensemble comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from naviernet.audit import (
    AuditGrid,
    audit_grid,
    audit_run,
    mirror_breaking_fraction,
    outlet_strip_record,
    scoring_summary,
    temperature_record,
)
from naviernet.data.dataset import BubbleDataset
from naviernet.physics import mass_balance
from naviernet.physics.groups import compute_groups
from naviernet.training import load_model, train

from .conftest import staged_capsule_run

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_ensemble import compare_runs  # noqa: E402

SHARP_THERMAL = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.thermal_model=sharp",
    "model.source_mode=thermal",
    "model.footprint_evaporation=true",
    "experiment.T_inlet_C=56.6",
    "experiment.T_init_C=58.0",
    "experiment.wet_fraction_mode=fit",
    "experiment.wet_fraction=0.8",
    "training.collocation_time_bins=2",
    "training.stage_b_warmup_steps=1",
    "evaluation.audit_nx=33",
    "evaluation.audit_ny=17",
    "evaluation.audit_scan_times=5",
]
VERDICTS = {
    "mirror",
    "residuals",
    "flux",
    "heat",
    "outline",
    "outline_drop",
    "nose",
    "residuals_in_time",
    "flux_in_time",
    "jump_shape",
    "jump_pressure",
    "liquid_gradient",
    "subcooling",
}


def _no_constants(token: str):
    raise AssertionError(f"audit.json carries {token}, which a browser cannot parse")


def _grid_fields():
    ys = np.linspace(-1.0, 1.0, 21)[:, None]
    xs = np.linspace(0.0, 2.0, 31)[None, :]
    return ys, xs


def test_mirror_breaking_fraction_separates_even_from_odd_fields():
    ys, xs = _grid_fields()
    even, odd = np.cos(ys) * xs, ys * xs

    assert mirror_breaking_fraction(even) == pytest.approx(0.0, abs=1e-12)
    assert mirror_breaking_fraction(odd) == pytest.approx(1.0)
    assert mirror_breaking_fraction(odd, odd=True) == pytest.approx(0.0, abs=1e-12)
    assert mirror_breaking_fraction(even + odd) == pytest.approx(
        np.sqrt(np.mean(odd**2) / np.mean((even + odd - (even + odd).mean()) ** 2))
    )


def test_audit_grid_is_symmetric_about_the_channel_centreline(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path)
    data = BubbleDataset(cfg, paths)

    grid = audit_grid(data, 33, 17)

    assert grid.y_center == pytest.approx(0.5 * (data.y_lo + data.y_hi))
    assert grid.ys[0] + grid.ys[-1] == pytest.approx(2 * grid.y_center)
    assert np.allclose(grid.ys + grid.ys[::-1], 2 * grid.y_center)


def test_mass_balance_plan_can_be_built_at_given_instants(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path)
    data = BubbleDataset(cfg, paths)

    plan = mass_balance.build_plan(
        data.domain,
        (data.y_lo, data.y_hi),
        compute_groups(cfg),
        cfg.training,
        "cpu",
        at_times=torch.tensor([0.25]),
    )

    assert plan.n_times == 1
    assert torch.all(plan.points[:, 2] == 0.25)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a second device")
def test_mass_balance_plan_accepts_instants_on_another_device(tmp_path):
    """The audit hands the plan a time tensor on the model's device; the plan
    must not mix it with its own CPU grids (the GPU audit crashed on that)."""
    cfg, paths = staged_capsule_run(tmp_path)
    data = BubbleDataset(cfg, paths)

    plan = mass_balance.build_plan(
        data.domain,
        (data.y_lo, data.y_hi),
        compute_groups(cfg),
        cfg.training,
        "mps",
        at_times=torch.tensor([0.25], device="mps"),
    )

    assert plan.points.device.type == "mps" and plan.n_times == 1


def test_audit_of_a_trained_run_reports_every_test_with_verdicts(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, SHARP_THERMAL)
    train(cfg, paths)
    model, data, _ = load_model(cfg, paths)

    report = audit_run(cfg, model, data, paths)

    assert paths.audit_json.exists() and paths.audit_fields_npz.exists()
    json.loads(paths.audit_json.read_text(), parse_constant=_no_constants)
    assert set(report["verdicts"]) == VERDICTS
    assert all(isinstance(v["pass"], bool) for v in report["verdicts"].values())
    assert set(report["warmup_iou"]) == {str(f) for f in data.frame_numbers[: data.n_event]}
    outline = report["verdicts"]["outline"]
    assert outline["pass"] == (outline["worst"] >= outline["min"])
    assert report["verdicts"]["nose"]["max"] == cfg.evaluation.acceptance.nose_tol
    frame = report["frames"][1]
    assert set(frame["mirror"]) == {"alpha", "u", "v", "p", "T", "s"}
    assert {"brinkman", "energy_liquid"} & set(frame["residual_rms"]) or "darcy" in frame[
        "residual_rms"
    ]
    assert frame["flux"]["max_abs"] >= 0 and len(frame["flux"]["defect"]) == len(
        frame["flux"]["stations"]
    )
    assert {
        "latent_from_source_mW",
        "latent_for_growth_mW",
        "latent_from_flux_mW",
        "latent_from_footprint_mW",
        "wall_into_liquid_mW",
        "wall_under_footprint_mW",
        "wet_fraction",
        "spread",
    } <= set(frame["heat"])
    assert frame["heat"]["latent_from_footprint_mW"] > 0, (
        "the film under the capsule evaporates"
    )
    assert frame["front"] is not None and frame["front"]["kinematic_rms_upper"] >= 0
    assert set(frame["front"]["jump_px_by_segment"]) <= {"upper", "lower", "root", "nose"}
    assert frame["front"]["jump_px_p90"] >= frame["front"]["jump_px_median"] >= 0
    scan = report["scan"]
    assert len(scan["t_star"]) == 5 == len(scan["flux_max_abs"]) == len(scan["laplace_rms"])
    assert all(len(v) == 5 for v in scan["residual_rms"].values())
    assert set(frame["outlet_strip"]) == set(frame["residual_rms"])
    assert all(0.0 <= v["share_of_error"] <= 1.0 for v in frame["outlet_strip"].values())
    assert {"rms", "mean", "rms_without_root", "root_mean", "root_rms"} == set(
        frame["front"]["laplace_by_length_Pa"]
    )
    assert set(report["scoring"]) == {"all_frames", "without_last_frame"}
    assert (
        report["scoring"]["all_frames"]["flux"]
        >= report["scoring"]["without_last_frame"]["flux"]
    )
    assert 0.0 <= frame["outline"]["iou"] <= 1.0
    assert frame["outline"]["nose_error"] >= 0.0
    assert all(np.isfinite(v) for v in frame["mirror"].values())
    snapshots = np.load(paths.audit_fields_npz)
    assert f"{frame['frame']}/u" in snapshots.files


def test_evaluate_writes_the_audit_verdicts_into_metrics(tmp_path):
    from naviernet.evaluation import evaluate

    cfg, paths = staged_capsule_run(tmp_path, SHARP_THERMAL)
    train(cfg, paths)
    model, data, _ = load_model(cfg, paths)

    metrics = evaluate(cfg, model, data, paths)

    assert set(metrics["audit"]) == VERDICTS
    assert metrics["heating"]["wet_fraction_mode"] == "fit"
    assert 0.0 < metrics["heating"]["wet_fraction"] < 1.0
    assert json.loads(paths.metrics_json.read_text())["audit"] == metrics["audit"]


def test_a_missing_nose_counts_as_the_whole_channel_off():
    from naviernet.audit import _outline_verdicts

    class Acceptance:
        iou_min, iou_drop_max, nose_tol = 0.85, 0.05, 0.1

    frames = [
        {"frame": 1, "outline": {"iou": 0.9, "nose_error": 0.02}},
        {"frame": 2, "outline": {"iou": 0.9, "nose_error": None}},
    ]

    verdicts = _outline_verdicts(frames, Acceptance(), None, nose_missing=5.75)

    assert verdicts["nose"] == {"worst": 5.75, "max": 0.1, "pass": False}
    assert "outline_drop" not in verdicts, "no warm-up record, no drop verdict"


def test_the_jump_mismatch_reads_as_the_shape_change_that_removes_it():
    """A mismatch equal to what a one-pixel displacement of a feature of radius R
    changes in the jump reads as one pixel; a flat stretch is held to the
    channel's half-width as its radius."""
    from naviernet.audit import jump_shape_pixels
    from naviernet.physics.residuals import IN_PLANE_WEIGHT

    groups, pixel, width = {"We": 2.0}, 0.01, 1.0
    kappa = np.array([5.0, 0.0])
    one_pixel = IN_PLANE_WEIGHT * np.array([5.0, 2.0]) ** 2 * pixel / groups["We"]

    pixels = jump_shape_pixels(one_pixel * np.array([1.0, 3.0]), kappa, groups, pixel, width)

    assert pixels == pytest.approx([1.0, 3.0])


def test_the_jump_pressure_verdict_holds_the_worst_frame_to_the_pascal_limit():
    from naviernet.audit import _jump_verdict

    class Acceptance:
        jump_px_max, jump_pa_max = 2.0, 5.0

    frames = [
        {"front": {"jump_px_p90": 1.0, "laplace_rms_Pa": 3.0}},
        {"front": {"jump_px_p90": 1.5, "laplace_rms_Pa": 7.0}},
        {"front": None},
    ]

    verdicts = _jump_verdict(frames, Acceptance())

    assert verdicts["jump_pressure"] == {"worst": 7.0, "max": 5.0, "pass": False}
    assert verdicts["jump_shape"] == {"worst": 1.5, "max": 2.0, "pass": True}


def test_seed_ensemble_comparison_scores_agreement_and_disagreement():
    ys, xs = _grid_fields()
    field = np.cos(ys) * xs + 0.3 * ys * xs
    run_a = {"1/u": field, "1/v": ys * xs, "xs": xs.ravel(), "ys": ys.ravel()}
    run_b = {"1/u": field.copy(), "1/v": (ys * xs).copy()}
    mirrored = {"1/u": field[::-1, :], "1/v": -(ys * xs)}

    agree = compare_runs({"a": run_a, "b": run_b})
    disagree = compare_runs({"a": run_a, "m": mirrored})

    assert agree["fields"]["u"]["min_full_corr"] == pytest.approx(1.0)
    assert agree["fields"]["u"]["frames"]["1"]["a|b"]["breaking"] == pytest.approx(1.0)
    assert disagree["fields"]["u"]["frames"]["1"]["a|m"]["breaking"] == pytest.approx(-1.0)
    assert disagree["fields"]["v"]["frames"]["1"]["a|m"]["full"] == pytest.approx(-1.0)


def _open_channel_grid():
    """A grid of 15 um cells with a bubble in the middle, the liquid on both sides."""
    xs = np.linspace(0.0, 3.0, 61)
    ys = np.linspace(-0.5, 0.5, 21)
    alpha = np.zeros((len(ys), len(xs)))
    alpha[8:13, 20:40] = 1.0
    return AuditGrid(xs=xs, ys=ys, y_center=0.0), alpha


def test_a_smooth_warm_liquid_passes_the_temperature_checks():
    grid, alpha = _open_channel_grid()
    theta = np.tile(0.1 * grid.xs, (len(grid.ys), 1))  # 0.1 K per L_ref: 0.33 K/mm

    record = temperature_record(
        {"alpha": alpha, "T": theta}, grid, {"dT_ref": 1.0, "theta_inlet": 0.0}, 300.0
    )

    assert record["gradient_K_mm"] == pytest.approx(0.1 / 0.3, rel=1e-6)
    assert record["subcooling_K"] == 0.0


def test_a_temperature_step_in_the_open_liquid_is_measured_where_it_sits():
    grid, alpha = _open_channel_grid()
    theta = np.where(grid.xs[None, :] < 2.5, 0.3, 0.0) * np.ones((len(grid.ys), 1))

    record = temperature_record(
        {"alpha": alpha, "T": theta}, grid, {"dT_ref": 2.0, "theta_inlet": 0.0}, 300.0
    )

    assert record["gradient_K_mm"] == pytest.approx(0.6 / (0.05 * 0.3), rel=1e-6)
    assert record["gradient_at_um"][0] == pytest.approx(2.475 * 300.0, rel=1e-6)


def test_the_thermal_layer_at_the_front_is_not_a_step():
    grid, alpha = _open_channel_grid()
    # The superheat drops to saturation over the two cells around the bubble.
    theta = np.full((len(grid.ys), len(grid.xs)), 0.5)
    theta[6:15, 18:42] = 0.0

    record = temperature_record(
        {"alpha": alpha, "T": theta}, grid, {"dT_ref": 1.0, "theta_inlet": 0.0}, 300.0
    )

    assert record["gradient_K_mm"] == 0.0


def test_liquid_below_the_inlet_temperature_is_subcooling():
    grid, alpha = _open_channel_grid()
    theta = np.full((len(grid.ys), len(grid.xs)), 0.2)
    theta[0, 50] = -0.04  # one cold cell in the liquid
    theta[10, 30] = -1.0  # inside the bubble: not liquid, not counted

    record = temperature_record(
        {"alpha": alpha, "T": theta}, grid, {"dT_ref": 1.0, "theta_inlet": 0.1}, 300.0
    )

    assert record["subcooling_K"] == pytest.approx(0.14)


def test_a_model_without_a_temperature_has_no_temperature_record():
    grid, alpha = _open_channel_grid()

    assert (
        temperature_record({"alpha": alpha}, grid, {"dT_ref": 1.0, "theta_inlet": 0.0}, 300.0)
        == {}
    )


def test_the_outlet_strip_separates_the_last_columns_from_the_rest():
    grid = AuditGrid(xs=np.linspace(0.0, 4.0, 5), ys=np.array([0.0, 1.0]), y_center=0.5)
    squared = {"brinkman": np.array([1.0, 1.0, 1.0, 1.0, 16.0] * 2)}

    record = outlet_strip_record(squared, grid, strip_star=0.5)["brinkman"]

    assert record["rms_strip"] == pytest.approx(4.0)
    assert record["rms_rest"] == pytest.approx(1.0)
    assert record["share_of_error"] == pytest.approx(0.8)


def _scored_frame(flux: float) -> dict:
    return {
        "residual_rms": {"brinkman": 2.0 * flux},
        "flux": {"max_abs": flux},
        "heat": {"spread": 0.1},
    }


def test_the_scoring_summary_reports_the_run_with_and_without_its_last_frame():
    summary = scoring_summary([_scored_frame(0.05), _scored_frame(0.07), _scored_frame(0.5)])

    assert summary["all_frames"]["flux"] == 0.5
    assert summary["without_last_frame"]["flux"] == 0.07
    assert summary["without_last_frame"]["residual"] == pytest.approx(0.14)


def test_a_single_frame_has_nothing_to_leave_out():
    assert set(scoring_summary([_scored_frame(0.05)])) == {"all_frames"}
