"""The acceptance audit and the seed-ensemble comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from naviernet.audit import audit_grid, audit_run, mirror_breaking_fraction
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
]
VERDICTS = {"mirror", "residuals", "flux", "heat", "outline", "outline_drop", "nose"}


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
