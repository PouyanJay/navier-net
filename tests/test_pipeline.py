"""Trainer mechanics and end-to-end behaviour against the real dataset."""

from __future__ import annotations

import pytest
import torch

from naviernet.data.preprocess import MIN_USABLE_FRAMES, usable_frame_numbers
from naviernet.pipeline import Pipeline
from naviernet.training import REBALANCED_TERMS, _rebalance
from naviernet.utils.paths import RunPaths

from .conftest import make_config


def test_rebalance_equalises_gradient_contributions():
    """A term with a small gradient gets a larger weight, and vice versa."""
    weights = {"data": 10.0, "vof": 1.0, "div": 1.0, "src": 0.1, "bc": 1.0}
    before = dict(weights)
    # `data` contributes 10 * 1 = 10; vof's gradient is tiny, div's is huge.
    _rebalance(weights, {"data": 1.0, "vof": 0.01, "div": 100.0, "bc": 1.0})

    assert weights["vof"] > before["vof"]
    assert weights["div"] < before["div"]
    assert weights["src"] == before["src"], "src is a fixed penalty, not rebalanced"
    assert weights["data"] == before["data"], "data is the reference scale"


def test_rebalance_stays_within_bounds():
    weights = dict.fromkeys(("data", "vof", "div", "src", "bc"), 1.0)
    for _ in range(50):
        _rebalance(weights, {"data": 1.0, "vof": 1e-30, "div": 1e30, "bc": 1.0})

    for term in REBALANCED_TERMS:
        assert 1e-2 <= weights[term] <= 1e3


def test_stage_a_training_is_byte_for_byte_unchanged(tmp_path):
    """Guards the equation-registry refactor.

    The Stage-A loss trajectory and rebalanced weights must match the golden
    captured from the pre-registry trainer. If deriving the active terms from
    the registry ever changes a number, this fails loudly.
    """
    import json
    from pathlib import Path

    from naviernet.training import train

    golden = json.loads(
        (Path(__file__).parent / "fixtures" / "stage_a_golden.json").read_text()
    )
    cfg = make_config([f"paths.root={tmp_path}", *golden["overrides"]])
    paths = RunPaths.from_config(cfg)
    paths.ensure()
    paths.tensors.parent.mkdir(parents=True, exist_ok=True)
    _write_tensors(paths.tensors, list(range(1, 9)), n_event=8)

    _, _, state = train(cfg, paths)

    assert len(state["hist"]) == len(golden["hist"])
    for got, want in zip(state["hist"], golden["hist"], strict=True):
        assert got.keys() == want.keys()
        for key, wanted in want.items():
            assert got[key] == pytest.approx(wanted, rel=1e-6, abs=1e-8), f"{key} drifted"
    for key, wanted in golden["w"].items():
        assert state["w"][key] == pytest.approx(wanted, rel=1e-6, abs=1e-8), (
            f"weight {key} drifted"
        )


def test_curriculum_ramps_evaporation_in_as_the_source_penalty_decays():
    """Soft -> hard: at the start the penalty is full and the closure off; by the
    end the closure is full and the penalty gone. Disabled (0) leaves both at 1."""
    from naviernet.training import _curriculum

    assert _curriculum(0, 100) == (1.0, 0.0)
    assert _curriculum(50, 100) == (0.5, 0.5)
    assert _curriculum(100, 100) == (0.0, 1.0)
    assert _curriculum(150, 100) == (0.0, 1.0), "clamped past the end"
    assert _curriculum(50, 0) == (1.0, 1.0), "0 disables the schedule"


def test_curriculum_schedule_targets_real_weight_keys():
    """The trainer applies the schedule by the names 'src' and 'evap'; guard those
    strings against drifting from the registry's weight keys (else it silently
    no-ops via schedule.get(name, 1.0))."""
    from naviernet.physics.registry import REGISTRY

    keys = {e.weight_key for e in REGISTRY}
    assert {"src", "evap"} <= keys


_TINY_TRAIN = [
    "model.hidden=8",
    "model.layers=2",
    "model.fourier_feats=4",
    "training.n_data=16",
    "training.n_coll=16",
    "training.n_bc=8",
]
_TINY_STAGE_B = [
    "model=stage_b",
    "training=stage_b",
    "model.per_field.p.hidden=8",
    "model.per_field.p.layers=2",
    "model.per_field.T.hidden=8",
    "model.per_field.T.layers=2",
]


def _stage(paths):
    paths.ensure()
    paths.tensors.parent.mkdir(parents=True, exist_ok=True)
    _write_tensors(paths.tensors, list(range(1, 9)), n_event=8)


def test_stage_b_smoke_run_trains_pressure_and_temperature(tmp_path):
    """A tiny Stage-B run builds the p/T networks, activates momentum + energy +
    evaporation, and takes real steps without producing a NaN."""
    import math

    from naviernet.training import train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            *_TINY_STAGE_B,
            *_TINY_TRAIN,
            "training.steps=2",
            "training.rebalance_every=2",
            "training.curriculum_steps=2",
        ]
    )
    paths = RunPaths.from_config(cfg)
    _stage(paths)

    model, _, state = train(cfg, paths)

    assert {"p", "T"} <= set(model.fields)
    assert {"mom", "energy", "evap"} <= set(state["hist"][-1])
    for record in state["hist"]:
        for name, value in record.items():
            assert math.isfinite(value), f"{name} was not finite"
    assert paths.checkpoint.exists()


def test_stage_b_warm_starts_from_a_stage_a_checkpoint(tmp_path, caplog):
    """A Stage-A checkpoint seeds a Stage-B run: phi/u/v/s load, p/T and the
    evaporation unknowns initialise fresh, and the step count carries forward."""
    import logging

    from naviernet.training import train

    cfg_a = make_config([f"paths.root={tmp_path}", "training.steps=2", *_TINY_TRAIN])
    paths = RunPaths.from_config(cfg_a)
    _stage(paths)
    _, _, state_a = train(cfg_a, paths)
    assert state_a["done"] == 2 and "p" not in state_a["w"]

    cfg_b = make_config(
        [
            f"paths.root={tmp_path}",
            *_TINY_STAGE_B,
            *_TINY_TRAIN,
            "training.steps=2",
            "training.curriculum_steps=2",
        ]
    )
    with caplog.at_level(logging.INFO, logger="naviernet.training"):
        model, _, state_b = train(cfg_b, RunPaths.from_config(cfg_b))

    # The warm-start branch ran: it loaded phi/u/v/s (strict=False with only the
    # new p/T/evaporation params missing) rather than starting from scratch. The
    # unexpected-key guard would have raised had the checkpoint not matched.
    assert "warm start" in caplog.text
    assert {"p", "T"} <= set(model.fields)
    assert state_b["done"] == 4, "warm start carries the Stage-A step count forward"
    assert {"mom", "energy", "evap"} <= set(state_b["w"])
    # The data loss at the first Stage-B step continues from where Stage A left
    # off (phi loaded), not a fresh-random-net value.
    assert state_b["hist"][0]["data"] == pytest.approx(state_a["hist"][-1]["data"], rel=0.5)


def test_unknown_stage_is_rejected(tiny_cfg):
    with pytest.raises(ValueError, match="unknown stage"):
        Pipeline(tiny_cfg).run("trian")


def test_missing_tensors_give_an_actionable_error(tiny_cfg):
    """Asking to evaluate before preprocessing should say what to run."""
    with pytest.raises(FileNotFoundError, match="stage=train"):
        Pipeline(tiny_cfg).evaluate()


def test_pipeline_creates_the_run_directories(tiny_cfg):
    paths = RunPaths.from_config(tiny_cfg)
    Pipeline(tiny_cfg)
    assert paths.output_dir.is_dir() and paths.figures_dir.is_dir()


# --- frame exclusion --------------------------------------------------------


def test_usable_frames_are_the_window_when_nothing_is_excluded(cfg):
    assert usable_frame_numbers(cfg) == list(range(1, cfg.experiment.n_frames_usable + 1))


def test_excluded_frames_are_dropped_from_the_usable_window():
    cfg = make_config(["experiment.excluded_frames=[3,7]"])

    kept = usable_frame_numbers(cfg)

    assert 3 not in kept and 7 not in kept
    assert kept == [n for n in range(1, cfg.experiment.n_frames_usable + 1) if n not in (3, 7)]


def test_excluding_frames_outside_the_window_changes_nothing(cfg):
    """Frames past `n_frames_usable` were never fed to the model to begin with."""
    beyond = cfg.experiment.n_frames_usable + 1
    excluded = make_config([f"experiment.excluded_frames=[{beyond}]"])

    assert usable_frame_numbers(excluded) == usable_frame_numbers(cfg)


def test_excluding_almost_everything_is_rejected(cfg):
    survivors = list(range(1, MIN_USABLE_FRAMES))  # one short of the floor
    excluded = [n for n in range(1, cfg.experiment.n_frames_usable + 1) if n not in survivors]
    too_few = make_config([f"experiment.excluded_frames=[{','.join(map(str, excluded))}]"])

    with pytest.raises(ValueError, match="at least"):
        usable_frame_numbers(too_few)


def _write_tensors(path, frame_numbers: list[int], n_event: int) -> None:
    """A minimal archive: only the keys BubbleDataset reads on construction."""
    import json

    import numpy as np

    n_t = len(frame_numbers)
    alpha = np.zeros((n_t, 4, 5), dtype=np.float32)
    alpha[:, 1:3, 1:3] = 1.0
    np.savez_compressed(
        path,
        alpha=alpha,
        sdf=((0.5 - alpha) * 0.1).astype(np.float32),
        valid=np.ones_like(alpha),
        masks_camera=(alpha > 0.5).astype(np.uint8),
        x_star=np.linspace(0, 1, 5, dtype=np.float32),
        y_star=np.linspace(0, 1, 4, dtype=np.float32),
        t_star=((np.asarray(frame_numbers) - 1) * 0.1).astype(np.float32),
        meta=json.dumps(
            {
                "x_pin_star": 0.5,
                "t_ref_ms": 1.5,
                "n_frames_usable": n_t,
                "n_frames_event": n_event,
                "frame_numbers": frame_numbers,
            }
        ),
    )


def test_holdout_resolves_to_a_row_when_earlier_frames_are_excluded(tiny_cfg):
    """Camera frame 6 sits at row 4 once frame 3 is gone — supervising row 5
    would train on the holdout while still reporting it as held out."""
    from naviernet.data.dataset import BubbleDataset

    paths = RunPaths.from_config(tiny_cfg)
    paths.ensure()
    paths.tensors.parent.mkdir(parents=True, exist_ok=True)
    kept = [1, 2, 4, 5, 6, 7, 8]
    _write_tensors(paths.tensors, kept, n_event=len(kept))

    data = BubbleDataset(tiny_cfg, paths)  # tiny_cfg holds the default holdout

    assert tiny_cfg.training.holdout_frame == 5, "fixture assumes camera frame 6"
    assert data.frame_numbers == kept
    assert data.holdout_row == kept.index(6) == 4
    assert data.holdout_row not in data._ti[data._train_idx], "the holdout row is supervised"


def test_an_excluded_holdout_leaves_no_holdout_row(tiny_cfg):
    from naviernet.data.dataset import BubbleDataset

    paths = RunPaths.from_config(tiny_cfg)
    paths.ensure()
    paths.tensors.parent.mkdir(parents=True, exist_ok=True)
    _write_tensors(paths.tensors, [1, 2, 3, 4, 5, 7, 8], n_event=7)

    data = BubbleDataset(tiny_cfg, paths)

    assert data.holdout_row == -1
    assert len(data._train_idx) > 0, "every row is trainable when none is held out"


def test_event_frames_are_camera_numbers_not_row_indices(tiny_cfg):
    from naviernet.data.dataset import BubbleDataset

    paths = RunPaths.from_config(tiny_cfg)
    paths.ensure()
    paths.tensors.parent.mkdir(parents=True, exist_ok=True)
    _write_tensors(paths.tensors, [1, 2, 4, 5, 6, 7, 11], n_event=6)

    data = BubbleDataset(tiny_cfg, paths)

    assert data.n_event == 6
    assert data.event_frames == [1, 2, 4, 5, 6, 7]


# --- tests below need the real dataset -------------------------------------


@pytest.mark.needs_data
def test_dataset_shapes_agree(trained):
    _, data = trained
    n_frames, height, width = data.shape

    assert data.sdf.shape == data.alpha.shape == data.valid.shape
    assert len(data.x) == width and len(data.y) == height and len(data.t) == n_frames


@pytest.mark.needs_data
def test_holdout_frame_is_never_sampled(trained, cfg):
    """The generalisation claim rests on this: no supervision from that frame."""
    import numpy as np

    _, data = trained
    rng = np.random.default_rng(0)
    _, _ = data.sample_supervised(4096, rng)

    # The row is resolved from the archive, not assumed to be the config index:
    # excluding an earlier frame shifts it, and supervising the shifted row
    # would train on the very frame the headline IoU calls unseen.
    assert data.frame_numbers[data.holdout_row] == cfg.training.holdout_frame + 1
    assert data.holdout_row not in data._ti[data._train_idx]


@pytest.mark.needs_data
def test_signed_distance_is_negative_inside_the_bubble(trained):
    _, data = trained
    inside = data.alpha > 0.5

    assert data.sdf[inside].max() <= 0.0
    assert data.sdf[~inside].min() >= 0.0


@pytest.mark.needs_data
def test_sampled_points_lie_inside_the_domain(trained):
    import numpy as np

    _, data = trained
    d = data.domain
    points = data.sample_supervised(512, np.random.default_rng(1))[0]

    assert points[:, 0].min() >= d.x_min and points[:, 0].max() <= d.x_max
    assert points[:, 1].min() >= d.y_min and points[:, 1].max() <= d.y_max


@pytest.mark.needs_data
@pytest.mark.slow
def test_holdout_iou_meets_the_published_figure(trained, cfg):
    """The headline result: >0.95 IoU on a frame the model never saw."""
    from naviernet.evaluation import frame_iou

    model, data = trained
    iou = frame_iou(cfg, model, data, data.holdout_row)
    assert iou > 0.95


@pytest.mark.needs_data
@pytest.mark.slow
def test_inferred_nose_speed_matches_the_measurement(trained, cfg):
    """Inferred within 10% of the measured 180 mm/s, with no velocity supervision."""
    from naviernet.evaluation import nose_trajectory

    model, data = trained
    times, nose, _ = nose_trajectory(cfg, model, data)
    speed = torch.tensor(nose).diff() / torch.tensor(times).diff()
    middle = slice(len(times) // 5, 4 * len(times) // 5)
    mm_s = float(speed[middle].mean()) * cfg.scales.U_ref * 1e3

    assert mm_s == pytest.approx(180.0, rel=0.10)


@pytest.mark.needs_data
@pytest.mark.slow
def test_multirun_seed_sweep_produces_independent_runs(cfg, paths, tmp_path):
    """Regression: each --multirun job must own its output directory.

    Previously every sweep job shared ``outputs/<run_name>``, so the second
    seed silently resumed the first job's checkpoint instead of training a fresh
    model. The CLI now binds the output directory to Hydra's per-job runtime
    directory; this drives a real two-seed sweep and checks the jobs stay
    independent.
    """
    import shutil
    import subprocess
    import sys

    # Train reads only the preprocessed tensors, so stage those under a
    # throwaway root -- no raw frames needed.
    processed = tmp_path / "data" / "processed" / cfg.dataset
    processed.mkdir(parents=True)
    shutil.copy(paths.tensors, processed / "tensors.npz")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "naviernet.cli",
            "--multirun",
            "stage=train",
            "training.steps=1",
            "training.log_every=1",
            f"paths.root={tmp_path}",
            "training.seed=0,1",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    checkpoints = sorted((tmp_path / "outputs" / "multirun").rglob("ckpt.pt"))
    assert len(checkpoints) == 2, f"expected two independent runs, got {checkpoints}"

    # A fresh single-step run reports done == 1. If the second job had resumed
    # the first's checkpoint it would report 2.
    for ckpt in checkpoints:
        state = torch.load(ckpt, weights_only=False)["state"]
        assert state["done"] == 1, f"{ckpt} completed {state['done']} steps (resumed?)"
