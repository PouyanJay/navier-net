"""Trainer mechanics and end-to-end behaviour against the real dataset."""

from __future__ import annotations

import pytest
import torch

from naviernet.pipeline import Pipeline
from naviernet.training import REBALANCED_TERMS, _rebalance
from naviernet.utils.paths import RunPaths


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

    assert cfg.training.holdout_frame not in data._ti[data._train_idx]


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
    iou = frame_iou(cfg, model, data, cfg.training.holdout_frame)
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
