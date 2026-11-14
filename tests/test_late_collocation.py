"""Denser collocation at the end of the time window."""

from __future__ import annotations

import numpy as np
import pytest

from naviernet.data.dataset import BubbleDataset

from .conftest import staged_run

N = 4096
LATE = ["training.collocation_late_share=0.5", "training.collocation_late_frac=0.2"]


def _dataset(tmp_path, overrides=()) -> BubbleDataset:
    cfg, paths = staged_run(tmp_path, list(overrides))
    return BubbleDataset(cfg, paths)


def _late_fraction(data: BubbleDataset, times: np.ndarray, frac: float) -> float:
    d = data.domain
    return float(np.mean(times >= d.t_max - frac * (d.t_max - d.t_min)))


def test_off_by_default_the_draw_is_unchanged(tmp_path):
    # Arrange
    plain = _dataset(tmp_path / "plain")
    explicit = _dataset(tmp_path / "explicit", ["training.collocation_late_share=0.0"])

    # Act
    a = plain.sample_collocation(N, np.random.default_rng(3), 8).detach().numpy()
    b = explicit.sample_collocation(N, np.random.default_rng(3), 8).detach().numpy()

    # Assert
    np.testing.assert_array_equal(a, b)


def test_the_late_share_lands_at_the_end_of_the_window(tmp_path):
    # Arrange
    data = _dataset(tmp_path, LATE)

    # Act
    points = data.sample_collocation(N, np.random.default_rng(0)).detach().numpy()

    # Assert: half the uniform half, plus the uniform half's own 20% of the rest.
    uniform = points[: N // 2, 2]
    assert _late_fraction(data, uniform, 0.2) == pytest.approx(0.5 + 0.5 * 0.2, abs=0.03)


def test_binned_late_points_snap_to_their_own_instants(tmp_path):
    # Arrange
    data = _dataset(tmp_path, [*LATE, "training.collocation_late_bins=6"])
    late_start = data.domain.t_max - 0.2 * (data.domain.t_max - data.domain.t_min)

    # Act
    points = data.sample_collocation(N, np.random.default_rng(1), 4).detach().numpy()

    # Assert
    late = points[: N // 4, 2]
    assert late.min() >= late_start - 1e-6
    assert len(np.unique(late)) == 6


def test_an_out_of_range_share_is_refused(tmp_path):
    # Arrange
    data = _dataset(
        tmp_path, ["training.collocation_late_share=1.5", "training.collocation_late_frac=0.2"]
    )

    # Act / Assert
    with pytest.raises(ValueError, match="collocation_late_share"):
        data.sample_collocation(N, np.random.default_rng(0))
