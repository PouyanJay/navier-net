"""The channel can run past the camera's view at the outlet end."""

from __future__ import annotations

import numpy as np
import pytest

from naviernet.audit import audit_grid
from naviernet.data.dataset import BubbleDataset

from .conftest import staged_run

EXTENSION_UM = 600.0


def _dataset(tmp_path, extension_um: float) -> BubbleDataset:
    cfg, paths = staged_run(tmp_path, [f"experiment.outlet_extension_um={extension_um}"])
    return BubbleDataset(cfg, paths)


def _extension_star(data: BubbleDataset) -> float:
    return EXTENSION_UM / float(data.cfg.scales.L_ref_um)


def test_no_extension_ends_the_channel_at_the_view(tmp_path):
    # Arrange / Act
    data = _dataset(tmp_path, 0.0)

    # Assert
    assert data.domain.x_max == pytest.approx(float(data.x[-1]))
    assert data.domain.x_view == pytest.approx(float(data.x[-1]))


def test_the_outlet_moves_past_the_view_by_the_extension(tmp_path):
    # Arrange / Act
    data = _dataset(tmp_path, EXTENSION_UM)

    # Assert
    assert data.domain.x_view == pytest.approx(float(data.x[-1]))
    assert data.domain.x_max == pytest.approx(float(data.x[-1]) + _extension_star(data))


def test_the_area_stays_the_view_area(tmp_path):
    # Arrange
    plain = _dataset(tmp_path / "plain", 0.0)

    # Act
    extended = _dataset(tmp_path / "extended", EXTENSION_UM)

    # Assert: image pixel fractions are shares of the view, not of the channel.
    assert extended.domain.area == pytest.approx(plain.domain.area)


def test_the_equations_reach_the_extended_outlet(tmp_path):
    # Arrange
    data = _dataset(tmp_path, EXTENSION_UM)
    rng = np.random.default_rng(0)

    # Act
    outlet = data.sample_outlet(16, rng).detach().numpy()
    _, walls = data.sample_boundary(4096, rng)
    collocation = data.sample_collocation(4096, rng).detach().numpy()

    # Assert
    assert np.allclose(outlet[:, 0], data.domain.x_max)
    assert walls[:, 0].max().item() > data.domain.x_view
    assert collocation[:, 0].max() > data.domain.x_view


def test_the_audit_stays_inside_the_view(tmp_path):
    # Arrange
    data = _dataset(tmp_path, EXTENSION_UM)

    # Act
    grid = audit_grid(data, nx=33, ny=9)

    # Assert
    assert grid.xs[-1] == pytest.approx(data.domain.x_view)


def test_a_view_cut_drops_the_extension(tmp_path):
    # Arrange
    data = _dataset(tmp_path, EXTENSION_UM)

    # Act
    view = data.domain.in_view()

    # Assert
    assert view.x_max == pytest.approx(data.domain.x_view)
    assert view.area == pytest.approx(data.domain.area)


def test_a_negative_extension_is_refused(tmp_path):
    # Arrange
    cfg, paths = staged_run(tmp_path, ["experiment.outlet_extension_um=-10.0"])

    # Act / Assert
    with pytest.raises(ValueError, match="outlet_extension_um"):
        BubbleDataset(cfg, paths)
