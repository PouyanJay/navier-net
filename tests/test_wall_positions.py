"""Channel-wall positions: recorded by preprocessing, read by the dataset's Domain."""

from __future__ import annotations

import json

import numpy as np
import pytest

from naviernet.data.dataset import BubbleDataset
from naviernet.data.preprocess import Calibration, wall_positions_star

from .conftest import staged_run, write_growing_bubble

WALL_TOP, WALL_BOTTOM = 469, 647  # image rows, as detected on Series-1
MARGIN_TOP, MARGIN_BOTTOM = 3, 2
CHANNEL_WIDTH_UM, L_REF_UM = 300.0, 300.0


def _calibration() -> Calibration:
    um_per_px = CHANNEL_WIDTH_UM / (WALL_BOTTOM - WALL_TOP)
    roi = (WALL_TOP + MARGIN_TOP, WALL_BOTTOM - MARGIN_BOTTOM)
    return Calibration(WALL_TOP, WALL_BOTTOM, um_per_px, roi)


def test_walls_are_one_channel_width_apart_in_star_units():
    walls = wall_positions_star(_calibration(), L_REF_UM)

    assert walls["y_wall_hi_star"] - walls["y_wall_lo_star"] == pytest.approx(
        CHANNEL_WIDTH_UM / L_REF_UM
    )
    assert walls["y_center_star"] == pytest.approx(
        0.5 * (walls["y_wall_lo_star"] + walls["y_wall_hi_star"])
    )


def test_walls_sit_at_the_detected_rows_outside_the_retained_crop():
    calibration = _calibration()
    pixel = calibration.um_per_px / L_REF_UM
    roi_top, roi_bottom = calibration.roi
    first_row_centre = 0.5 * pixel
    last_row_centre = (roi_bottom - roi_top - 0.5) * pixel

    walls = wall_positions_star(calibration, L_REF_UM)

    assert walls["y_wall_lo_star"] == pytest.approx((WALL_TOP - roi_top + 0.5) * pixel)
    assert walls["y_wall_hi_star"] == pytest.approx((WALL_BOTTOM - roi_top + 0.5) * pixel)
    assert walls["y_wall_lo_star"] < first_row_centre
    assert walls["y_wall_hi_star"] > last_row_centre


def _write_bubble_with_meta(path, extra_meta: dict | None) -> None:
    """The synthetic growth event, with ``extra_meta`` merged into its meta."""
    write_growing_bubble(path)
    archive = dict(np.load(path))
    meta = json.loads(str(archive["meta"]))
    meta.update(extra_meta or {})
    archive["meta"] = json.dumps(meta)
    np.savez_compressed(path, **archive)


def test_domain_reads_the_wall_positions_from_the_archive(tmp_path):
    walls = {"y_wall_lo_star": -0.02, "y_wall_hi_star": 0.52, "y_center_star": 0.25}
    cfg, paths = staged_run(tmp_path, write=lambda path: _write_bubble_with_meta(path, walls))

    domain = BubbleDataset(cfg, paths).domain

    assert (domain.y_wall_lo, domain.y_wall_hi) == pytest.approx((-0.02, 0.52))
    assert domain.y_center == pytest.approx(0.25)
    assert domain.y_wall_lo < domain.y_min
    assert domain.y_wall_hi > domain.y_max


def test_domain_falls_back_to_the_crop_edges_for_an_archive_without_walls(tmp_path, caplog):
    cfg, paths = staged_run(tmp_path, write=lambda path: _write_bubble_with_meta(path, None))

    with caplog.at_level("WARNING"):
        domain = BubbleDataset(cfg, paths).domain

    assert domain.y_wall_lo == pytest.approx(domain.y_min)
    assert domain.y_wall_hi == pytest.approx(domain.y_max)
    assert "crop edges" in caplog.text
