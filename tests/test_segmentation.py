"""Meniscus-centreline segmentation of the bubble.

The imaged bubble edge is a thick dark rim, not a line. These tests pin the two
things that makes right: the mask boundary sits at the *centre* of that rim (not
its outer edge), and thin dark structure (heater/microchannel traces, speckle)
is not mistaken for the bubble.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from naviernet.data.preprocess import (
    _largest_component,
    _meniscus_midline,
    segment_frame,
)
from naviernet.utils.paths import RunPaths

from .conftest import make_config


def _ellipse(shape, cy, cx, ay, ax) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (((yy - cy) / ay) ** 2 + ((xx - cx) / ax) ** 2 <= 1.0).astype(np.uint8)


def _outer_extent(mask: np.ndarray) -> tuple[int, int]:
    """Half-height and half-width of a mask's bounding box, in pixels."""
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    return (rows[-1] - rows[0]) // 2, (cols[-1] - cols[0]) // 2


def test_largest_component_keeps_the_bubble_and_drops_speckle():
    mask = np.zeros((60, 60), np.uint8)
    mask[10:50, 10:50] = 1  # the bubble
    mask[2, 2] = mask[57, 57] = 1  # two stray speckles
    kept = _largest_component(mask)
    assert kept is not None
    assert int(kept.sum()) == 40 * 40  # only the block survives


def test_largest_component_is_none_on_an_empty_mask():
    assert _largest_component(np.zeros((10, 10), np.uint8)) is None


def test_midline_sits_between_the_rim_edges():
    # A thick annular rim: inner edge at ~20 px, outer edge at ~40 px.
    shape = (200, 200)
    outer = _ellipse(shape, 100, 100, 40, 40)
    inner = _ellipse(shape, 100, 100, 20, 20)
    band = (outer & (1 - inner)).astype(np.uint8)

    mid = _meniscus_midline(band, min_hole_fraction=0.05)
    ry, rx = _outer_extent(mid)
    # The interface must land near the band centre (~30 px), well inside the
    # outer edge (40) and outside the inner edge (20).
    assert 27 <= ry <= 33 and 27 <= rx <= 33


def test_midline_falls_back_to_the_filled_outline_for_a_solid_nucleus():
    # A small dark blob with no enclosed interior: nothing to centre in.
    band = _ellipse((200, 200), 100, 100, 15, 15)
    mid = _meniscus_midline(band, min_hole_fraction=0.05)
    ry, rx = _outer_extent(mid)
    assert 14 <= ry <= 16 and 14 <= rx <= 16  # the outline itself


def test_midline_falls_back_when_the_rim_is_cut_by_the_frame_edge():
    # A rim running off the left edge is not a closed ring; centring is unsafe.
    shape = (200, 200)
    outer = _ellipse(shape, 100, 40, 40, 60)
    inner = _ellipse(shape, 100, 40, 20, 40)
    band = (outer & (1 - inner)).astype(np.uint8)
    band[:, 0] = band[:, 0] | outer[:, 0]  # touches the left border
    assert band[:, 0].any()

    mid = _meniscus_midline(band, min_hole_fraction=0.05)
    # Fallback fills to the outer edge, so the interior hole is covered.
    assert int(mid.sum()) >= int((outer & (1 - inner)).sum())


def _synthetic_frame(path) -> None:
    """A bright frame with a ringed bubble over thin dark microchannel traces."""
    h, w = 200, 400
    img = np.full((h, w), 210, np.uint8)
    # Microchannel traces: thin dark verticals the opening must erase.
    for x in range(0, w, 12):
        img[:, x : x + 1] = 40
    outer = _ellipse((h, w), 100, 200, 50, 120)
    inner = _ellipse((h, w), 100, 200, 26, 80)
    band = (outer & (1 - inner)).astype(bool)
    img[band] = 30  # the dark meniscus rim
    img[inner.astype(bool)] = 190  # interior seen through the vapour
    Image.fromarray(img).save(path)


def test_segment_frame_traces_the_meniscus_and_ignores_traces(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _synthetic_frame(raw / "1.tif")
    cfg = make_config([f"paths.raw_dir={raw}"])
    paths = RunPaths.from_config(cfg)

    mask = segment_frame(cfg, paths, 1, roi=(0, 200))

    # One clean blob: the thin verticals were opened away, not captured.
    assert _largest_component(mask) is not None
    n_components, _ = __import__("cv2").connectedComponents(mask)
    assert n_components == 2  # background + the single bubble
    # The boundary sits at the rim centre: between the inner (26/80) and outer
    # (50/120) edges, not on the outer edge.
    ry, rx = _outer_extent(mask)
    assert 34 <= ry <= 44 and 92 <= rx <= 108
    # It does not bleed out to the microchannel traces at the frame edges.
    assert not mask[:, 0].any() and not mask[:, -1].any()
