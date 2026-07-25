"""Meniscus-centreline segmentation of the bubble.

The imaged bubble edge is a thick dark rim, not a line. These tests pin the two
things that makes right: the mask boundary sits at the *centre* of that rim (not
its outer edge), and thin dark structure (heater/microchannel traces, speckle)
is not mistaken for the bubble.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from naviernet.data.contour import smooth_closed_contour
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


def _radius(polygon: np.ndarray, cy: float, cx: float) -> np.ndarray:
    """Distance of each [x, y] contour point from a centre (cy row, cx col)."""
    return np.hypot(polygon[:, 0] - cx, polygon[:, 1] - cy)


def test_largest_component_keeps_the_bubble_and_drops_speckle():
    mask = np.zeros((60, 60), np.uint8)
    mask[10:50, 10:50] = 1  # the bubble
    mask[2, 2] = mask[57, 57] = 1  # two stray speckles
    kept = _largest_component(mask)
    assert kept is not None
    assert int(kept.sum()) == 40 * 40  # only the block survives


def test_largest_component_is_none_on_an_empty_mask():
    assert _largest_component(np.zeros((10, 10), np.uint8)) is None


def test_midline_sits_between_the_rim_edges_without_spiking():
    # A thick annular rim: inner edge at ~20 px, outer edge at ~40 px.
    shape = (200, 200)
    outer = _ellipse(shape, 100, 100, 40, 40)
    inner = _ellipse(shape, 100, 100, 20, 20)
    band = (outer & (1 - inner)).astype(np.uint8)

    mid = _meniscus_midline(band, min_hole_fraction=0.05)
    radius = _radius(mid, 100, 100)
    # The interface lands near the band centre (~30 px), between inner and outer,
    # and is a clean ring: a small radius spread, no medial-axis spike.
    assert 27 <= radius.mean() <= 33
    assert radius.std() < 2.0


def test_midline_falls_back_to_the_outer_edge_for_a_solid_nucleus():
    # A small dark blob with no enclosed interior: nothing to centre in.
    band = _ellipse((200, 200), 100, 100, 15, 15)
    mid = _meniscus_midline(band, min_hole_fraction=0.05)
    assert 13 <= _radius(mid, 100, 100).mean() <= 16  # the outer edge itself


def test_midline_falls_back_when_the_rim_is_cut_by_the_frame_edge():
    # A rim running off the left edge is not a closed ring; centring is unsafe.
    shape = (200, 200)
    outer = _ellipse(shape, 100, 40, 40, 60)
    inner = _ellipse(shape, 100, 40, 20, 40)
    band = (outer & (1 - inner)).astype(np.uint8)
    band[:, 0] = band[:, 0] | outer[:, 0]  # touches the left border
    assert band[:, 0].any()

    mid = _meniscus_midline(band, min_hole_fraction=0.05)
    # Fallback traces the outer edge, reaching ~col 100, not the ~col 80 inner.
    assert mid.ndim == 2 and mid.shape[1] == 2
    assert mid[:, 0].max() >= 95


def test_smoothing_a_noisy_circle_recovers_the_circle():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    r = 50.0
    rng = np.random.default_rng(0)
    noisy = np.column_stack([r * np.cos(t), r * np.sin(t)]) + rng.normal(0, 1.5, (400, 2))

    smooth = smooth_closed_contour(noisy, sigma=4.0, n_points=360)
    radii = np.hypot(smooth[:, 0], smooth[:, 1])
    noisy_radii = np.hypot(noisy[:, 0], noisy[:, 1])
    assert smooth.shape == (360, 2)
    assert abs(radii.mean() - r) < 2.0  # radius preserved
    assert radii.std() < 0.6 * noisy_radii.std()  # jitter substantially reduced


def test_smoothing_follows_the_shape_it_does_not_flatten_it():
    # An elongated ellipse: a global fit would bow off it; a local low-pass keeps
    # its extent. Corners of the jittered outline should still be tracked.
    t = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    ell = np.column_stack([120 * np.cos(t), 30 * np.sin(t)])
    rng = np.random.default_rng(1)
    smooth = smooth_closed_contour(ell + rng.normal(0, 0.8, ell.shape), sigma=3.0)
    assert abs(smooth[:, 0].max() - 120) < 3 and abs(smooth[:, 1].max() - 30) < 3


def test_smoothing_leaves_a_contour_too_short_to_smooth_untouched():
    pts = np.array([[0, 0], [2, 0], [2, 2]], float)
    assert np.array_equal(smooth_closed_contour(pts, sigma=3.0), pts)


def test_smoothing_is_disabled_by_a_nonpositive_scale():
    pts = np.array([[0, 0], [4, 0], [4, 4], [0, 4], [2, 5]], float)
    assert np.array_equal(smooth_closed_contour(pts, sigma=0.0), pts)


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

    interface = segment_frame(cfg, paths, 1, roi=(0, 200))

    # A smooth closed [x, y] contour, not a mask.
    assert interface.ndim == 2 and interface.shape[1] == 2
    # The boundary sits at the rim centre: between the inner (26/80) and outer
    # (50/120) edges, not on the outer edge.
    ry = (interface[:, 1].max() - interface[:, 1].min()) / 2
    rx = (interface[:, 0].max() - interface[:, 0].min()) / 2
    assert 34 <= ry <= 44 and 92 <= rx <= 108
    # It does not bleed out to the thin microchannel traces at the frame edges.
    assert interface[:, 0].min() > 2 and interface[:, 0].max() < 398
