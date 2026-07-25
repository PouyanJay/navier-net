"""Meniscus-centreline segmentation of the bubble.

The imaged bubble edge is a thick dark rim, not a line. These tests pin the two
things that makes right: the mask boundary sits at the *centre* of that rim (not
its outer edge), and thin dark structure (heater/microchannel traces, speckle)
is not mistaken for the bubble.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from naviernet.data.contour import fair_closed_contour
from naviernet.data.preprocess import (
    _largest_component,
    _meniscus_interface,
    segment_frame,
)
from naviernet.utils.paths import RunPaths

from .conftest import make_config


def _max_turn_deg(curve: np.ndarray) -> float:
    """Largest turning angle between consecutive segments of a closed curve."""
    d = np.diff(curve, axis=0, prepend=curve[-1:])
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs((np.diff(ang, prepend=ang[-1:]) + np.pi) % (2 * np.pi) - np.pi)
    return float(np.degrees(turn).max())


def _ellipse(shape, cy, cx, ay, ax) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (((yy - cy) / ay) ** 2 + ((xx - cx) / ax) ** 2 <= 1.0).astype(np.uint8)


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


def test_interface_settles_on_the_rim_centreline():
    # A thick annular rim: inner edge at ~20 px, outer edge at ~40 px.
    shape = (200, 200)
    outer = _ellipse(shape, 100, 100, 40, 40)
    inner = _ellipse(shape, 100, 100, 20, 20)
    band = (outer & (1 - inner)).astype(np.uint8)

    curve = _meniscus_interface(band, min_hole_fraction=0.05, seed_blur_px=5.0)
    radius = _radius(curve, 100, 100)
    # The active contour lands near the band centre (~30 px), a clean ring.
    assert 27 <= radius.mean() <= 33
    assert radius.std() < 2.0


def test_interface_stays_smooth_on_a_jagged_band():
    # A rim whose outer edge wobbles fast with angle: a discrete centreline
    # would kink, but the active contour's rigidity smooths it out.
    yy, xx = np.ogrid[:200, :200]
    ang = np.arctan2(yy - 100, xx - 100)
    r = np.hypot(yy - 100, xx - 100)
    band = ((r <= 40 + 4 * np.sin(17 * ang)) & (r >= 20)).astype(np.uint8)

    curve = _meniscus_interface(band, min_hole_fraction=0.05, seed_blur_px=5.0)
    assert _max_turn_deg(curve) < 25  # smooth despite the jagged rim
    assert 26 <= _radius(curve, 100, 100).mean() <= 34  # still on the centre


def test_interface_falls_back_to_the_outer_edge_for_a_solid_nucleus():
    # A small dark blob with no enclosed interior: nothing to centre in.
    band = _ellipse((200, 200), 100, 100, 15, 15)
    curve = _meniscus_interface(band, min_hole_fraction=0.05, seed_blur_px=5.0)
    assert 13 <= _radius(curve, 100, 100).mean() <= 16  # the outer edge itself


def _curvature(curve: np.ndarray) -> np.ndarray:
    x, y = curve[:, 0], curve[:, 1]
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)


def test_fairing_a_noisy_circle_recovers_the_circle():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    r = 50.0
    rng = np.random.default_rng(0)
    noisy = np.column_stack([r * np.cos(t), r * np.sin(t)]) + rng.normal(0, 1.5, (400, 2))

    fair = fair_closed_contour(noisy, wavelength_px=40.0, n_points=360)
    radii = np.hypot(fair[:, 0], fair[:, 1])
    noisy_radii = np.hypot(noisy[:, 0], noisy[:, 1])
    assert fair.shape == (360, 2)
    assert abs(radii.mean() - r) < 2.0  # radius preserved
    assert radii.std() < 0.5 * noisy_radii.std()  # jitter removed


def test_fairing_smooths_the_curvature_of_a_bumpy_curve():
    # A circle with a fast radial wobble: fairing removes it and the curvature
    # stops jumping, while the overall size is preserved.
    t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    radius = 60 + 3 * np.sin(21 * t)
    bumpy = np.column_stack([radius * np.cos(t), radius * np.sin(t)])

    fair = fair_closed_contour(bumpy, wavelength_px=40.0)
    jump = np.abs(np.diff(_curvature(fair))).max()
    assert jump < np.abs(np.diff(_curvature(bumpy))).max() / 3  # curvature settles
    assert abs(np.hypot(fair[:, 0], fair[:, 1]).mean() - 60) < 2


def test_fairing_is_scale_invariant_and_keeps_an_elongated_shape():
    # A long ellipse gets more harmonics than a small circle at the same
    # wavelength, so its ends are not bowed in.
    t = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    ell = np.column_stack([200 * np.cos(t), 40 * np.sin(t)])
    fair = fair_closed_contour(ell, wavelength_px=60.0)
    assert abs(fair[:, 0].max() - 200) < 3 and abs(fair[:, 1].max() - 40) < 3


def test_fairing_is_disabled_by_a_nonpositive_wavelength():
    pts = np.array([[0, 0], [4, 0], [4, 4], [0, 4], [2, 5]], float)
    assert np.array_equal(fair_closed_contour(pts, wavelength_px=0.0), pts)


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
