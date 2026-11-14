"""The background filter keeps a moving bubble and removes static structure."""

from __future__ import annotations

import numpy as np
from naviernet_studio import cleaning

HEIGHT, WIDTH = 60, 120


def _heater_pattern() -> np.ndarray:
    """Vertical stripes and a horizontal wall: static structure of the set-up."""
    background = np.full((HEIGHT, WIDTH), 150.0, dtype=np.float32)
    background[:, ::9] = 200.0
    background[2:4, :] = 60.0
    return background


def _vapour() -> np.ndarray:
    """The bubble's vapour with the heater bars seen through it."""
    texture = np.full((HEIGHT, WIDTH), 40.0, dtype=np.float32)
    texture[:, ::9] = 70.0
    return texture


def _disc(centre_x: int, radius: int) -> np.ndarray:
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    return (xx - centre_x) ** 2 + (yy - HEIGHT // 2) ** 2 <= radius**2


def test_the_static_background_is_recovered_from_frames_with_a_moving_bubble():
    pattern = _heater_pattern()
    masks = np.stack([_disc(30 + 8 * k, 12) for k in range(6)])
    frames = np.stack([np.where(m, _vapour(), pattern) for m in masks]).astype(np.float32)

    estimate = cleaning.static_background(frames, masks)

    outside = ~_disc(30 + 8 * 2, 12)
    plain = np.ones((HEIGHT, WIDTH), dtype=bool)
    plain[:, ::9] = False  # the bright stripes
    plain[2:4, :] = False  # the dark wall
    assert np.allclose(estimate[outside & plain], 150.0, atol=1e-3)


def test_the_bubble_is_kept_and_the_heaters_are_flattened():
    pattern = _heater_pattern()
    masks = np.stack([_disc(30 + 8 * k, 12) for k in range(6)])
    frames = np.stack([np.where(m, _vapour(), pattern) for m in masks]).astype(np.float32)
    background = cleaning.static_background(frames, masks)
    frame, bubble = frames[3], masks[3]

    cleaned = cleaning.clean_frame(frame, background, bubble)

    far_from_bubble = ~_disc(30 + 8 * 3, 30)
    assert np.std(cleaned[far_from_bubble]) < 1.0
    # The dark rim stays dark; the stripes inside the vapour are smoothed away.
    rim = bubble & ~_disc(30 + 8 * 3, 12 - 3)
    assert np.all(cleaned[rim] < 100.0)
    inside = _disc(30 + 8 * 3, 12 - 4)
    assert np.std(cleaned[inside]) < np.std(frame[inside])


def test_isolated_structure_away_from_the_bubble_is_dropped():
    pattern = _heater_pattern()
    masks = np.stack([_disc(30 + 8 * k, 12) for k in range(6)])
    frames = np.stack([np.where(m, _vapour(), pattern) for m in masks]).astype(np.float32)
    background = cleaning.static_background(frames, masks)
    frame = frames[0].copy()
    frame[50:54, 100:106] = 240.0  # a bright fragment far from the bubble

    cleaned = cleaning.clean_frame(frame, background, masks[0])

    assert cleaned[51, 103] < 180.0
