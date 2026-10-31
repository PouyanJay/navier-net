"""Background removal for the camera frames.

The heaters, electrodes and channel walls do not move; the bubble does. Their
appearance is the same in every event frame away from the bubble, so a
per-pixel median over the frames recovers them. What remains after removing
them is the bubble, its dark rim and the liquid film beside it.

The film is found the way a reader sees it: measure how far the darkness
reaches from the measured outline, cut where it falls off, and keep only the
dark pixels inside that reach. Bright halo and heater bars next to the edge
are brighter than the set-up and are flattened.

Inside the bubble and its rim the background cannot be measured reliably: the
bubble covers those pixels in most frames. The band is therefore smoothed with
a normalised Gaussian (weights only from inside the band, so nothing bleeds in
from outside). The sigma is where the heater bars stop showing.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage

# Darkness below this many grey levels is set-up noise, not film.
DARKNESS_FLOOR = 18.0
# Darkness at the outline: film is kept above this share of it.
FILM_SHARE_OF_EDGE = 0.4
# The reach of the film is where its darkness falls below this share of the peak.
REACH_FALLOFF = 0.4
MIN_REACH_PX = 6
MAX_REACH_PX = 40
DARKNESS_SMOOTH_SIGMA = 2.5
CONTOUR_SIGMA = 2.5
# Keep the dark rim and the bright halo at its edge in the smoothed band.
RIM_PX = 4
SMOOTH_SIGMA = 8.0
EDGE_SOFT_SIGMA = 1.0
FALLBACK_TONE = 140.0
_DISK = ndimage.generate_binary_structure(2, 1)


def static_background(stack: np.ndarray, bubble: np.ndarray) -> np.ndarray:
    """Per-pixel median of the frames where the pixel is outside the bubble.

    ``stack`` is ``(T, H, W)`` grey levels and ``bubble`` the matching boolean
    masks. A pixel the bubble covers in every frame falls back to the plain
    median, which then includes the bubble: the only place the estimate is
    weaker, and the bubble is kept there anyway.
    """
    free = np.where(bubble, np.nan, stack)
    with warnings.catch_warnings():
        # A pixel covered in every frame is all-NaN: expected, handled below.
        warnings.simplefilter("ignore", RuntimeWarning)
        estimate = np.nanmedian(free, axis=0)
    return np.where(np.isfinite(estimate), estimate, np.median(stack, axis=0))


def _film_reach(darkness: np.ndarray, distance: np.ndarray) -> int:
    """How many pixels from the outline the darkness of the film extends."""
    profile = []
    for d in range(1, MAX_REACH_PX + 1):
        ring = np.round(distance) == d
        profile.append(float(darkness[ring].mean()) if ring.any() else 0.0)
    profile_arr = np.asarray(profile)
    peak = float(profile_arr[:MIN_REACH_PX].max())
    below = np.nonzero(profile_arr[MIN_REACH_PX:] < REACH_FALLOFF * peak)[0]
    return MIN_REACH_PX + int(below[0]) + 1 if below.size else MAX_REACH_PX


def _film(darkness: np.ndarray, bubble: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """The dark film around the outline, within its measured reach."""
    reach = _film_reach(darkness, distance)
    edge = darkness[(distance > 0) & (distance <= 3)]
    threshold = max(
        DARKNESS_FLOOR * FILM_SHARE_OF_EDGE, FILM_SHARE_OF_EDGE * float(np.median(edge))
    )
    film = (darkness > threshold) & (distance <= reach)
    film = ndimage.binary_opening(film, structure=_DISK, iterations=2)
    return ndimage.binary_closing(film | bubble, iterations=3)


def _normalized_blur(values: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of ``values`` counting only the pixels where ``weights`` holds."""
    w = weights.astype(np.float32)
    numerator = ndimage.gaussian_filter(values * w, sigma)
    denominator = ndimage.gaussian_filter(w, sigma)
    return numerator / np.maximum(denominator, 1e-6)


def clean_frame(frame: np.ndarray, background: np.ndarray, bubble: np.ndarray) -> np.ndarray:
    """The frame with everything but the bubble and its film flattened."""
    if not bubble.any():
        return np.full(frame.shape, float(np.median(frame)), dtype=np.float32)
    distance = ndimage.distance_transform_edt(~bubble)
    darkness = ndimage.gaussian_filter(
        np.clip(background - frame, 0.0, None), DARKNESS_SMOOTH_SIGMA
    )
    keep = _film(darkness, bubble, distance) | bubble
    # The outline is smoothed so no pixel-level ragged edge survives.
    keep = ndimage.binary_fill_holes(
        ndimage.gaussian_filter(keep.astype(np.float32), CONTOUR_SIGMA) > 0.5
    )
    tone = float(np.median(frame[~keep])) if (~keep).any() else FALLBACK_TONE
    band = ndimage.binary_dilation(bubble, iterations=RIM_PX)
    base = np.where(keep, frame, tone)
    base = np.where(band, _normalized_blur(frame, band, SMOOTH_SIGMA), base)
    soft = ndimage.gaussian_filter(keep.astype(np.float32), EDGE_SOFT_SIGMA)
    return (soft * base + (1.0 - soft) * tone).astype(np.float32)
