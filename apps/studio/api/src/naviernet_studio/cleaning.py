"""Background removal for the camera frames.

The heaters, electrodes and channel walls do not move; the bubble does. Their
appearance is therefore the same in every event frame away from the bubble,
and a per-pixel median over the frames recovers them. A pixel that differs
from that background, and is attached to the bubble, is kept; everything else
is replaced by a flat tone.

Inside the bubble and its dark rim the background cannot be measured reliably:
the bubble covers those pixels in most frames. What remains there is the
heater pattern seen through the vapour, which is fine-scale, while the bubble's
own appearance (the rim, the shading, the bright nose) is coarse. The band is
therefore smoothed with a normalised Gaussian (weights only from inside the
band, so nothing bleeds in from outside). The sigma is set where the heater
bars stop showing; a smaller sigma keeps the bars.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage

# A change must exceed this many grey levels, and this many robust sigmas of the
# background noise, to count as bubble.
CHANGE_FLOOR = 18.0
CHANGE_SIGMAS = 4.0
# Keep the bubble's dark rim and the bright halo at its edge.
RIM_PX = 4
# Smoothing of the bubble band: large enough that the heater bars disappear.
SMOOTH_SIGMA = 8.0
# A film region counts as bubble only if it reaches this close to the bubble.
ATTACH_PX = 8
# Soften the cut between kept and flattened pixels.
EDGE_SOFT_SIGMA = 0.7
MAD_TO_SIGMA = 1.4826
# Round the film's edge so fragments do not show as speckle.
OPEN_PX = 2
_DISK = ndimage.generate_binary_structure(2, 1)
MIN_FILM_PX = 60
CONTOUR_SIGMA = 2.0
FALLBACK_TONE = 140.0


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


def _normalized_blur(values: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of ``values`` counting only the pixels where ``weights`` holds."""
    w = weights.astype(np.float32)
    numerator = ndimage.gaussian_filter(values * w, sigma)
    denominator = ndimage.gaussian_filter(w, sigma)
    return numerator / np.maximum(denominator, 1e-6)


def clean_frame(frame: np.ndarray, background: np.ndarray, bubble: np.ndarray) -> np.ndarray:
    """The frame with everything but the bubble flattened, as float grey levels."""
    rim = ndimage.binary_dilation(bubble, iterations=RIM_PX)
    outside = ~rim
    noise = float(np.median(np.abs(frame - background)[outside]))
    threshold = max(CHANGE_FLOOR, CHANGE_SIGMAS * MAD_TO_SIGMA * noise)
    # The liquid film and the dark rim are darker than the set-up. The bright
    # halo around the edge is brighter, and is not part of the bubble.
    darker = (background - frame) > threshold
    film = ndimage.binary_opening(darker & outside, structure=_DISK, iterations=OPEN_PX)
    film = _drop_small(film, MIN_FILM_PX)
    labels, _ = ndimage.label(film)
    attached = ndimage.binary_dilation(rim, iterations=ATTACH_PX)
    touching = np.unique(labels[attached & (labels > 0)])
    raw_keep = rim | np.isin(labels, touching[touching > 0])
    # The outline itself is smoothed; pixels take their raw value only where
    # they really belong to the bubble or its film.
    contour = ndimage.gaussian_filter(raw_keep.astype(np.float32), CONTOUR_SIGMA) > 0.5
    contour = ndimage.binary_fill_holes(contour | bubble)
    tone = float(np.median(frame[~raw_keep])) if (~raw_keep).any() else FALLBACK_TONE
    # Outside the rim only the darker film is real. Bright halo and heater bars
    # next to the edge are background, whatever region they fell into.
    base = np.where(raw_keep & (rim | darker), frame, tone)
    base = np.where(rim, _normalized_blur(frame, rim, SMOOTH_SIGMA), base)
    soft = ndimage.gaussian_filter(
        (contour & (raw_keep | rim)).astype(np.float32), EDGE_SOFT_SIGMA
    )
    return (soft * base + (1.0 - soft) * tone).astype(np.float32)


def _drop_small(mask: np.ndarray, minimum: int) -> np.ndarray:
    """Remove connected regions smaller than ``minimum`` pixels."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    keep = np.zeros(count + 1, dtype=bool)
    keep[1:] = sizes >= minimum
    return keep[labels]
