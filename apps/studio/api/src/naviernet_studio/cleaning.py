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
CHANGE_FLOOR = 6.0
CHANGE_SIGMAS = 4.0
# Keep the bubble's dark rim and the bright halo at its edge.
RIM_PX = 4
# Smoothing of the bubble band: large enough that the heater bars disappear.
SMOOTH_SIGMA = 8.0
# A changed region counts as bubble only if it reaches this close to the bubble.
ATTACH_PX = 8
# Soften the cut between kept and flattened pixels.
EDGE_SOFT_SIGMA = 0.7
MAD_TO_SIGMA = 1.4826
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
    change = np.abs(frame - background)
    outside = ~rim
    noise = float(np.median(change[outside])) if outside.any() else 0.0
    spread = MAD_TO_SIGMA * noise
    threshold = max(CHANGE_FLOOR, CHANGE_SIGMAS * spread)
    changed = ndimage.binary_opening((change > threshold) & outside, iterations=1)
    labels, count = ndimage.label(changed)
    attached = ndimage.binary_dilation(rim, iterations=ATTACH_PX)
    touching = np.unique(labels[attached & (labels > 0)])
    kept_outside = np.isin(labels, touching[touching > 0]) if count else np.zeros_like(rim)
    keep = rim | kept_outside
    tone = float(np.median(frame[~keep])) if (~keep).any() else FALLBACK_TONE
    base = np.where(keep, frame, tone)
    base = np.where(rim, _normalized_blur(frame, rim, SMOOTH_SIGMA), base)
    soft = ndimage.gaussian_filter(keep.astype(np.float32), EDGE_SOFT_SIGMA)
    return (soft * base + (1.0 - soft) * tone).astype(np.float32)
