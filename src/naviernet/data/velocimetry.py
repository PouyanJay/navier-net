"""Measured front kinematics: how fast the interface actually moved.

The shape is supervised at every training frame, but its RATE is not. Under the
front-geometry parameterisation the width profile travels along ``rate(u)`` and
the nose along its own integrated rate, and those rates are what carry the shape
past the last supervised frame -- yet nothing measures them. They are inferred
indirectly, from a handful of snapshots. This module measures them.

**What two masks can and cannot tell you.** Not the material velocity of a
surface point: a curve sliding along itself looks identical between frames, so
the tangential component is unobservable (the aperture problem) and tracking a
generic point on a smooth arc is not possible. That costs nothing here, because
only the NORMAL component moves an interface -- tangential motion does not
change the shape at all. So the measurable part, the normal velocity, is
complete information for the shape's evolution.

**How it is measured.** The preprocessed archive already carries a signed
distance per frame (negative inside the vapour), which makes the interface the
zero level set of ``sdf`` and the front's own motion the level-set advection
speed::

    sdf_t + v_n * |grad sdf| = 0   =>   v_n = -(sdf_next - sdf_now) / (dt * |grad sdf_now|)

Positive is outward -- out of the vapour -- matching the sign convention of the
model's own :attr:`~naviernet.models.geometry.FrontSamples.normal_speed`.

Off the zero contour this same expression is the level-set EXTENSION velocity:
the normal speed of the front, carried out into a neighbourhood. That is what
makes it legitimate to evaluate at the model's front rather than the measured
one -- early in training the two do not coincide, and a point-to-point pairing
would either invent a correspondence or compare the wrong points.

**Where this estimate is weakest, and why that is covered.** The difference is
taken at a FIXED point over a whole frame, so it is first-order accurate in the
distance the front travels in that time -- fine while the normal displacement is
small against the local radius of curvature, and increasingly low-biased when it
is not. Measured on Series-1: along the body the front moves ~2 px per frame
against a ~75 px radius and the estimate is solid, but at the fast nose cap it
moves ~52 px and reads about 15% below the apex's own tracked displacement.

That is exactly the division of labour the two supervision terms have. The
profile term carries the body, where this estimate is accurate and where
``rate(u)`` actually lives; the nose is supervised by the apex's exact 2-D
displacement instead, which involves no level-set approximation at all.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter

# Below this, ``|grad sdf|`` carries no usable direction: dividing by it turns
# segmentation noise into an arbitrarily large velocity. Such pixels are reported
# as unresolvable instead of being floored into a plausible-looking number. The
# scale is set by a true distance field, where the magnitude is 1 by
# construction -- so this is three orders of magnitude below "normal".
GRAD_FLOOR = 1e-3

# How far a Gaussian of a given sigma is treated as reaching, in sigmas. Two
# covers ~95% of its mass, which is the radius within which a neighbouring
# pixel's value materially changes this one's after the blur.
BLUR_REACH_SIGMAS = 2.0


def normal_velocity(
    sdf_now: np.ndarray,
    sdf_next: np.ndarray,
    dt: float,
    spacing: tuple[float, float],
    smooth_px: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The measured outward normal velocity at the ``sdf_now`` interface.

    ``sdf_now``/``sdf_next`` are one frame pair's signed distance fields
    ``(H, W)``, ``dt`` their separation on the non-dimensional time axis, and
    ``spacing`` the grid step ``(dy, dx)`` in the same length units the fields
    are expressed in -- so the result is in those units per unit time.

    ``smooth_px`` is an isotropic Gaussian applied to BOTH fields before
    differencing, in pixels. A binary mask quantises the interface to the pixel
    grid, and its signed distance inherits that staircase; differentiating it
    raw puts single-pixel noise straight into the velocity. Smoothing both
    fields identically shifts neither, so it costs no bias.

    Returns ``(v_n, resolvable)``: the velocity field, and the pixels where
    ``|grad sdf_now|`` was large enough to define a direction. ``v_n`` is zero
    where it is not -- a value no consumer should read, which is what
    ``resolvable`` is for.
    """
    if dt <= 0:
        raise ValueError(f"dt={dt} must be > 0 to measure a velocity from two frames")
    if sdf_now.shape != sdf_next.shape:
        raise ValueError(
            f"frame pair has mismatched shapes {sdf_now.shape} and {sdf_next.shape}"
        )
    if smooth_px < 0:
        raise ValueError(f"smooth_px={smooth_px} must be >= 0")

    now = _smoothed(sdf_now, smooth_px)
    later = _smoothed(sdf_next, smooth_px)
    d_dy, d_dx = np.gradient(now, spacing[0], spacing[1])
    magnitude = np.hypot(d_dx, d_dy)

    resolvable = magnitude > GRAD_FLOOR
    v_n = np.zeros_like(now)
    np.divide(-(later - now) / dt, magnitude, out=v_n, where=resolvable)
    return v_n.astype(np.float32), resolvable


def survives_blur(valid: np.ndarray, smooth_px: float) -> np.ndarray:
    """The pixels whose smoothed value was built only from valid data.

    The blur in :func:`normal_velocity` runs over the whole frame, so a pixel
    within reach of an invalid one is mixed with whatever the segmentation put
    there -- and would then be reported as a perfectly good measurement. The
    invalid regions are real (the field-of-view cut on the last usable frame,
    see ``imaging.truncated_cols``), so this shrinks validity by the blur's own
    reach rather than trusting a value the blur contaminated.

    A no-op when the blur is off, where each pixel is only ever itself.
    """
    if smooth_px <= 0:
        return valid.astype(bool)
    reach = int(math.ceil(BLUR_REACH_SIGMAS * smooth_px))
    # A pixel survives only if EVERY pixel within the blur's reach was valid,
    # which is exactly a minimum filter over the boolean mask.
    return minimum_filter(valid.astype(bool), size=2 * reach + 1, mode="nearest")


def _smoothed(field: np.ndarray, smooth_px: float) -> np.ndarray:
    """``field`` blurred by ``smooth_px``, or itself when the blur is off."""
    values = field.astype(np.float64)
    return values if smooth_px <= 0 else gaussian_filter(values, smooth_px, mode="nearest")
