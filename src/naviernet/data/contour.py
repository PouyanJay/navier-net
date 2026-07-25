"""Smooth a closed interface contour with a local arc-length low-pass.

A bubble interface is a smooth closed curve, but a contour traced from a pixel
mask is a staircase of pixel-scale steps. Smoothing it must be *local*: a global
fit (e.g. truncated Fourier descriptors) has a fixed budget of detail for the
whole perimeter, so on an elongated bubble it bows away from the true edge to
spend that budget elsewhere.

Convolving the closed curve with a Gaussian along its arc length instead has a
single length-scale knob, ``sigma``, in the coordinate units of the points:
features finer than ``sigma`` (the staircase, microchannel nicks) are erased,
everything coarser (the bubble's actual shape) is followed. The curve is
resampled densely enough to resolve ``sigma`` first, so the knob means the same
thing whether the points are mask pixels or non-dimensional ``x*``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Pixel length scale below which contour detail is treated as staircase noise.
DEFAULT_SMOOTH_PX = 3.0

# Samples per sigma when resampling: enough to resolve the Gaussian, cheaply.
_SAMPLES_PER_SIGMA = 4


def smooth_closed_contour(
    points: np.ndarray, sigma: float = DEFAULT_SMOOTH_PX, n_points: int = 360
) -> np.ndarray:
    """Low-pass a closed contour, returned as ``n_points`` ordered ``[x, y]``.

    ``points`` is an ordered ``(N, 2)`` closed contour (first and last need not
    coincide). A non-positive ``sigma`` or a contour too short to smooth is
    returned unchanged so callers need no special-casing.
    """
    pts = np.asarray(points, dtype=np.float64)
    if sigma <= 0 or len(pts) < 4:
        return pts

    # Arc length around the closed curve (back to the first point).
    steps = np.hypot(*np.diff(pts, axis=0, append=pts[:1]).T)
    arc = np.concatenate(([0.0], np.cumsum(steps)))
    perimeter = arc[-1]
    if perimeter == 0:
        return pts

    # Resample uniformly, fine enough to resolve sigma regardless of coordinate
    # scale, then convolve periodically so the seam is not a discontinuity.
    work = max(n_points, int(perimeter / (sigma / _SAMPLES_PER_SIGMA)) + 1)
    sample = np.linspace(0.0, perimeter, work, endpoint=False)
    xs = np.interp(sample, arc, np.append(pts[:, 0], pts[0, 0]))
    ys = np.interp(sample, arc, np.append(pts[:, 1], pts[0, 1]))
    sigma_samples = sigma * work / perimeter
    xs = gaussian_filter1d(xs, sigma_samples, mode="wrap")
    ys = gaussian_filter1d(ys, sigma_samples, mode="wrap")

    keep = np.linspace(0, work, n_points, endpoint=False).astype(int)
    return np.column_stack([xs[keep], ys[keep]])
