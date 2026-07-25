"""Smooth a closed interface contour with truncated Fourier descriptors.

A bubble interface is a smooth closed curve, but a contour traced from a pixel
mask is a staircase: it carries high-frequency steps and the little bumps where
microchannel features graze the rim. Resampling the closed curve uniformly in
arc length and keeping only its low Fourier harmonics discards that detail while
preserving the shape, with the number of harmonics as the single smoothness
knob -- more harmonics keep finer features, fewer round the curve off.
"""

from __future__ import annotations

import numpy as np

# Enough harmonics to keep the teardrop nose, few enough to shed pixel jaggies.
DEFAULT_HARMONICS = 12


def smooth_closed_contour(
    points: np.ndarray, n_harmonics: int = DEFAULT_HARMONICS, n_points: int = 720
) -> np.ndarray:
    """Low-pass a closed contour, returned as ``n_points`` ordered ``[x, y]``.

    ``points`` is an ordered ``(N, 2)`` closed contour (first and last need not
    coincide). Contours too short to smooth, or a non-positive ``n_harmonics``,
    are returned unchanged so callers need no special-casing.
    """
    pts = np.asarray(points, dtype=np.float64)
    if n_harmonics <= 0 or len(pts) < 2 * n_harmonics + 1:
        return pts

    # Resample uniformly in arc length so harmonics mean the same thing all the
    # way round, regardless of how densely the tracer placed points.
    steps = np.hypot(*np.diff(pts, axis=0, append=pts[:1]).T)
    arc = np.concatenate(([0.0], np.cumsum(steps)))
    perimeter = arc[-1]
    if perimeter == 0:
        return pts
    sample = np.linspace(0.0, perimeter, n_points, endpoint=False)
    xs = np.interp(sample, arc, np.append(pts[:, 0], pts[0, 0]))
    ys = np.interp(sample, arc, np.append(pts[:, 1], pts[0, 1]))

    # Treat the curve as a complex signal; zero the harmonics above the cutoff.
    spectrum = np.fft.fft(xs + 1j * ys)
    order = np.fft.fftfreq(n_points, d=1.0 / n_points)
    spectrum[np.abs(order) > n_harmonics] = 0.0
    smoothed = np.fft.ifft(spectrum)
    return np.column_stack([smoothed.real, smoothed.imag])
