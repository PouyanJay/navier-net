"""Fair a closed interface contour to a curve of smoothly varying curvature.

A bubble interface is not just smooth in position but *fair*: its curvature (the
second derivative) varies smoothly, with no sudden jumps. Truncated Fourier
descriptors give exactly that -- a band-limited curve is C-infinity, so every
derivative is smooth -- but a *fixed* harmonic count bows a long curve (too few
harmonics for its detail) and under-smooths a short one.

Scaling the cut-off with the perimeter fixes both: keep harmonics down to a
fixed arc-length ``wavelength``, so ``harmonics = perimeter / wavelength``. The
smoothing is then scale-invariant -- the same physical detail is removed whether
the bubble is small or large -- a small bubble keeps few harmonics (its foot
smooths out) and a long one keeps many (its tail is not bowed).
"""

from __future__ import annotations

import numpy as np

# Arc-length below which contour detail is faired away, in pixels.
DEFAULT_WAVELENGTH_PX = 100.0

# Minimum harmonics kept, so a tiny contour still has a defined shape.
_MIN_HARMONICS = 6


def fair_closed_contour(
    points: np.ndarray, wavelength_px: float = DEFAULT_WAVELENGTH_PX, n_points: int = 480
) -> np.ndarray:
    """Fair a closed contour, returned as ``n_points`` ordered ``[x, y]``.

    ``points`` is an ordered ``(N, 2)`` closed contour (first and last need not
    coincide). Harmonics are kept down to ``wavelength_px`` of arc length, so the
    fairing is scale-invariant. A non-positive ``wavelength_px`` or a contour too
    short to fair is returned unchanged so callers need no special-casing.
    """
    pts = np.asarray(points, dtype=np.float64)
    if wavelength_px <= 0 or len(pts) < 2 * _MIN_HARMONICS:
        return pts

    # Arc length around the closed curve (back to the first point).
    steps = np.hypot(*np.diff(pts, axis=0, append=pts[:1]).T)
    arc = np.concatenate(([0.0], np.cumsum(steps)))
    perimeter = arc[-1]
    if perimeter == 0:
        return pts

    # Resample uniformly so the harmonics mean the same all the way round.
    sample = np.linspace(0.0, perimeter, n_points, endpoint=False)
    xs = np.interp(sample, arc, np.append(pts[:, 0], pts[0, 0]))
    ys = np.interp(sample, arc, np.append(pts[:, 1], pts[0, 1]))

    # Keep the harmonics whose wavelength is longer than the cut-off, and drop
    # the finer ones; band-limiting the curve makes its curvature smooth.
    harmonics = max(_MIN_HARMONICS, round(perimeter / wavelength_px))
    spectrum = np.fft.fft(xs + 1j * ys)
    order = np.fft.fftfreq(n_points, d=1.0 / n_points)
    spectrum[np.abs(order) > harmonics] = 0.0
    faired = np.fft.ifft(spectrum)
    return np.column_stack([faired.real, faired.imag])
