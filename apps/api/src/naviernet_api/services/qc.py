"""Preprocessing QC data for the browser's interactive charts.

The same three checks the pipeline's matplotlib QC figure draws (growth
kinematics, interface evolution, signed distance field), computed from the
preprocessed tensors with the same arithmetic as `naviernet.viz.qc` — but
returned as data so the web app can render them on its own canvases.
"""

from __future__ import annotations

import numpy as np

from naviernet.utils.logging import get_logger
from naviernet_api.services.datasets import tensors_meta, tensors_path
from naviernet_api.settings import Settings

log = get_logger(__name__)

_CONTOUR_FRAME_STRIDE = 2  # every 2nd frame, matching the pipeline's QC figure
_SDF_MAX_CELLS = 200  # decimate the SDF grid so the payload stays small


def qc_data(settings: Settings, dataset: str) -> dict | None:
    """Kinematics + interface contours + mid-frame SDF, or None if unprocessed."""
    path = tensors_path(settings, dataset)
    if path is None:
        return None
    meta = tensors_meta(settings, dataset)
    if not meta:
        return None

    with np.load(path) as data:
        alpha = data["alpha"]
        sdf = data["sdf"]
        xs = data["x_star"]
        ys = data["y_star"]
        ts = data["t_star"]

    t_ms = ts * float(meta["t_ref_ms"])
    um_per_px = float(meta["um_per_px"])
    n_event = int(meta.get("n_frames_event", len(ts)))

    # Bubble length per frame: streamwise mask extent (same as viz.qc).
    lengths_um = np.array(
        [np.ptp(np.nonzero((frame > 0.5).any(axis=0))[0]) * um_per_px for frame in alpha]
    )
    fit = np.polyfit(t_ms[:n_event], lengths_um[:n_event], 1)

    return {
        "dataset": dataset,
        "n_frames_event": n_event,
        "kinematics": {
            "t_ms": _round(t_ms),
            "length_um": _round(lengths_um, 1),
            "fit_slope_mm_s": round(float(fit[0]), 1),
            "fit_intercept_um": round(float(fit[1]), 1),
        },
        "interface": {
            "x_pin_star": float(meta.get("x_pin_star", 0.0)),
            "x_range": [float(xs[0]), float(xs[-1])],
            "y_range": [float(ys[0]), float(ys[-1])],
            "frames": [
                {
                    "index": i,
                    "t_ms": round(float(t_ms[i]), 2),
                    "contours": _contours(xs, ys, alpha[i]),
                }
                for i in range(0, len(ts), _CONTOUR_FRAME_STRIDE)
            ],
        },
        "sdf": _sdf_payload(sdf, xs, ys, t_ms),
    }


def _round(values: np.ndarray, digits: int = 3) -> list[float]:
    return [round(float(v), digits) for v in values]


def _contours(xs: np.ndarray, ys: np.ndarray, field: np.ndarray) -> list[list[list[float]]]:
    """The α = 0.5 interface polylines, rounded [x*, y*] pairs."""
    from contourpy import contour_generator

    generator = contour_generator(x=xs, y=ys, z=field)
    return [
        [[round(float(x), 4), round(float(y), 4)] for x, y in line]
        for line in generator.lines(0.5)
    ]


def _sdf_payload(sdf: np.ndarray, xs: np.ndarray, ys: np.ndarray, t_ms: np.ndarray) -> dict:
    """The mid-frame SDF, decimated to a browser-friendly grid."""
    mid = len(t_ms) // 2
    field = sdf[mid]
    stride = max(1, int(np.ceil(max(field.shape) / _SDF_MAX_CELLS)))
    decimated = field[::stride, ::stride]
    return {
        "frame_index": mid,
        "t_ms": round(float(t_ms[mid]), 2),
        "x_range": [float(xs[0]), float(xs[-1])],
        "y_range": [float(ys[0]), float(ys[-1])],
        "values": [_round(row, 3) for row in decimated],
    }
