"""Preprocessing QC data for the browser's interactive charts.

The same three checks the pipeline's matplotlib QC figure draws (growth
kinematics, interface evolution, signed distance field), computed from the
preprocessed tensors with the same arithmetic as `naviernet.viz.qc` — but
returned as data so the web app can render them on its own canvases.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from naviernet.utils.logging import get_logger
from naviernet.viz.qc import CONTOUR_FRAME_STRIDE
from naviernet_api.models import QcData, QcInterface, QcInterfaceFrame, QcKinematics, QcSdf
from naviernet_api.services.datasets import tensors_meta, tensors_path
from naviernet_api.settings import Settings

log = get_logger(__name__)

_SDF_MAX_CELLS = 200  # decimate the SDF grid so the payload stays small


class _Tensors(NamedTuple):
    """The preprocessed arrays the three QC checks are computed from."""

    alpha: np.ndarray
    sdf: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    t_ms: np.ndarray
    um_per_px: float
    x_pin_star: float
    n_event: int


def qc_data(settings: Settings, dataset: str) -> QcData | None:
    """Kinematics + interface contours + mid-frame SDF, or None if unprocessed."""
    tensors = _load(settings, dataset)
    if tensors is None:
        return None
    return QcData(
        dataset=dataset,
        n_frames_event=tensors.n_event,
        kinematics=_kinematics_payload(tensors),
        interface=_interface_payload(tensors),
        sdf=_sdf_payload(tensors),
    )


def _load(settings: Settings, dataset: str) -> _Tensors | None:
    path = tensors_path(settings, dataset)
    if path is None:
        return None
    meta = tensors_meta(settings, dataset)
    if not meta:
        return None
    with np.load(path) as data:
        ts = data["t_star"]
        return _Tensors(
            alpha=data["alpha"],
            sdf=data["sdf"],
            xs=data["x_star"],
            ys=data["y_star"],
            t_ms=ts * float(meta["t_ref_ms"]),
            um_per_px=float(meta["um_per_px"]),
            x_pin_star=float(meta.get("x_pin_star", 0.0)),
            n_event=int(meta.get("n_frames_event", len(ts))),
        )


def _kinematics_payload(tensors: _Tensors) -> QcKinematics:
    """Bubble length per frame (streamwise mask extent, same as viz.qc) + fit."""
    lengths_um = np.array(
        [
            np.ptp(np.nonzero((frame > 0.5).any(axis=0))[0]) * tensors.um_per_px
            for frame in tensors.alpha
        ]
    )
    fit = np.polyfit(tensors.t_ms[: tensors.n_event], lengths_um[: tensors.n_event], 1)
    return QcKinematics(
        t_ms=np.round(tensors.t_ms, 3).tolist(),
        length_um=np.round(lengths_um, 1).tolist(),
        fit_slope_mm_s=round(float(fit[0]), 1),
        fit_intercept_um=round(float(fit[1]), 1),
    )


def _interface_payload(tensors: _Tensors) -> QcInterface:
    return QcInterface(
        x_pin_star=tensors.x_pin_star,
        x_range=[float(tensors.xs[0]), float(tensors.xs[-1])],
        y_range=[float(tensors.ys[0]), float(tensors.ys[-1])],
        frames=[
            QcInterfaceFrame(
                index=i,
                t_ms=round(float(tensors.t_ms[i]), 2),
                contours=_contours(tensors.xs, tensors.ys, tensors.alpha[i]),
            )
            for i in range(0, len(tensors.t_ms), CONTOUR_FRAME_STRIDE)
        ],
    )


def _contours(xs: np.ndarray, ys: np.ndarray, field: np.ndarray) -> list[list[list[float]]]:
    """The α = 0.5 interface polylines, rounded [x*, y*] pairs."""
    from contourpy import contour_generator

    generator = contour_generator(x=xs, y=ys, z=field)
    return [np.round(line, 4).tolist() for line in generator.lines(0.5)]


def _sdf_payload(tensors: _Tensors) -> QcSdf:
    """The mid-frame SDF, decimated to a browser-friendly grid."""
    mid = len(tensors.t_ms) // 2
    field = tensors.sdf[mid]
    stride = max(1, int(np.ceil(max(field.shape) / _SDF_MAX_CELLS)))
    return QcSdf(
        frame_index=mid,
        t_ms=round(float(tensors.t_ms[mid]), 2),
        x_range=[float(tensors.xs[0]), float(tensors.xs[-1])],
        y_range=[float(tensors.ys[0]), float(tensors.ys[-1])],
        values=np.round(field[::stride, ::stride], 3).tolist(),
    )
