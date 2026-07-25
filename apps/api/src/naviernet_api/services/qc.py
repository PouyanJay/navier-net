"""Preprocessing QC data for the browser's interactive charts.

The same three checks the pipeline's matplotlib QC figure draws (growth
kinematics, interface evolution, signed distance field), computed from the
preprocessed tensors with the same arithmetic as `naviernet.viz.qc`; but
returned as data so the web app can render them on its own canvases.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from naviernet.data.contour import DEFAULT_SMOOTH_PX, smooth_closed_contour
from naviernet.utils.logging import get_logger
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
    l_ref_um: float
    x_pin_star: float
    n_event: int
    # Tensor-row → 1-based camera frame, so a silhouette can be laid on its
    # raw frame image, and the top ROI row that the rings' y* = 0 sits at.
    frame_numbers: list[int]
    y_roi_top: int
    contour_smooth_px: float  # interface-smoothing scale used at preprocess time


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
        frame_numbers = [int(n) for n in meta.get("frame_numbers", range(1, len(ts) + 1))]
        return _Tensors(
            alpha=data["alpha"],
            sdf=data["sdf"],
            xs=data["x_star"],
            ys=data["y_star"],
            t_ms=ts * float(meta["t_ref_ms"]),
            um_per_px=float(meta["um_per_px"]),
            l_ref_um=float(meta["L_ref_um"]),
            x_pin_star=float(meta.get("x_pin_star", 0.0)),
            n_event=int(meta.get("n_frames_event", len(ts))),
            frame_numbers=frame_numbers,
            y_roi_top=int(meta.get("y_roi", [0, 0])[0]),
            contour_smooth_px=float(meta.get("contour_smooth_px", DEFAULT_SMOOTH_PX)),
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
    # Every frame, not every other: the web app overlays a silhouette on each
    # raw frame in the lightbox, so a gap would leave frames with no boundary.
    # (The matplotlib QC figure still strides; this is the interactive path.)
    # Smoothing runs in x* space, so convert the preprocess pixel scale to match.
    sigma_star = tensors.contour_smooth_px * tensors.um_per_px / tensors.l_ref_um
    return QcInterface(
        x_pin_star=tensors.x_pin_star,
        x_range=[float(tensors.xs[0]), float(tensors.xs[-1])],
        y_range=[float(tensors.ys[0]), float(tensors.ys[-1])],
        l_ref_um=tensors.l_ref_um,
        y_roi_top=tensors.y_roi_top,
        frames=[
            QcInterfaceFrame(
                index=i,
                camera_frame=tensors.frame_numbers[i],
                t_ms=round(float(tensors.t_ms[i]), 2),
                rings=_rings(tensors.xs, tensors.ys, tensors.alpha[i], sigma_star),
            )
            for i in range(len(tensors.t_ms))
        ],
    )


def _rings(
    xs: np.ndarray, ys: np.ndarray, field: np.ndarray, sigma_star: float
) -> list[list[list[float]]]:
    """The α > 0.5 region as closed [x*, y*] rings.

    Filled regions rather than contour lines: the bubble touches the top and
    bottom of the imaged band, so its α = 0.5 *line* is cut there and comes back
    as two open arcs (nose and tail) with the wall-adjacent stretches missing.
    Filling closes the outline along the domain edge, which both completes the
    silhouette and collapses those arcs back into one ring per bubble.
    """
    from contourpy import FillType, contour_generator

    generator = contour_generator(x=xs, y=ys, z=field, fill_type=FillType.OuterOffset)
    points, offsets = generator.filled(0.5, 1.5)
    rings: list[list[list[float]]] = []
    for polygon, boundaries in zip(points, offsets, strict=True):
        # A polygon is an outer ring followed by any holes; each is emitted
        # separately and drawn with an even-odd fill, so holes stay holes.
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            # Low-pass the traced ring so the overlay reads as the smooth curve a
            # bubble interface is, not the mask's pixel staircase; close the loop
            # so the ring is a proper polygon (first point repeated at the end).
            ring = smooth_closed_contour(polygon[start:end], sigma_star, n_points=480)
            closed = np.vstack([ring, ring[:1]])
            rings.append(np.round(closed, 4).tolist())
    return rings


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
