"""Raw TIFF frames -> calibrated masks -> training tensors.

The pipeline, in order:

1. **Wall detection.** The two channel walls are the darkest horizontal lines in
   the search band. Their known physical separation calibrates um/px directly
   from the images -- no external scale bar, and the calibration is re-derived
   for every dataset rather than carried over.

2. **Segmentation.** Threshold the dark structure, then open with a large
   kernel (erasing the thin heater traces that would otherwise merge with the
   bubble) and close to seal gaps in the bubble's dark ring. The largest
   connected component is the bubble's meniscus band. That band is a thick dark
   rim, not a line -- its outer edge over-reads the vapour and its inner edge
   under-reads it -- so the interface is taken midway across the rim, by
   averaging its inner and outer edges point for point. That curve is then
   low-passed along its arc length, replacing the pixel staircase with the smooth
   closed curve a bubble interface is, while still hugging the shape.

3. **Tensor assembly.** Volume fraction, signed distance (negative inside the
   vapour), and a validity mask. The x axis is flipped so that downstream is
   ``+x``, since the raw camera sees the flow running right to left.

4. **Quality control.** A figure covering growth kinematics, interface
   evolution, and an example signed-distance field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np
import scipy.signal as ss
from PIL import Image

from naviernet.data.contour import smooth_closed_contour
from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths

log = get_logger(__name__)

# Below this the time axis degenerates: the trajectory span runs t[0]..t[-2],
# and a continuous reconstruction of one instant is not a reconstruction.
MIN_USABLE_FRAMES = 3


@dataclass(frozen=True)
class Calibration:
    """Channel geometry recovered from the raw frames."""

    wall_top: int  # image row of the upper channel wall
    wall_bottom: int  # image row of the lower channel wall
    um_per_px: float
    roi: tuple[int, int]  # row band retained after trimming inside the walls


def detect_walls(cfg, paths: RunPaths) -> Calibration:
    """Locate the channel walls in frame 1 and derive the um/px calibration."""
    grey = np.asarray(Image.open(paths.raw_frame(1)).convert("L"), dtype=float)

    r0, r1 = cfg.imaging.wall_search_rows
    row_mean = grey.mean(axis=1)
    peaks, _ = ss.find_peaks(-row_mean[r0:r1], prominence=10)
    peaks = peaks + r0
    if len(peaks) < 2:
        raise RuntimeError(
            f"expected two channel walls in rows {r0}-{r1}, found {len(peaks)}. "
            f"Adjust imaging.wall_search_rows."
        )

    darkest_two = np.sort(peaks[np.argsort(row_mean[peaks])[:2]])
    top, bottom = int(darkest_two[0]), int(darkest_two[1])
    um_per_px = cfg.experiment.channel_width_um / (bottom - top)

    y0 = top + cfg.imaging.wall_margin_top
    y1 = bottom - cfg.imaging.wall_margin_bottom
    log.info(
        "walls at rows %d/%d -> %.3f um/px, ROI rows %d-%d",
        top,
        bottom,
        um_per_px,
        y0,
        y1,
    )
    return Calibration(top, bottom, float(um_per_px), (y0, y1))


def _fill_holes(component: np.ndarray) -> np.ndarray:
    """Fill enclosed holes by flood-filling the background from a padded corner."""
    padded = np.pad(component, 1)
    flooded = padded.copy()
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flooded, mask, (0, 0), 1)
    return (padded | (1 - flooded))[1:-1, 1:-1]


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    """The biggest connected component of a binary mask, or None if it is empty."""
    n_components, labels = cv2.connectedComponents(mask)
    if n_components <= 1:
        return None
    areas = [int((labels == i).sum()) for i in range(1, n_components)]
    return (labels == 1 + int(np.argmax(areas))).astype(np.uint8)


# Samples per rim edge when averaging the two edges into the centreline.
_MIDLINE_POINTS = 400


def _outer_contour(mask: np.ndarray) -> np.ndarray:
    """Largest external contour of a mask as ordered float ``[x, y]`` points."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea).squeeze(1).astype(np.float64)


def _resample_closed(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed contour to ``n`` points, uniform in arc length."""
    steps = np.hypot(*np.diff(points, axis=0, append=points[:1]).T)
    arc = np.concatenate(([0.0], np.cumsum(steps)))
    sample = np.linspace(0.0, arc[-1], n, endpoint=False)
    xs = np.interp(sample, arc, np.append(points[:, 0], points[0, 0]))
    ys = np.interp(sample, arc, np.append(points[:, 1], points[0, 1]))
    return np.column_stack([xs, ys])


def _meniscus_midline(band: np.ndarray, min_hole_fraction: float) -> np.ndarray:
    """The interface contour: the curve midway between the rim's two edges.

    The imaged edge is a thick dark rim, so its outer contour over-reads the
    vapour and its inner contour under-reads it. The two edges are resampled to a
    common length, anchored at the downstream nose, and averaged point for point
    -- literally the curve halfway across the rim. (A distance-transform ridge
    would give the same centre but spikes where the rim's edges converge at the
    nose; averaging the edges stays smooth there.)

    Falls back to the outer edge when there is no enclosed interior to bound the
    rim: a near-solid nucleus, or a bubble cut open by the field-of-view edge.
    """
    filled = _fill_holes(band)
    hole = (filled & (1 - band)).astype(np.uint8)
    cut_by_fov = bool(band[:, 0].any() or band[:, -1].any())
    if int(hole.sum()) < min_hole_fraction * int(filled.sum()) or cut_by_fov:
        return _outer_contour(filled)

    outer = _resample_closed(_outer_contour(filled), _MIDLINE_POINTS)
    inner = _resample_closed(_outer_contour(hole), _MIDLINE_POINTS)
    # Anchor both at the downstream nose (least x) and traverse the same way, so
    # index i pairs an outer point with the inner point across the rim from it.
    outer = np.roll(outer, -int(np.argmin(outer[:, 0])), axis=0)
    inner = np.roll(inner, -int(np.argmin(inner[:, 0])), axis=0)
    if np.dot(outer[1] - outer[0], inner[1] - inner[0]) < 0:
        inner = np.roll(inner[::-1], 1, axis=0)
    return (outer + inner) / 2.0


def segment_frame(cfg, paths: RunPaths, n: int, roi: tuple[int, int]) -> np.ndarray:
    """Binary bubble mask for raw frame ``n`` (1-based), cropped to the ROI.

    The mask is filled to the meniscus centreline, so its boundary is the
    interface itself rather than the outer edge of the dark rim.
    """
    imaging = cfg.imaging
    y0, y1 = roi
    grey = np.asarray(Image.open(paths.raw_frame(n)).convert("L"))
    dark = (grey[y0:y1, :] < imaging.dark_thresh).astype(np.uint8)

    ellipse = cv2.MORPH_ELLIPSE
    k_open = cv2.getStructuringElement(ellipse, (imaging.open_kernel,) * 2)
    k_close = cv2.getStructuringElement(ellipse, (imaging.close_kernel,) * 2)

    thick = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k_open)  # erase thin traces
    ring = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, k_close)  # seal ring gaps
    band = _largest_component(ring)
    if band is None:
        raise RuntimeError(
            f"no bubble found in frame {n}; check imaging.dark_thresh "
            f"(currently {imaging.dark_thresh})"
        )

    interface = _meniscus_midline(band, imaging.min_rim_hole_fraction)
    return _rasterise(interface, dark.shape, imaging.contour_smooth_px)


def _rasterise(interface: np.ndarray, shape: tuple[int, int], sigma_px: float) -> np.ndarray:
    """Smooth a closed interface contour and fill it into a binary bubble mask."""
    if sigma_px > 0:
        interface = smooth_closed_contour(interface, sigma_px, n_points=720)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(interface).astype(np.int32)], 1)
    return mask


def usable_frame_numbers(cfg) -> list[int]:
    """The 1-based camera frames that become tensor rows, in order.

    The usable window (``1..n_frames_usable``) minus the frames excluded for
    this series. Row ``i`` of every tensor is camera frame ``result[i]``; the
    mapping is written into the archive's meta so downstream stages resolve
    frame numbers instead of assuming a contiguous run.
    """
    excluded = {int(n) for n in cfg.experiment.excluded_frames}
    kept = [n for n in range(1, int(cfg.experiment.n_frames_usable) + 1) if n not in excluded]
    if len(kept) < MIN_USABLE_FRAMES:
        raise ValueError(
            f"only {len(kept)} usable frame(s) left after excluding "
            f"{sorted(excluded)}; at least {MIN_USABLE_FRAMES} are needed"
        )
    return kept


def preprocess(cfg, paths: RunPaths) -> dict:
    """Run the full preprocessing pipeline and write the tensor archive."""
    paths.ensure()
    calibration = detect_walls(cfg, paths)
    um_per_px = calibration.um_per_px
    frame_numbers = usable_frame_numbers(cfg)
    n_usable = len(frame_numbers)

    masks = np.stack([segment_frame(cfg, paths, n, calibration.roi) for n in frame_numbers])
    # Flip x so downstream is +x; the raw camera sees flow right to left.
    alpha = masks.astype(np.float32)[:, :, ::-1].copy()

    l_ref = cfg.scales.L_ref_um
    sdf = np.zeros_like(alpha)
    for i in range(n_usable):
        binary = (alpha[i] > 0.5).astype(np.uint8)
        outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
        inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        sdf[i] = (outside - inside) * um_per_px / l_ref  # negative inside vapour

    # Mask the field-of-view cut on the final usable frame -- unless that frame
    # was excluded, in which case there is no truncated row to mask.
    valid = np.ones_like(alpha, dtype=np.uint8)
    last_usable = int(cfg.experiment.n_frames_usable)
    if cfg.imaging.truncated_cols > 0 and last_usable in frame_numbers:
        truncated_row = frame_numbers.index(last_usable)
        valid[truncated_row, :, -cfg.imaging.truncated_cols :] = 0

    height_px, width_px = alpha.shape[1:]
    from naviernet.physics.groups import reference_time_ms

    t_ref_ms = reference_time_ms(cfg.scales)
    x_star = (np.arange(width_px) + 0.5) * um_per_px / l_ref
    y_star = (np.arange(height_px) + 0.5) * um_per_px / l_ref
    # Real acquisition times: an excluded frame leaves a gap on the time axis
    # rather than pulling every later frame earlier.
    t_star = (np.asarray(frame_numbers) - 1) * cfg.experiment.dt_frame_ms / t_ref_ms

    # Rows (not camera frames) belonging to the growth event -- what downstream
    # stages index with.
    n_event = sum(1 for n in frame_numbers if n <= int(cfg.experiment.n_frames_event))

    # The nucleation cavity pins the bubble's upstream end: in flipped
    # coordinates that is the (near-stationary) right edge of the raw mask.
    right_ends = [np.nonzero(m.any(axis=0))[0].max() for m in masks[:n_event]]
    x_pin = (width_px - float(np.median(right_ends))) * um_per_px / l_ref

    meta = {
        "dataset": cfg.dataset,
        "um_per_px": um_per_px,
        "wall_rows": [calibration.wall_top, calibration.wall_bottom],
        "y_roi": list(calibration.roi),
        "L_ref_um": l_ref,
        "U_ref": cfg.scales.U_ref,
        "t_ref_ms": t_ref_ms,
        "x_pin_star": x_pin,
        "n_frames_usable": n_usable,
        "n_frames_event": n_event,
        # Interface-smoothing scale, so the QC overlay smooths its rings to match.
        "contour_smooth_px": float(cfg.imaging.contour_smooth_px),
        # Row -> camera frame. Downstream reads this rather than assuming
        # row i is frame i+1, which stops holding once frames are excluded.
        "frame_numbers": frame_numbers,
        "excluded_frames": sorted({int(n) for n in cfg.experiment.excluded_frames}),
        "frames_used": _frames_used(cfg, frame_numbers),
        "x_convention": "x* runs downstream; raw camera flow is right to left",
    }

    np.savez_compressed(
        paths.tensors,
        alpha=alpha,
        sdf=sdf,
        valid=valid,
        x_star=x_star.astype(np.float32),
        y_star=y_star.astype(np.float32),
        t_star=t_star.astype(np.float32),
        masks_camera=masks,
        meta=json.dumps(meta),
    )
    log.info("wrote %s  alpha%s", paths.tensors, alpha.shape)

    from naviernet.viz.qc import qc_figure

    qc_figure(cfg, paths, alpha, sdf, x_star, y_star, t_star, um_per_px, x_pin, n_event)
    return meta


def _frames_used(cfg, frame_numbers: list[int]) -> str:
    """Human-readable provenance for the archive's meta record."""
    window = f"1-{cfg.experiment.n_frames_usable} of {cfg.experiment.n_frames_raw}"
    dropped = sorted(
        set(range(1, int(cfg.experiment.n_frames_usable) + 1)) - set(frame_numbers)
    )
    return window if not dropped else f"{window}, excluding {dropped}"
