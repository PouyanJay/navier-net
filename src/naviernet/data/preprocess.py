"""Raw TIFF frames -> calibrated masks -> training tensors.

The pipeline, in order:

1. **Wall detection.** The two channel walls are the darkest horizontal lines in
   the search band. Their known physical separation calibrates um/px directly
   from the images -- no external scale bar, and the calibration is re-derived
   for every dataset rather than carried over.

2. **Segmentation.** Threshold the dark structure, then open with a large
   kernel (erasing the thin heater traces that would otherwise merge with the
   bubble) and close to seal gaps in the bubble's dark ring. The largest
   connected component is the bubble's meniscus band -- a thick dark rim, not a
   line. Rather than extract a centreline from that discrete band (which fights
   the pixel grid and kinks), an **active contour** is evolved onto it: a closed
   curve seeded near the rim centre that settles onto the centre-line while a
   rigidity term keeps it smooth. Smoothness is intrinsic to the model -- the
   curve cannot bend sharply -- which is also physical, rigidity standing in for
   the surface tension that keeps real bubbles smooth. A final scale-invariant
   fairing band-limits the curve so its curvature varies smoothly too.

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
from scipy.ndimage import gaussian_filter, map_coordinates

from naviernet.data.contour import DEFAULT_WAVELENGTH_PX, fair_closed_contour
from naviernet.physics.groups import compute_groups
from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths

# The condition fields baked into the preprocessed tensors: the frame interval
# sets the time axis, the channel width sets um/px, and the reference velocity
# sets the reference time. This is the single source of truth for "what a
# re-preprocess is needed for" -- the API imports it (see datasets service) so
# the two never drift. Everything else only feeds the dimensionless groups.
BAKED_CONDITION_FIELDS: tuple[str, ...] = ("dt_frame_ms", "channel_width_um", "U_ref")


def baked_conditions(cfg) -> dict[str, float]:
    """The values of the tensor-baked condition fields for a composed config."""
    return {
        "dt_frame_ms": cfg.experiment.dt_frame_ms,
        "channel_width_um": cfg.experiment.channel_width_um,
        "U_ref": cfg.scales.U_ref,
    }


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


# Points on the stored interface curve (rasterised into alpha, drawn on the QC
# overlay). Enough to draw as a smooth polyline without a heavy payload.
_INTERFACE_POINTS = 480


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


# Active-contour parameters. Tension keeps the points evenly spread; rigidity
# is the smoothness (higher ignores local band defects); the field blur smooths
# the target so the curve does not chase them; the rest are the max per-step move
# and the number of points evolved before the curve is resampled for storage.
_SNAKE_TENSION = 0.05
_SNAKE_RIGIDITY = 40.0
_SNAKE_FIELD_BLUR = 8.0
_SNAKE_STEP = 1.5
_SNAKE_ITERATIONS = 400
_SNAKE_POINTS = 200


def _snake_internal_inverse(n: int, tension: float, rigidity: float) -> np.ndarray:
    """Inverse of the closed active contour's internal-energy matrix ``(A + I)``.

    ``A`` is the circulant pentadiagonal from the tension (2nd-difference) and
    rigidity (4th-difference) terms; the ``+ I`` is the implicit time step. One
    inverse drives every iteration, so the evolution is a stable linear solve.
    """
    diag = 2 * tension + 6 * rigidity + 1.0  # + I (unit time step)
    off1 = -(tension + 4 * rigidity)
    off2 = rigidity
    row = np.zeros(n)
    row[0] = diag
    row[1] = row[-1] = off1
    row[2] = row[-2] = off2
    return np.linalg.inv(np.stack([np.roll(row, i) for i in range(n)]))


def _snake_centreline(signed: np.ndarray, init: np.ndarray) -> np.ndarray:
    """Evolve a closed active contour onto the meniscus centreline.

    ``signed`` is ``distance-to-outer-edge − distance-to-inner-edge``: zero
    exactly midway across the rim, negative toward the outer edge, positive
    toward the interior. It is blurred first so the curve settles on the rim's
    overall centre rather than chasing local band defects (a notch where a
    microchannel grazes the rim). The external force is a Newton step toward the
    zero level-set -- proportional to the distance still to go, so it shrinks to
    nothing as the curve arrives and never oscillates -- and the internal energy
    smooths the curve each step, so it can neither overshoot nor kink.
    """
    field = gaussian_filter(signed, _SNAKE_FIELD_BLUR)
    grad_y, grad_x = np.gradient(field)
    inverse = _snake_internal_inverse(len(init), _SNAKE_TENSION, _SNAKE_RIGIDITY)
    x, y = init[:, 0].copy(), init[:, 1].copy()
    for _ in range(_SNAKE_ITERATIONS):
        here = map_coordinates(field, [y, x], order=1, mode="nearest")
        gx = map_coordinates(grad_x, [y, x], order=1, mode="nearest")
        gy = map_coordinates(grad_y, [y, x], order=1, mode="nearest")
        # Newton step h·∇/|∇|² lands on the zero level-set; clip so a flat spot
        # (tiny gradient) cannot fling a point away.
        step = np.clip(here / (gx * gx + gy * gy + 1e-6), -_SNAKE_STEP, _SNAKE_STEP)
        x = inverse @ (x - step * gx)
        y = inverse @ (y - step * gy)
    return np.column_stack([x, y])


def _seal_fov_cut(band: np.ndarray) -> np.ndarray:
    """Close the rim across a field-of-view edge the bubble runs off.

    The bubble is capped at the image edge -- all that is seen of it -- so its
    off-frame lobe's interior becomes enclosed and the centreline extraction can
    treat the whole visible bubble like any other.
    """
    sealed = band.copy()
    for col in (0, -1):
        rows = np.nonzero(band[:, col])[0]
        if rows.size:
            sealed[rows.min() : rows.max() + 1, col] = 1
    return sealed


def _meniscus_interface(
    band: np.ndarray, min_hole_fraction: float, seed_blur_px: float
) -> np.ndarray:
    """The interface: a fair closed curve on the meniscus rim's centreline.

    The rim centre is the zero level-set of a medial field (distance to the outer
    edge minus distance to the inner edge). An active contour, seeded from the
    blurred level-set, settles onto it and stays smooth by its own rigidity --
    where a discrete centreline would kink or self-intersect. A final fairing
    band-limits the curve so its curvature varies smoothly, with no sudden jumps.

    A bubble cut by the field-of-view edge (a pinching bubble at critical
    tension) is first sealed across the cut so its off-frame lobe's interior is
    enclosed and the same centreline snake can run -- reconstructing the
    interface across the truncation rather than tracing only the enclosed lobe.

    Falls back to the whole outer outline only when there is no enclosed interior
    at all to centre in: a near-solid nucleus.
    """
    if band[:, 0].any() or band[:, -1].any():
        band = _seal_fov_cut(band)
    filled = _fill_holes(band)
    hole = (filled & (1 - band)).astype(np.uint8)
    if int(hole.sum()) < min_hole_fraction * int(filled.sum()):
        return fair_closed_contour(
            _outer_contour(filled), DEFAULT_WAVELENGTH_PX, _INTERFACE_POINTS
        )

    to_outer = cv2.distanceTransform(filled, cv2.DIST_L2, 5)  # 0 at the outer edge
    to_inner = cv2.distanceTransform(1 - hole, cv2.DIST_L2, 5)  # 0 at the inner edge
    signed = (to_outer - to_inner).astype(np.float32)

    seed_field = cv2.GaussianBlur(signed, (0, 0), seed_blur_px) if seed_blur_px > 0 else signed
    seed = _largest_component((seed_field >= 0).astype(np.uint8))
    if seed is None:  # no interior half survived the blur; take the outer edge
        return fair_closed_contour(
            _outer_contour(filled), DEFAULT_WAVELENGTH_PX, _INTERFACE_POINTS
        )
    init = _resample_closed(_outer_contour(_fill_holes(seed)), _SNAKE_POINTS)
    return fair_closed_contour(
        _snake_centreline(signed, init), DEFAULT_WAVELENGTH_PX, _INTERFACE_POINTS
    )


def segment_frame(cfg, paths: RunPaths, n: int, roi: tuple[int, int]) -> np.ndarray:
    """Smooth closed interface contour for raw frame ``n`` (1-based), in ROI
    pixel coordinates.

    The contour is an active contour settled onto the meniscus centreline -- a
    smooth closed curve. The same curve is rasterised into the training tensors
    and, converted to ``x*``, drawn on the QC overlay -- one curve, so the model
    and the picture never disagree.
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

    return _meniscus_interface(band, imaging.min_rim_hole_fraction, imaging.contour_smooth_px)


def _fill_interface(interface: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a closed interface contour into a binary bubble mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(interface).astype(np.int32)], 1)
    return mask


def _interface_to_star(
    interface: np.ndarray, width_px: int, um_per_px: float, l_ref: float
) -> np.ndarray:
    """Interface polygon (ROI pixels) → non-dimensional ``[x*, y*]``.

    Undoes the x-flip and the ROI offset exactly as the tensors' ``x_star`` /
    ``y_star`` axes do, so the QC overlay draws the interface where alpha carries
    it. ``x* = (W - 0.5 - col)·µm/px / L_ref``; ``y* = (row + 0.5)·µm/px / L_ref``.
    """
    xs = (width_px - 0.5 - interface[:, 0]) * um_per_px / l_ref
    ys = (interface[:, 1] + 0.5) * um_per_px / l_ref
    return np.column_stack([xs, ys])


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

    interfaces = [segment_frame(cfg, paths, n, calibration.roi) for n in frame_numbers]
    width_px = np.asarray(Image.open(paths.raw_frame(frame_numbers[0])).convert("L")).shape[1]
    roi_shape = (calibration.roi[1] - calibration.roi[0], width_px)
    masks = np.stack([_fill_interface(curve, roi_shape) for curve in interfaces])
    # Flip x so downstream is +x; the raw camera sees flow right to left.
    alpha = masks.astype(np.float32)[:, :, ::-1].copy()

    l_ref = cfg.scales.L_ref_um
    # The same interface curves in x* coordinates, for the QC overlay to draw
    # directly -- no re-tracing of the alpha raster, which would re-quantise the
    # smooth curve back onto the pixel grid.
    interface_star = np.stack(
        [_interface_to_star(curve, width_px, um_per_px, l_ref) for curve in interfaces]
    ).astype(np.float32)
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

    height_px = alpha.shape[1]
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
        # Row -> camera frame. Downstream reads this rather than assuming
        # row i is frame i+1, which stops holding once frames are excluded.
        "frame_numbers": frame_numbers,
        "excluded_frames": sorted({int(n) for n in cfg.experiment.excluded_frames}),
        # The conditions baked into these tensors (time axis, um/px, reference
        # time). The API compares these to the series' current conditions to know
        # when a baked-condition edit needs a re-preprocess.
        "baked_conditions": baked_conditions(cfg),
        "frames_used": _frames_used(cfg, frame_numbers),
        "x_convention": "x* runs downstream; raw camera flow is right to left",
        # The dataset's dimensionless groups, so joint (transfer-learning)
        # training reads each series' regime and conditioning vector straight
        # from its tensors — no per-dataset Hydra recomposition at train time.
        "groups": compute_groups(cfg),
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
        interface_star=interface_star,
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
