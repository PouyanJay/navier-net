"""Geometry: the interface contour at any instant, the camera-pixel transform,
the raw frames, and how well the model's outline agrees with the images."""

from __future__ import annotations

import io
import re
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from contourpy import contour_generator
from PIL import Image

from naviernet.evaluation import frame_iou, front_position, predict_alpha_fullres
from naviernet_studio import cleaning
from naviernet_studio.scene import Scene

CONTOUR_NX = 721
CONTOUR_NY = 241
MIN_CONTOUR_POINTS = 8
# Rows of raw frame to keep above and below the detected walls when cropping.
CROP_MARGIN_PX = 24
INTERFACE_LEVEL = 0.5
RAW_SAMPLE_FRAME = 1

_contour_cache: dict[tuple[str, float, float], dict] = {}
_contour_lock = threading.Lock()
CONTOUR_CACHE_SIZE = 256


def _phi_or_alpha(scene: Scene, points: torch.Tensor) -> np.ndarray:
    """A level-set field whose zero is the interface."""
    model = scene.model
    with scene.lock, torch.no_grad():
        if getattr(model, "front_geometry", False):
            value = model._phi_with_frozen_geometry(points, None)
        else:
            value = model.alpha(points) - INTERFACE_LEVEL
    return value.cpu().numpy().ravel()


def interface(scene: Scene, t_star: float) -> dict:
    """The closed interface contour(s) at ``t_star`` in µm, with the nose."""
    key = (scene.run_id, scene.checkpoint_mtime, round(t_star, 6))
    with _contour_lock:
        hit = _contour_cache.get(key)
    if hit is not None:
        return hit
    data, scales = scene.data, scene.scales
    xs = np.linspace(data.domain.x_min, data.domain.x_max, CONTOUR_NX)
    ys = np.linspace(data.y_lo, data.y_hi, CONTOUR_NY)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = torch.tensor(
        np.column_stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, t_star)]),
        dtype=torch.float32,
    )
    field = _phi_or_alpha(scene, points).reshape(grid_x.shape)
    lines = contour_generator(x=xs * scales.length_um, y=ys * scales.length_um, z=field).lines(
        0.0
    )
    rings = [np.asarray(line) for line in lines if len(line) >= MIN_CONTOUR_POINTS]
    rings.sort(key=lambda r: -len(r))
    nose = float(max((r[:, 0].max() for r in rings), default=float("nan")))
    root = float(min((r[:, 0].min() for r in rings), default=float("nan")))
    payload = {
        "t_star": t_star,
        "t_ms": t_star * scales.time_ms,
        "rings_um": [np.round(r, 2).tolist() for r in rings],
        "nose_um": nose,
        "root_um": root,
        "vapour_area_mm2": float((field > 0).mean()) * _span_area(scene) * scales.area_mm2,
    }
    with _contour_lock:
        _contour_cache[key] = payload
        while len(_contour_cache) > CONTOUR_CACHE_SIZE:
            del _contour_cache[next(iter(_contour_cache))]
    return payload


def _span_area(scene: Scene) -> float:
    data = scene.data
    return float(data.domain.x_max - data.domain.x_min) * float(data.y_hi - data.y_lo)


def loop_arclength(ring: np.ndarray, y_center_um: float) -> tuple[np.ndarray, np.ndarray]:
    """The ring re-rooted at its root apex (minimum x) and oriented so it runs
    along the upper edge first; returns ``(vertices, cumulative arclength)``."""
    closed = ring if np.allclose(ring[0], ring[-1]) else np.vstack([ring, ring[:1]])
    open_ring = closed[:-1]
    start = int(np.argmin(open_ring[:, 0]))
    rolled = np.roll(open_ring, -start, axis=0)
    ahead = rolled[: max(2, len(rolled) // 8)]
    if ahead[:, 1].mean() < y_center_um:
        rolled = np.vstack([rolled[:1], rolled[1:][::-1]])
    steps = np.linalg.norm(np.diff(rolled, axis=0), axis=1)
    return rolled, np.concatenate([[0.0], np.cumsum(steps)])


def project_arclength(
    points_um: np.ndarray, ring: np.ndarray, y_center_um: float
) -> np.ndarray:
    """Arclength of the nearest ring vertex for each point."""
    vertices, s = loop_arclength(ring, y_center_um)
    d = ((points_um[:, None, :] - vertices[None, :, :]) ** 2).sum(-1)
    return s[np.argmin(d, axis=1)]


@lru_cache(maxsize=8)
def _raw_image(path: str) -> Image.Image:
    return Image.open(path).convert("L")


def transform(scene: Scene) -> dict:
    """Everything needed to place model coordinates on the raw camera frame."""
    meta, scales, data = scene.data.meta, scene.scales, scene.data
    sample = scene.paths.raw_frame(RAW_SAMPLE_FRAME)
    width, height = _raw_image(str(sample)).size if sample.is_file() else (None, None)
    um_per_px = float(meta["um_per_px"])
    y_roi = [int(v) for v in meta["y_roi"]]
    wall_rows = [int(v) for v in meta.get("wall_rows", y_roi)]
    return {
        "frame_px": [width, height],
        "um_per_px": um_per_px,
        "L_ref_um": scales.length_um,
        "y_roi_rows": y_roi,
        "wall_rows": wall_rows,
        "crop_rows": [
            max(0, wall_rows[0] - CROP_MARGIN_PX),
            (wall_rows[1] + CROP_MARGIN_PX)
            if height is None
            else min(height, wall_rows[1] + CROP_MARGIN_PX),
        ],
        "x_flipped": True,
        "to_pixel": {
            "col": "W - 0.5 - x_um / um_per_px",
            "row": "y_um / um_per_px - 0.5 + y_roi_rows[0]",
        },
        "walls_um": [float(data.y_lo) * scales.length_um, float(data.y_hi) * scales.length_um],
        "centreline_um": 0.5 * float(data.y_lo + data.y_hi) * scales.length_um,
        "domain_um": [
            float(data.domain.x_min) * scales.length_um,
            float(data.domain.x_max) * scales.length_um,
            float(data.domain.y_min) * scales.length_um,
            float(data.domain.y_max) * scales.length_um,
        ],
        "x_pin_um": float(data.domain.x_pin) * scales.length_um,
    }


def raw_frame_png(scene: Scene, frame: int, crop: bool) -> bytes | None:
    path = scene.paths.raw_frame(frame)
    if not path.is_file():
        return None
    image = _raw_image(str(path))
    if crop:
        rows = transform(scene)["crop_rows"]
        image = image.crop((0, rows[0], image.size[0], rows[1]))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _mask_rings(
    mask: np.ndarray, xs_um: np.ndarray, ys_um: np.ndarray
) -> list[list[list[float]]]:
    lines = contour_generator(x=xs_um, y=ys_um, z=mask.astype(np.float32)).lines(
        INTERFACE_LEVEL
    )
    return [
        np.round(np.asarray(line), 2).tolist()
        for line in lines
        if len(line) >= MIN_CONTOUR_POINTS
    ]


def agreement(scene: Scene, frame: int) -> dict | None:
    """How the model's outline compares with the measured mask at a camera frame."""
    row = scene.row_of(frame)
    if row is None:
        return None
    cfg, data, scales = scene.cfg, scene.data, scene.scales
    t_star = float(data.t[row])
    xs_um, ys_um = data.x * scales.length_um, data.y * scales.length_um
    with scene.lock:
        predicted = predict_alpha_fullres(scene.model, data, t_star)
    threshold = float(cfg.evaluation.threshold)
    measured = (data.alpha[row] > threshold) & (data.valid[row] > 0)
    model_mask = predicted > threshold
    pixel_mm2 = float(data.x[1] - data.x[0]) * float(data.y[1] - data.y[0]) * scales.area_mm2
    union = (measured | model_mask).sum()
    with scene.lock:
        iou = frame_iou(cfg, scene.model, data, row)
    nose_model = front_position(model_mask, data.x, data.domain.x_pin)
    nose_measured = front_position(measured, data.x, data.domain.x_pin)
    return {
        "frame": frame,
        "t_ms": t_star * scales.time_ms,
        "role": next(f.role for f in scene.frames() if f.frame == frame),
        "iou": iou,
        "iou_fullres": float((measured & model_mask).sum() / max(union, 1)),
        "nose_model_um": nose_model * scales.length_um,
        "nose_measured_um": nose_measured * scales.length_um,
        "nose_error_um": (nose_model - nose_measured) * scales.length_um,
        "area_model_mm2": float(model_mask.sum()) * pixel_mm2,
        "area_measured_mm2": float(measured.sum()) * pixel_mm2,
        "only_model_mm2": float((model_mask & ~measured).sum()) * pixel_mm2,
        "only_measured_mm2": float((measured & ~model_mask).sum()) * pixel_mm2,
        "measured_rings_um": _mask_rings(measured, xs_um, ys_um),
        "model_rings_um": interface(scene, t_star)["rings_um"],
        "valid_fraction": float((data.valid[row] > 0).mean()),
    }


_backgrounds: dict[str, tuple[float, np.ndarray]] = {}
_background_lock = threading.Lock()


def _event_stack(scene: Scene) -> tuple[np.ndarray, np.ndarray] | None:
    """The event frames as grey levels and the bubble masks placed on the same
    full-width pixel grid, or None when a raw frame is missing."""
    data = scene.data
    numbers = data.frame_numbers[: data.n_event]
    paths = [scene.paths.raw_frame(n) for n in numbers]
    if not all(p.is_file() for p in paths):
        return None
    stack = np.stack([np.asarray(Image.open(p).convert("L"), dtype=np.float32) for p in paths])
    bubble = np.zeros(stack.shape, dtype=bool)
    top = int(data.meta["y_roi"][0])
    rows = data.masks_camera.shape[1]
    bubble[:, top : top + rows, :] = data.masks_camera[: data.n_event] > 0
    return stack, bubble


def _background(scene: Scene) -> np.ndarray | None:
    """The static background of the set-up, computed once per checkpoint."""
    with _background_lock:
        cached = _backgrounds.get(scene.run_id)
        if cached is not None and cached[0] == scene.checkpoint_mtime:
            return cached[1]
    event = _event_stack(scene)
    if event is None:
        return None
    background = cleaning.static_background(*event)
    with _background_lock:
        _backgrounds[scene.run_id] = (scene.checkpoint_mtime, background)
    return background


def rendered_frame_png(scene: Scene, frame: int, crop: bool) -> bytes | None:
    """The camera frame with the set-up removed and the bubble kept alone, edged
    by the measured outline, as PNG."""
    row = scene.row_of(frame)
    background = _background(scene)
    path = scene.paths.raw_frame(frame)
    if row is None or background is None or not path.is_file():
        return None
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    bubble = np.zeros(image.shape, dtype=bool)
    top = int(scene.data.meta["y_roi"][0])
    rows = scene.data.masks_camera.shape[1]
    bubble[top : top + rows] = scene.data.masks_camera[row] > 0
    render = cleaning.outline_frame
    out = Image.fromarray(np.clip(render(image, background, bubble), 0, 255).astype(np.uint8))
    if crop:
        rows_crop = transform(scene)["crop_rows"]
        out = out.crop((0, rows_crop[0], out.size[0], rows_crop[1]))
    buffer = io.BytesIO()
    out.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


CLEANED_DIR = Path("apps/studio/cleaned")


def cleaned_frame_png(repo_root: Path, run_id: str, frame: int, crop: bool) -> bytes | None:
    """The generated cleaned frame stored in the repository, as PNG; None when
    it has not been generated yet (see ``apps/studio/tools/generate_cleaned.py``)."""
    if not re.match(r"^[A-Za-z0-9._-]+$", run_id):
        return None
    path = Path(repo_root) / CLEANED_DIR / run_id / f"frame_{frame:02d}.png"
    if not path.is_file():
        return None
    out = Image.open(path).convert("RGB")
    if crop:
        out = out.crop((0, 445, out.size[0], 671))
    buffer = io.BytesIO()
    out.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
