"""Interface reconstruction data for the viewport.

Evaluates the trained model's volume fraction on a strided grid over a set of
continuous timesteps and extracts the alpha = 0.5 interface contours as
polylines (plus the measured contours at the camera instants). This is pure
visualization geometry: the physics all lives in the pipeline; contour
extraction uses contourpy, the same engine matplotlib's own contour plots use.

Loading the checkpoint + tensors takes a moment, so results are memoized per
(run, frame-count), invalidated when the checkpoint changes on disk. The cache
is a small bounded FIFO; concurrent misses on the same key may both build (the
build is idempotent and read-only; accepted, like `_mint_run_id`'s documented
filesystem race, to keep the lock away from model inference).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from naviernet.utils.logging import get_logger
from naviernet_api.services import runs as runs_service
from naviernet_api.settings import Settings

if TYPE_CHECKING:
    import numpy as np

log = get_logger(__name__)

# Maximum prediction-grid stride (pixels); matches the evaluation default.
# Small (e.g. test) tensors get a finer stride so contours stay resolvable.
MAX_STRIDE = 4

# Contours with fewer points than this are speckle, not an interface.
MIN_CONTOUR_POINTS = 8

_lock = threading.Lock()
_cache: dict[tuple[str, int], tuple[float, dict]] = {}
_CACHE_SIZE = 4


def interface_frames(settings: Settings, run_id: str, n_frames: int) -> dict | None:
    """Contour frames for a trained run, or None if it cannot be evaluated."""
    paths = runs_service.run_paths_for(settings, run_id)
    if paths is None or not paths.checkpoint.is_file() or not paths.tensors.is_file():
        return None

    key = (run_id, n_frames)
    checkpoint_mtime = paths.checkpoint.stat().st_mtime
    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached[0] == checkpoint_mtime:
            return cached[1]

    payload = _build(settings, run_id, n_frames)
    if payload is None:
        return None
    with _lock:
        _cache[key] = (checkpoint_mtime, payload)
        while len(_cache) > _CACHE_SIZE:
            del _cache[next(iter(_cache))]
    return payload


@dataclass(frozen=True)
class _Scene:
    """A loaded run: model + data + the strided grid it is rendered on."""

    cfg: object
    model: object
    data: object
    stride: int
    xs: np.ndarray  # µm
    ys: np.ndarray  # µm


def _build(settings: Settings, run_id: str, n_frames: int) -> dict | None:
    scene = _load_scene(settings, run_id)
    if scene is None:
        return None
    return {
        "run_id": run_id,
        "domain": _domain(scene),
        "frames": _predicted_frames(scene, n_frames),
        "measured": _measured_frames(scene),
    }


def _load_scene(settings: Settings, run_id: str) -> _Scene | None:
    from omegaconf import OmegaConf

    from naviernet.training import load_model

    cfg = runs_service.load_run_config(settings, run_id)
    paths = runs_service.run_paths_for(settings, run_id)
    if cfg is None or paths is None:
        return None
    OmegaConf.set_readonly(cfg, True)  # match compose_cfg's contract
    model, data, _ = load_model(cfg, paths)

    l_ref = float(cfg.scales.L_ref_um)
    _, height, width = data.alpha.shape
    stride = max(1, min(MAX_STRIDE, min(height, width) // 32))
    return _Scene(
        cfg=cfg,
        model=model,
        data=data,
        stride=stride,
        xs=data.x[::stride] * l_ref,
        ys=data.y[::stride] * l_ref,
    )


def _domain(scene: _Scene) -> dict:
    l_ref = float(scene.cfg.scales.L_ref_um)
    return {
        "x_um": [round(float(scene.xs[0]), 1), round(float(scene.xs[-1]), 1)],
        "y_um": [round(float(scene.ys[0]), 1), round(float(scene.ys[-1]), 1)],
        "x_pin_um": round(float(scene.data.domain.x_pin) * l_ref, 1),
    }


def _predicted_frames(scene: _Scene, n_frames: int) -> list[dict]:
    import numpy as np

    from naviernet.evaluation import predict_alpha

    t_ref_ms = float(scene.data.meta["t_ref_ms"])
    # The final camera frame is FOV-truncated; stop one frame short, like the
    # evaluation's own trajectory does.
    times = np.linspace(float(scene.data.t[0]), float(scene.data.t[-2]), n_frames)
    return [
        {
            "t_ms": round(float(t) * t_ref_ms, 4),
            "contours": _contours(
                scene, predict_alpha(scene.model, scene.data, float(t), scene.stride)
            ),
        }
        for t in times
    ]


def _measured_frames(scene: _Scene) -> list[dict]:
    t_ref_ms = float(scene.data.meta["t_ref_ms"])
    return [
        {
            "t_ms": round(float(scene.data.t[frame]) * t_ref_ms, 4),
            "contours": _contours(
                scene, scene.data.alpha[frame, :: scene.stride, :: scene.stride]
            ),
        }
        for frame in range(int(scene.data.n_event))
    ]


def _contours(scene: _Scene, field: np.ndarray) -> list[list[list[float]]]:
    """The field's threshold-level contours as rounded [x, y] µm polylines."""
    from contourpy import contour_generator

    generator = contour_generator(x=scene.xs, y=scene.ys, z=field)
    lines = generator.lines(float(scene.cfg.evaluation.threshold))
    return [
        [[round(float(x), 1), round(float(y), 1)] for x, y in line]
        for line in lines
        if len(line) >= MIN_CONTOUR_POINTS
    ]
