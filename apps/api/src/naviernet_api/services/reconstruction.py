"""Interface reconstruction data for the viewport.

Evaluates the trained model's volume fraction on a strided grid over a set of
continuous timesteps and extracts the alpha = 0.5 interface contours as
polylines (plus the measured contours at the camera instants). This is pure
visualization geometry — the physics all lives in the pipeline; contour
extraction uses contourpy, the same engine matplotlib's own contour plots use.

Loading the checkpoint + tensors takes a moment, so results are memoized per
(run, frame-count) and invalidated when the checkpoint changes on disk.
"""

from __future__ import annotations

import threading

from naviernet.utils.logging import get_logger
from naviernet_api.services import runs as runs_service
from naviernet_api.settings import Settings

log = get_logger(__name__)

# Maximum prediction-grid stride (pixels) — matches the evaluation default.
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


def _build(settings: Settings, run_id: str, n_frames: int) -> dict | None:
    import numpy as np
    from contourpy import contour_generator

    from naviernet.evaluation import predict_alpha
    from naviernet.training import load_model

    cfg = runs_service.load_run_config(settings, run_id)
    paths = runs_service.run_paths_for(settings, run_id)
    if cfg is None or paths is None:
        return None
    model, data, _ = load_model(cfg, paths)

    l_ref = float(cfg.scales.L_ref_um)
    t_ref_ms = float(data.meta["t_ref_ms"])
    _, height, width = data.alpha.shape
    stride = max(1, min(MAX_STRIDE, min(height, width) // 32))
    xs = data.x[::stride] * l_ref
    ys = data.y[::stride] * l_ref

    def contours_of(field: np.ndarray) -> list[list[list[float]]]:
        generator = contour_generator(x=xs, y=ys, z=field)
        lines = generator.lines(float(cfg.evaluation.threshold))
        return [
            [[round(float(x), 1), round(float(y), 1)] for x, y in line]
            for line in lines
            if len(line) >= MIN_CONTOUR_POINTS
        ]

    # The final camera frame is FOV-truncated; stop one frame short, like the
    # evaluation's own trajectory does.
    times = np.linspace(float(data.t[0]), float(data.t[-2]), n_frames)
    frames = [
        {
            "t_ms": round(float(t) * t_ref_ms, 4),
            "contours": contours_of(predict_alpha(model, data, float(t), stride)),
        }
        for t in times
    ]
    measured = [
        {
            "t_ms": round(float(data.t[frame]) * t_ref_ms, 4),
            "contours": contours_of(data.alpha[frame, ::stride, ::stride]),
        }
        for frame in range(int(cfg.experiment.n_frames_event))
    ]
    return {
        "run_id": run_id,
        "domain": {
            "x_um": [round(float(xs[0]), 1), round(float(xs[-1]), 1)],
            "y_um": [round(float(ys[0]), 1), round(float(ys[-1]), 1)],
            "x_pin_um": round(float(data.domain.x_pin) * l_ref, 1),
        },
        "frames": frames,
        "measured": measured,
    }
