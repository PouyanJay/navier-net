"""Field grids at one instant, packed as binary for the renderer.

Layout of a grid response (little-endian):

    bytes 0-3    magic ``NVST``
    bytes 4-7    uint32 length of the JSON header
    header       UTF-8 JSON: shape, extent, quantities, units, ranges, boundary
    payload      one float32 array per quantity, row-major (ny, nx), in order

NaN marks a value the quantity does not define there (a liquid law inside the
vapour, for instance); the renderer leaves those cells transparent.
"""

from __future__ import annotations

import json
import queue
import struct
import threading
from pathlib import Path

import numpy as np

from naviernet.utils.logging import get_logger
from naviernet_studio import catalogue, geometry, sampler
from naviernet_studio.scene import Scene

log = get_logger(__name__)

MAGIC = b"NVST"
DEFAULT_NX = 512
DEFAULT_NY = 171
MAX_NX = 1536
MAX_NY = 513
GRID_CACHE_SIZE = 256
# Grids are kept on disk beside the run, under the checkpoint's own stamp, so a
# restart of the studio (or a retrained run) never serves a stale picture.
CACHE_DIRNAME = ".studio"


_cache: dict[tuple, tuple[dict, list[np.ndarray]]] = {}
_lock = threading.Lock()


def axes(scene: Scene, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    data = scene.data
    xs = np.linspace(data.domain.x_min, data.domain.x_max, nx)
    ys = np.linspace(data.y_lo, data.y_hi, ny)
    return xs, ys


def _key(scene: Scene, quantity_id: str, t_star: float, nx: int, ny: int) -> tuple:
    return (scene.run_id, scene.checkpoint_mtime, quantity_id, round(t_star, 6), nx, ny)


def _disk_path(scene: Scene, quantity_id: str, t_star: float, nx: int, ny: int) -> Path:
    stamp = f"{scene.checkpoint_mtime:.0f}"
    name = f"{quantity_id}_t{round(t_star, 6):+.6f}_{nx}x{ny}.npy"
    return scene.paths.output_dir / CACHE_DIRNAME / "grids" / stamp / name


def _from_disk(scene: Scene, quantity_id: str, t_star: float, nx: int, ny: int):
    path = _disk_path(scene, quantity_id, t_star, nx, ny)
    if not path.is_file():
        return None
    try:
        array = np.load(path)
    except (OSError, ValueError) as exc:
        log.warning("unreadable grid cache %s: %s", path, exc)
        return None
    if array.shape != (ny, nx):
        return None
    return array.astype(np.float32, copy=False)


def _to_disk(scene: Scene, quantity_id: str, t_star: float, nx: int, ny: int, array) -> None:
    path = _disk_path(scene, quantity_id, t_star, nx, ny)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".tmp.npy")
        np.save(partial, array)
        partial.replace(path)
    except OSError as exc:
        log.warning("could not keep grid cache %s: %s", path, exc)


def is_cached(scene: Scene, quantity_id: str, t_star: float, nx: int, ny: int) -> bool:
    with _lock:
        if _key(scene, quantity_id, t_star, nx, ny) in _cache:
            return True
    return _disk_path(scene, quantity_id, t_star, nx, ny).is_file()


def _remember(scene: Scene, quantity_id: str, t_star: float, nx: int, ny: int, array) -> None:
    with _lock:
        _cache[_key(scene, quantity_id, t_star, nx, ny)] = ({}, [array])
        while len(_cache) > GRID_CACHE_SIZE:
            del _cache[next(iter(_cache))]


def _arrays(
    scene: Scene, quantity_ids: list[str], t_star: float, nx: int, ny: int
) -> list[np.ndarray]:
    """The grids of several quantities at one instant: from memory, else from
    disk, else evaluated. The missing ones are evaluated in a single pass, so
    they share the flow solve and the model's intermediates instead of paying
    for each one separately."""
    with _lock:
        cached = {q: _cache.get(_key(scene, q, t_star, nx, ny)) for q in quantity_ids}
    missing = []
    for q in dict.fromkeys(q for q in quantity_ids if cached[q] is None):
        array = _from_disk(scene, q, t_star, nx, ny)
        if array is None:
            missing.append(q)
        else:
            _remember(scene, q, t_star, nx, ny, array)
    if missing:
        xs, ys = axes(scene, nx, ny)
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = sampler.points_at(grid_x.ravel(), grid_y.ravel(), t_star)
        values = sampler.evaluate(scene, points, missing)
        for q in missing:
            array = values[q].reshape(ny, nx).astype(np.float32)
            _remember(scene, q, t_star, nx, ny, array)
            _to_disk(scene, q, t_star, nx, ny, array)
    with _lock:
        return [_cache[_key(scene, q, t_star, nx, ny)][1][0] for q in quantity_ids]


# --- prefetch -----------------------------------------------------------------
# A picture of one frame is usually followed by a request for the next, so once
# a grid has been served the same quantities are evaluated at every other camera
# frame of the run, nearest frames first, on one background thread. The frame on
# screen keeps priority because the sampler takes the model lock per chunk.

_prefetch_queue: queue.Queue[tuple[Scene, tuple[str, ...], float, int, int] | None] = (
    queue.Queue()
)
_prefetch_pending: set[tuple] = set()
_prefetch_thread: threading.Thread | None = None


def _prefetch_worker() -> None:
    while True:
        job = _prefetch_queue.get()
        if job is None:
            return
        scene, ids, t_star, nx, ny = job
        pending_key = (scene.run_id, scene.checkpoint_mtime, ids, round(t_star, 6), nx, ny)
        try:
            missing = [q for q in ids if not is_cached(scene, q, t_star, nx, ny)]
            if missing:
                _arrays(scene, missing, t_star, nx, ny)
        except Exception as exc:  # noqa: BLE001 (a failed prefetch only costs the warm start)
            log.warning("prefetch of %s at t*=%.3f failed: %s", ", ".join(ids), t_star, exc)
        finally:
            with _lock:
                _prefetch_pending.discard(pending_key)
            _prefetch_queue.task_done()


def _drop_queued() -> None:
    """Forget the frames still waiting: the latest request is the one the person
    is looking at, so its frames go first and the rest are queued again if they
    come back into view. The job in progress finishes."""
    while True:
        try:
            job = _prefetch_queue.get_nowait()
        except queue.Empty:
            return
        if job is not None:
            scene, ids, t_star, nx, ny = job
            with _lock:
                _prefetch_pending.discard(
                    (scene.run_id, scene.checkpoint_mtime, ids, round(t_star, 6), nx, ny)
                )
        _prefetch_queue.task_done()


def prefetch_frames(
    scene: Scene, quantity_ids: list[str], t_star: float, nx: int, ny: int
) -> int:
    """Queue the quantities at every camera frame other than ``t_star``, nearest
    first, replacing whatever was still queued; returns how many frames were
    queued (already cached ones are skipped)."""
    global _prefetch_thread  # noqa: PLW0603 (one worker for the process)
    ids = tuple(dict.fromkeys(quantity_ids))
    _drop_queued()
    instants = sorted(
        (float(f.t_star) for f in scene.frames() if f.t_star is not None),
        key=lambda t: abs(t - t_star),
    )
    queued = 0
    for instant in instants:
        if abs(instant - t_star) < 1e-6:
            continue
        if all(is_cached(scene, q, instant, nx, ny) for q in ids):
            continue
        pending_key = (scene.run_id, scene.checkpoint_mtime, ids, round(instant, 6), nx, ny)
        with _lock:
            if pending_key in _prefetch_pending:
                continue
            _prefetch_pending.add(pending_key)
        _prefetch_queue.put((scene, ids, instant, nx, ny))
        queued += 1
    if queued and (_prefetch_thread is None or not _prefetch_thread.is_alive()):
        _prefetch_thread = threading.Thread(
            target=_prefetch_worker, name="studio-grid-prefetch", daemon=True
        )
        _prefetch_thread.start()
    return queued


def _range(array: np.ndarray) -> tuple[float | None, float | None]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None, None
    return float(finite.min()), float(finite.max())


def grid(
    scene: Scene, quantity_ids: list[str], t_star: float, nx: int, ny: int
) -> tuple[dict, list[np.ndarray]]:
    """The header and arrays for a set of quantities at one instant."""
    arrays = _arrays(scene, quantity_ids, t_star, nx, ny)
    xs, ys = axes(scene, nx, ny)
    L = scene.scales.length_um
    outline = geometry.interface(scene, t_star)
    header = {
        "run": scene.run_id,
        "frame": scene.frame_of(t_star),
        "t_ms": t_star * scene.scales.time_ms,
        "t_star": t_star,
        "shape": [ny, nx],
        "extent_um": [float(xs[0]) * L, float(xs[-1]) * L, float(ys[0]) * L, float(ys[-1]) * L],
        "quantities": [
            {
                "id": i,
                "unit": catalogue.BY_ID[i].unit,
                "signed": catalogue.BY_ID[i].signed,
                "cmap": catalogue.BY_ID[i].cmap,
                "vmin": _range(a)[0],
                "vmax": _range(a)[1],
            }
            for i, a in zip(quantity_ids, arrays, strict=True)
        ],
        "boundary_um": outline["rings_um"],
        "nose_um": outline["nose_um"],
        "dtype": "float32",
        "byte_order": "little",
    }
    return header, arrays


def pack(header: dict, arrays: list[np.ndarray]) -> bytes:
    head = json.dumps(header, allow_nan=False).encode("utf-8")
    body = b"".join(np.ascontiguousarray(a, dtype="<f4").tobytes() for a in arrays)
    return MAGIC + struct.pack("<I", len(head)) + head + body


def as_table(header: dict, arrays: list[np.ndarray]) -> tuple[list[str], list[list]]:
    """Long-format rows (x_um, y_um, values...) for CSV export."""
    ny, nx = header["shape"]
    x0, x1, y0, y1 = header["extent_um"]
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    columns = ["x_um", "y_um", *(q["id"] for q in header["quantities"])]
    stack = np.column_stack([grid_x.ravel(), grid_y.ravel(), *(a.ravel() for a in arrays)])
    return columns, stack.tolist()
