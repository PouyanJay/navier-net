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
import struct
import threading

import numpy as np

from naviernet_studio import catalogue, geometry, sampler
from naviernet_studio.scene import Scene

MAGIC = b"NVST"
DEFAULT_NX = 512
DEFAULT_NY = 171
MAX_NX = 1536
MAX_NY = 513
GRID_CACHE_SIZE = 256

_cache: dict[tuple, tuple[dict, list[np.ndarray]]] = {}
_lock = threading.Lock()


def axes(scene: Scene, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    data = scene.data
    xs = np.linspace(data.domain.x_min, data.domain.x_max, nx)
    ys = np.linspace(data.y_lo, data.y_hi, ny)
    return xs, ys


def _key(scene: Scene, quantity_id: str, t_star: float, nx: int, ny: int) -> tuple:
    return (scene.run_id, scene.checkpoint_mtime, quantity_id, round(t_star, 6), nx, ny)


def _arrays(
    scene: Scene, quantity_ids: list[str], t_star: float, nx: int, ny: int
) -> list[np.ndarray]:
    """The grids of several quantities at one instant. The ones not yet cached
    are evaluated in a single pass, so they share the flow solve and the model's
    intermediates instead of paying for each one separately."""
    with _lock:
        cached = {q: _cache.get(_key(scene, q, t_star, nx, ny)) for q in quantity_ids}
    missing = list(dict.fromkeys(q for q in quantity_ids if cached[q] is None))
    if missing:
        xs, ys = axes(scene, nx, ny)
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = sampler.points_at(grid_x.ravel(), grid_y.ravel(), t_star)
        values = sampler.evaluate(scene, points, missing)
        with _lock:
            for q in missing:
                array = values[q].reshape(ny, nx).astype(np.float32)
                _cache[_key(scene, q, t_star, nx, ny)] = ({}, [array])
            while len(_cache) > GRID_CACHE_SIZE:
                del _cache[next(iter(_cache))]
    with _lock:
        return [_cache[_key(scene, q, t_star, nx, ny)][1][0] for q in quantity_ids]


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
