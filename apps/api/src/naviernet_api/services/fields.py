"""Field maps: a run's predicted fields evaluated on a grid at one instant.

Read-side evaluation of the run's own checkpoint (the reconstruction service's
pattern): the API computes no physics of its own — it calls the model's field
heads and scales the outputs into physical units using the run's config.
Results are cached per (run, dataset, field, t) and invalidated when the
checkpoint changes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from naviernet.utils.logging import get_logger
from naviernet_api.services import runs as runs_service
from naviernet_api.settings import Settings

log = get_logger(__name__)

# Every field the endpoint can serve; p/T only exist on Stage-B checkpoints.
FIELD_NAMES = ("alpha", "u", "v", "umag", "s", "p", "T")
# |residual| maps: where each governing equation is least satisfied. Stage-A
# residuals exist on every checkpoint; momentum/energy need the p/T heads.
RESIDUAL_NAMES = ("res_vof", "res_div", "res_mom", "res_energy")

_CACHE_SIZE = 24
_cache: dict[tuple, tuple[float, dict]] = {}
_scenes: dict[tuple, tuple[float, _FieldScene]] = {}
_lock = threading.Lock()

# Like reconstruction: small (test) tensors get a finer stride so the grid
# stays resolvable; real frames are strided down to a browser-sized payload.
_MAX_STRIDE = 4


class FieldUnavailable(Exception):
    """A well-formed request for a field this checkpoint does not have."""


# Arrows need spacing, not pixels: a vector per grid cell is a solid band of
# ink. The quiver is sampled onto its own coarse lattice, sized so a channel
# reads as a flow rather than as a texture.
_QUIVER_COLUMNS = 34
_QUIVER_ROWS = 9
# A contour needs the opposite: the interface is the overlay that makes the
# quiver readable, so alpha is evaluated on the FULL field grid.
_MIN_CONTOUR_POINTS = 6


@dataclass(frozen=True)
class _FieldScene:
    """A loaded run, scoped to one dataset's data + conditioning."""

    model: object
    data: object
    c: object  # conditioning row (None for single-dataset checkpoints)
    cfg: object  # the run's own composed config (groups fallback for Stage B)
    stride: int
    l_ref_um: float
    u_ref_m_s: float


def field_map(
    settings: Settings, run_id: str, name: str, t_star: float, dataset: str | None
) -> dict | None:
    """The field map payload, None if the run/model is unavailable.

    Raises :class:`FieldUnavailable` for a Stage-B field on a Stage-A model.
    """
    paths = runs_service.run_paths_for(settings, run_id)
    if paths is None or not paths.checkpoint.is_file():
        return None

    mtime = paths.checkpoint.stat().st_mtime
    key = (run_id, dataset, name, round(float(t_star), 3))
    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    scene = _scene(settings, run_id, dataset, mtime)
    if scene is None:
        return None
    payload = _evaluate(scene, run_id, dataset, name, float(t_star))
    with _lock:
        _cache[key] = (mtime, payload)
        while len(_cache) > _CACHE_SIZE:
            del _cache[next(iter(_cache))]
    return payload


def _scene(
    settings: Settings, run_id: str, dataset: str | None, mtime: float
) -> _FieldScene | None:
    skey = (run_id, dataset)
    with _lock:
        cached = _scenes.get(skey)
        if cached is not None and cached[0] == mtime:
            return cached[1]
    scene = _load_scene(settings, run_id, dataset)
    if scene is None:
        return None
    with _lock:
        _scenes[skey] = (mtime, scene)
        while len(_scenes) > _CACHE_SIZE:
            del _scenes[next(iter(_scenes))]
    return scene


def _load_scene(settings: Settings, run_id: str, dataset: str | None) -> _FieldScene | None:
    from omegaconf import OmegaConf

    from naviernet.training import load_joint, load_model

    cfg = runs_service.load_run_config(settings, run_id)
    paths = runs_service.run_paths_for(settings, run_id)
    if cfg is None or paths is None:
        return None
    OmegaConf.set_readonly(cfg, True)

    joint = bool(cfg.datasets) and len(cfg.datasets) > 1
    try:
        if joint:
            model, contexts, _ = load_joint(cfg, paths)
            if dataset is None:
                context = contexts[0]  # the run's primary dataset
            else:
                # An unknown dataset must 404, never silently substitute
                # another condition's values under the requested name.
                context = next((cx for cx in contexts if cx.name == dataset), None)
                if context is None:
                    log.warning("run %s spans no dataset %r", run_id, dataset)
                    return None
            # Bind the dataset's conditioning row -- and, on a hard-pin run, its
            # root anchor -- into the model, so every downstream field/residual
            # call is per-dataset correct without threading extra context.
            model = model.bound(context.c, pin=context.pin)
            data, c = context.data, None
        else:
            model, data, _ = load_model(cfg, paths)
            c = None
    except FileNotFoundError:
        return None

    _, height, width = data.alpha.shape
    stride = max(1, min(_MAX_STRIDE, min(height, width) // 32))
    return _FieldScene(
        model=model,
        data=data,
        c=c,
        cfg=cfg,
        stride=stride,
        l_ref_um=float(cfg.scales.L_ref_um),
        u_ref_m_s=float(cfg.scales.U_ref),
    )


def _evaluate(
    scene: _FieldScene, run_id: str, dataset: str | None, name: str, t_star: float
) -> dict:
    import torch

    data, model = scene.data, scene.model
    # Clamp into the trained time span; extrapolating past the footage would
    # present pure invention as a result.
    t_lo, t_hi = float(data.t[0]), float(data.t[-1])
    t = min(max(t_star, t_lo), t_hi)

    points, _, shape = data.frame_grid(0, scene.stride)
    points = points.clone()
    points[:, 2] = t

    if name in RESIDUAL_NAMES:
        # Residuals differentiate through the model, so no no_grad here.
        values, unit = _residual_values(scene, model, points, name)
    else:
        ctx = None if scene.c is None else scene.c.expand(points.shape[0], -1)
        with torch.no_grad():
            values, unit = _field_values(scene, model, points, ctx, name)
    grid = values.cpu().numpy().reshape(shape)

    xs = (data.x[:: scene.stride] * scene.l_ref_um).tolist()
    ys = (data.y[:: scene.stride] * scene.l_ref_um).tolist()
    return {
        "run_id": run_id,
        "dataset": dataset,
        "name": name,
        "unit": unit,
        "t_star": round(t, 4),
        "t_min_star": round(t_lo, 4),
        "t_max_star": round(t_hi, 4),
        "x_um": [round(x, 2) for x in xs],
        "y_um": [round(y, 2) for y in ys],
        "values": [[round(float(v), 4) for v in row] for row in grid],
        "vmin": round(float(np.nanmin(grid)), 4),
        "vmax": round(float(np.nanmax(grid)), 4),
        "fields_available": available_fields(model),
    }


def available_fields(model) -> list[str]:
    """The servable names for this checkpoint (fields then residual maps)."""
    stage_b = {"p", "T"} <= set(model.fields)
    derived = {"alpha", "u", "v", "umag", "s", "res_vof", "res_div"} | (
        {"p", "T", "res_mom", "res_energy"} if stage_b else set()
    )
    ordered = list(FIELD_NAMES) + list(RESIDUAL_NAMES)
    return [name for name in ordered if name in derived]


def _field_values(scene: _FieldScene, model, points, ctx, name: str):
    """One field's values on the batch of points, plus its display unit."""
    import torch

    u_mm_s = scene.u_ref_m_s * 1e3
    if name == "alpha":
        return model.alpha(points, ctx), "–"
    if name in ("u", "v", "umag"):
        u, v = model.velocity(points, ctx)
        if name == "u":
            return u * u_mm_s, "mm·s⁻¹"
        if name == "v":
            return v * u_mm_s, "mm·s⁻¹"
        return torch.hypot(u, v) * u_mm_s, "mm·s⁻¹"
    if name == "s":
        # s* is scaled by U_ref/L_ref: the dilatation rate in 1/s.
        return (
            model.source(points, ctx) * (scene.u_ref_m_s / (scene.l_ref_um * 1e-6)),
            "s⁻¹",
        )
    try:
        if name == "p":
            return model.pressure(points, ctx), "p*"
        if name == "T":
            return model.temperature(points, ctx), "θ"
    except KeyError as exc:
        raise FieldUnavailable(str(exc)) from exc
    raise FieldUnavailable(f"unknown field {name!r}")


def _residual_values(scene: _FieldScene, model, points, name: str):
    """|residual| of one governing equation on the grid, via autograd."""
    import torch

    from naviernet.physics import residuals as residuals_mod

    x = points.clone().requires_grad_(True)
    if name in ("res_vof", "res_div"):
        stage_a = residuals_mod.stage_a_residuals(model, x, scene.c)
        value = stage_a.vof if name == "res_vof" else stage_a.div
        return value.abs().detach().squeeze(-1), "|r|"

    if not {"p", "T"} <= set(model.fields):
        raise FieldUnavailable(
            f"residual {name!r} needs the Stage-B p/T heads "
            f"(this model has: {model.fields}). Enable momentum & energy and retrain."
        )
    groups = getattr(scene.data, "groups", None) or _computed_groups(scene.cfg)
    stage_b = residuals_mod.stage_b_residuals(
        model, x, groups, r_int_star=float(model.r_int_star), c=scene.c
    )
    value = torch.hypot(stage_b.mom_x, stage_b.mom_y) if name == "res_mom" else stage_b.energy
    return value.abs().detach().squeeze(-1), "|r|"


def _computed_groups(cfg) -> dict[str, float]:
    from naviernet.physics.groups import compute_groups

    return compute_groups(cfg)


def velocity_field(
    settings: Settings, run_id: str, t_star: float, dataset: str | None
) -> dict | None:
    """The inferred velocity field at one instant, as a quiver plus the front.

    The figure this serves is the platform's strongest single claim: the camera
    measured an interface and nothing else, so every arrow here is inferred from
    the governing equations alone. It therefore travels with the interface
    contour at the same instant, because an arrow field without the boundary it
    is flowing around cannot be read.

    Two lattices, deliberately: the arrows on a coarse one (a vector per pixel
    is ink, not information) and alpha on the field grid, so the contour stays
    smooth.
    """
    paths = runs_service.run_paths_for(settings, run_id)
    if paths is None or not paths.checkpoint.is_file():
        return None

    mtime = paths.checkpoint.stat().st_mtime
    key = (run_id, dataset, "__quiver__", round(float(t_star), 3))
    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    scene = _scene(settings, run_id, dataset, mtime)
    if scene is None:
        return None
    payload = _evaluate_velocity(scene, run_id, dataset, float(t_star))
    with _lock:
        _cache[key] = (mtime, payload)
        while len(_cache) > _CACHE_SIZE:
            del _cache[next(iter(_cache))]
    return payload


def _evaluate_velocity(
    scene: _FieldScene, run_id: str, dataset: str | None, t_star: float
) -> dict:
    import torch

    data, model = scene.data, scene.model
    t_lo, t_hi = float(data.t[0]), float(data.t[-1])
    t = min(max(t_star, t_lo), t_hi)
    u_mm_s = scene.u_ref_m_s * 1e3

    xs_star = np.asarray(data.x)
    ys_star = np.asarray(data.y)
    ix = _interior(len(xs_star), _QUIVER_COLUMNS)
    iy = _interior(len(ys_star), _QUIVER_ROWS)
    gx, gy = np.meshgrid(xs_star[ix], ys_star[iy], indexing="xy")
    flat = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, t)], axis=1)
    template = data.frame_grid(0, scene.stride)[0]
    points = torch.as_tensor(flat, dtype=template.dtype, device=template.device)

    ctx = None if scene.c is None else scene.c.expand(points.shape[0], -1)
    with torch.no_grad():
        u, v = model.velocity(points, ctx)
    u_grid = (u.cpu().numpy() * u_mm_s).reshape(gx.shape)
    v_grid = (v.cpu().numpy() * u_mm_s).reshape(gx.shape)
    speed = np.hypot(u_grid, v_grid)

    return {
        "run_id": run_id,
        "dataset": dataset,
        "unit": "mm·s⁻¹",
        "t_star": round(t, 4),
        "t_min_star": round(t_lo, 4),
        "t_max_star": round(t_hi, 4),
        "t_ms": round(t * float(data.meta["t_ref_ms"]), 4),
        "x_um": [round(float(x) * scene.l_ref_um, 2) for x in xs_star[ix]],
        "y_um": [round(float(y) * scene.l_ref_um, 2) for y in ys_star[iy]],
        "u": [[round(float(value), 4) for value in row] for row in u_grid],
        "v": [[round(float(value), 4) for value in row] for row in v_grid],
        "speed_max": round(float(np.nanmax(speed)), 4),
        "speed_mean": round(float(np.nanmean(speed)), 4),
        "domain_um": [
            round(float(xs_star[0]) * scene.l_ref_um, 2),
            round(float(xs_star[-1]) * scene.l_ref_um, 2),
            round(float(ys_star[0]) * scene.l_ref_um, 2),
            round(float(ys_star[-1]) * scene.l_ref_um, 2),
        ],
        "interface": _interface_contours(scene, t),
    }


def _interior(count: int, wanted: int) -> np.ndarray:
    """`wanted` anchor indices strictly inside a grid of `count` points.

    Strictly inside on purpose: an arrow drawn on the wall is half outside the
    channel, and the first and last columns carry the inlet and outlet
    conditions rather than the flow. A grid too small to have an interior (the
    test tensors) collapses to its middle rather than falling back onto a wall.
    """
    if count <= 2:
        return np.array([count // 2])
    span = min(wanted, count - 2)
    return np.unique(np.linspace(1, count - 2, span).round().astype(int))


def _interface_contours(scene: _FieldScene, t: float) -> list[list[list[float]]]:
    """The alpha = threshold contour at this instant, in µm."""
    from contourpy import contour_generator

    from naviernet.evaluation import predict_alpha

    alpha = predict_alpha(scene.model, scene.data, float(t), scene.stride)
    xs = (np.asarray(scene.data.x)[:: scene.stride] * scene.l_ref_um).astype(float)
    ys = (np.asarray(scene.data.y)[:: scene.stride] * scene.l_ref_um).astype(float)
    generator = contour_generator(x=xs, y=ys, z=np.asarray(alpha))
    lines = generator.lines(float(scene.cfg.evaluation.threshold))
    return [
        [[round(float(x), 1), round(float(y), 1)] for x, y in line]
        for line in lines
        if len(line) >= _MIN_CONTOUR_POINTS
    ]
