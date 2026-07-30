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
            model, data, c = model.bound(context.c, pin=context.pin), context.data, None
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
