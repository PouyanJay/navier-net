"""Integral checks at one instant: the x-momentum budget along the gap beside
the head, the liquid flux through every section against mass conservation,
and the heat budget."""

from __future__ import annotations

import numpy as np
import torch

from naviernet import audit
from naviernet_studio import probes, sampler
from naviernet_studio.scene import Scene

DEFAULT_GAP_BINS_UM = (450.0, 300.0, 200.0, 150.0, 100.0, 40.0)
DEFAULT_GAP_STEP_UM = 5.0
FLUX_NX = 320
FLUX_NY = 121
# Half-window of the central difference of the vapour area, in t*.
AREA_RATE_STEP = 0.01
GAP_TERMS = ("mom_dpdx", "mom_friction", "mom_viscous", "mom_unsteady", "mom_convective")


def gap_budget(scene: Scene, t_star: float, upper: bool, bins_um: tuple[float, ...]) -> dict:
    """The pressure change each term of the trained momentum law alone would
    need across each bin of the mid-gap line, in Pa, and the leftover."""
    far, near = bins_um[0], bins_um[-1]
    n = int(round((far - near) / DEFAULT_GAP_STEP_UM)) + 1
    request = probes.ProbeRequest(
        kind="section",
        quantities=list(GAP_TERMS),
        section="mid_gap_top" if upper else "mid_gap_bottom",
        span_um=(far, near),
        n=n,
    )
    stations = probes.stations_for(scene, request, t_star)
    xs, ys = stations.star(scene)
    values = sampler.evaluate(scene, sampler.points_at(xs, ys, t_star), [*GAP_TERMS, "u"])
    # Along the line the law reads dp/dx = -(friction + viscous + inertia); the
    # integral of each term gives the pressure change it alone would require.
    per_mm = {k: np.nan_to_num(values[k]) for k in GAP_TERMS}
    dx_mm = np.gradient(stations.x_um) / 1e3
    bins = []
    for a, b in zip(bins_um[:-1], bins_um[1:], strict=True):
        pick = (stations.s_um <= a) & (stations.s_um >= b)
        integral = {k: float(np.sum(v[pick] * dx_mm[pick])) for k, v in per_mm.items()}
        dp = integral["mom_dpdx"]
        needed = {k: -integral[k] for k in GAP_TERMS if k != "mom_dpdx"}
        bins.append(
            {
                "from_um": a,
                "to_um": b,
                "dp_Pa": dp,
                **{f"{k}_Pa": v for k, v in needed.items()},
                "leftover_Pa": dp - sum(needed.values()),
            }
        )
    return {
        "side": "upper" if upper else "lower",
        "stations": {
            "behind_nose_um": stations.s_um.tolist(),
            "x_um": stations.x_um.tolist(),
            "y_um": stations.y_um.tolist(),
            "u_mm_s": values["u"].tolist(),
            **{f"{k}_Pa_per_mm": per_mm[k].tolist() for k in GAP_TERMS},
        },
        "bins": bins,
    }


def _grid_points(scene: Scene, t_star: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = scene.data
    xs = np.linspace(data.domain.x_min, data.domain.x_max, FLUX_NX)
    ys = np.linspace(float(data.y_lo), float(data.y_hi), FLUX_NY)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return xs, ys, sampler.points_at(grid_x.ravel(), grid_y.ravel(), t_star)


def _vapour_mask(scene: Scene, points: np.ndarray) -> np.ndarray:
    model = scene.model
    tensor = torch.tensor(points, dtype=torch.float32)
    with scene.lock, torch.no_grad():
        if getattr(model, "front_geometry", False):
            return model._phi_with_frozen_geometry(tensor, None).numpy().ravel() > 0
        return model.alpha(tensor).numpy().ravel() > 0.5


def flux_budget(scene: Scene, t_star: float) -> dict:
    """The liquid volume flux through each section against the rule that it
    equals the inlet flux plus the rate at which vapour area grows upstream."""
    data, scales = scene.data, scene.scales
    xs, ys, points = _grid_points(scene, t_star)
    dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    shape = (FLUX_NY, FLUX_NX)
    u = (sampler.evaluate(scene, points, ["u"])["u"] / scales.speed_mm_s).reshape(shape)
    vapour = _vapour_mask(scene, points).reshape(shape)
    q_total = u.sum(0) * dy
    q_liquid = (u * ~vapour).sum(0) * dy

    def upstream_area(t: float) -> np.ndarray:
        pts = sampler.points_at(points[:, 0], points[:, 1], t)
        mask = _vapour_mask(scene, pts).reshape(shape)
        return np.cumsum(mask.sum(0) * dy * dx)

    lo, hi = float(data.domain.t_min), float(data.domain.t_max)
    a, b = max(t_star - AREA_RATE_STEP, lo), min(t_star + AREA_RATE_STEP, hi)
    growth = (upstream_area(b) - upstream_area(a)) / max(b - a, 1e-9)
    q_required = scales.q_in_star + growth
    to_ml_h = scales.flux_mL_h
    return {
        "x_um": (xs * scales.length_um).tolist(),
        "q_total_mL_h": (q_total * to_ml_h).tolist(),
        "q_liquid_mL_h": (q_liquid * to_ml_h).tolist(),
        "q_required_mL_h": (q_required * to_ml_h).tolist(),
        "q_inlet_mL_h": scales.q_in_star * to_ml_h,
        "upstream_growth_mL_h": (growth * to_ml_h).tolist(),
        "defect": ((q_liquid - q_required) / max(scales.q_in_star, 1e-9)).tolist(),
    }


def heat_budget(scene: Scene, frame: int) -> dict | None:
    """The audit's latent-power estimates at a camera frame, in mW."""
    row = scene.row_of(frame)
    if row is None or row >= scene.data.n_event:
        return None
    cfg, data = scene.cfg, scene.data
    grid = audit.audit_grid(data, int(cfg.evaluation.audit_nx), int(cfg.evaluation.audit_ny))
    t_star = float(data.t[row])
    with scene.lock:
        fields = audit.fields_on_grid(scene.model, grid, t_star)
        budget = audit.heat_budget(cfg, scene.model, data, scene.groups, row, fields, grid)
    return {"frame": frame, "t_ms": t_star * scene.scales.time_ms, **budget}
