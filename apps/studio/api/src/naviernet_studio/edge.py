"""Quantities along the bubble edge, per explicit front sample, at one instant."""

from __future__ import annotations

import numpy as np
import torch

from naviernet.audit import _segment_codes, jump_shape_pixels
from naviernet.physics.residuals import (
    interface_mass_flux,
    interface_shear_residual,
    jump_gap_curvature,
    jump_total_curvature,
    kinematic_residual,
    laplace_jump_residual,
    pressure_implied_curvature,
)
from naviernet_studio import catalogue, geometry
from naviernet_studio.scene import Scene

SEGMENT_NAMES = ("upper", "lower", "root", "nose")


def _front(scene: Scene, t_star: float, n_body: int, n_cap: int):
    device = next(scene.model.parameters()).device
    instant = torch.tensor([[t_star]], dtype=torch.float32, device=device)
    return scene.model.front(instant, n_body, n_cap)


def _as_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().ravel()


def evaluate(
    scene: Scene, t_star: float, quantity_ids: list[str], n_body: int, n_cap: int
) -> dict:
    """Each requested edge quantity in physical units per front sample."""
    unknown = [i for i in quantity_ids if i not in catalogue.EDGE_IDS]
    if unknown:
        raise KeyError(f"not edge quantities: {', '.join(unknown)}")
    scales, groups, model = scene.scales, scene.groups, scene.model
    out: dict[str, np.ndarray] = {}
    with scene.lock, torch.enable_grad():
        front = _front(scene, t_star, n_body, n_cap)
        points = front.points
        xy_um = _as_np(points[:, 0]) * scales.length_um, _as_np(points[:, 1]) * scales.length_um
        normal = front.normal.detach().cpu().numpy()
        raw = _raw_edge(scene, front, quantity_ids)
    codes = _segment_codes(front)
    for i in quantity_ids:
        if i == "edge_x":
            out[i] = xy_um[0]
        elif i == "edge_y":
            out[i] = xy_um[1]
        elif i == "segment":
            out[i] = codes.astype(float)
        elif i == "normal_x":
            out[i] = normal[:, 0]
        elif i == "normal_y":
            out[i] = normal[:, 1]
        elif i == "arclength":
            out[i] = _arclength(scene, t_star, np.column_stack(xy_um))
        elif i == "jump_px":
            out[i] = _jump_pixels(scene, raw["yl_jump"], raw["kappa_par"])
        else:
            out[i] = _scaled(scene, i, raw[i])
    _ = model, groups
    return {
        "n_samples": int(len(codes)),
        "segments": SEGMENT_NAMES,
        "values": out,
    }


def _scaled(scene: Scene, quantity_id: str, value: np.ndarray) -> np.ndarray:
    q = catalogue.BY_ID[quantity_id]
    return value * getattr(scene.scales, q.scale) if q.scale else value


def _raw_edge(scene: Scene, front, quantity_ids: list[str]) -> dict[str, np.ndarray]:  # noqa: PLR0912
    """Model-unit values of the edge quantities the request needs (plus the
    ingredients of the derived ones), per sample."""
    model, groups = scene.model, scene.groups
    wanted = set(quantity_ids)
    if "jump_px" in wanted:
        wanted |= {"yl_jump", "kappa_par"}
    raw: dict[str, np.ndarray] = {}
    if wanted & {"kappa_par", "jump_px"}:
        raw["kappa_par"] = _as_np(front.kappa_par)
    if "kappa_gap" in wanted:
        raw["kappa_gap"] = _as_np(jump_gap_curvature(model, front, groups))
    if wanted & {"kappa_total", "capillary"}:
        total = jump_total_curvature(model, front, groups)
        raw["kappa_total"] = _as_np(total)
        raw["capillary"] = _as_np(total / groups["We"])
    if "normal_speed" in wanted:
        raw["normal_speed"] = _as_np(front.normal_speed)
    if wanted & {"u_normal", "u_tangent", "edge_u", "edge_v"}:
        u, v = model.velocity(front.points)
        n = front.normal
        raw["edge_u"], raw["edge_v"] = _as_np(u), _as_np(v)
        raw["u_normal"] = _as_np(u * n[:, 0:1] + v * n[:, 1:2])
        raw["u_tangent"] = _as_np(-u * n[:, 1:2] + v * n[:, 0:1])
    if "kinematic" in wanted:
        flux = interface_mass_flux(model, front, groups) if scene.sharp_thermal else None
        raw["kinematic"] = _as_np(kinematic_residual(model, front, mass_flux=flux))
    if wanted & {"edge_p_liquid", "edge_p_vapour"}:
        raw["edge_p_liquid"] = _as_np(model.liquid_pressure(front.points))
        raw["edge_p_vapour"] = _as_np(model.p_vapor(front.points[:, 2:3]))
    if wanted & {"yl_jump", "jump_px"}:
        raw["yl_jump"] = _as_np(laplace_jump_residual(model, front, groups))
    if "kappa_implied" in wanted:
        raw["kappa_implied"] = _as_np(pressure_implied_curvature(model, front, groups))
    if "shear" in wanted:
        raw["shear"] = _as_np(interface_shear_residual(model, front, groups))
    if wanted & {"flux_g", "mass_flux"}:
        g = model.interface_flux_at(front)
        raw["flux_g"] = _as_np(g)
        raw["mass_flux"] = _as_np(groups["mass_flux_to_speed"] * g)
    return raw


def _arclength(scene: Scene, t_star: float, xy_um: np.ndarray) -> np.ndarray:
    rings = geometry.interface(scene, t_star)["rings_um"]
    if not rings:
        return np.full(len(xy_um), np.nan)
    y_center = 0.5 * float(scene.data.y_lo + scene.data.y_hi) * scene.scales.length_um
    return geometry.project_arclength(xy_um, np.asarray(rings[0]), y_center)


def _jump_pixels(scene: Scene, jump: np.ndarray, kappa_par: np.ndarray) -> np.ndarray:
    data = scene.data
    pixel = float(data.x[1] - data.x[0])
    width = float(data.y_hi - data.y_lo)
    return jump_shape_pixels(jump, kappa_par, scene.groups, pixel, width)
