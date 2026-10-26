"""Each equation's error relative to the terms that should balance.

An absolute residual reads differently in different places: at the bubble edge
the temperature equation's terms are of order 5 (the liquid drops 1.4 K over
about 12 um), in the open liquid of order 0.1, so one threshold is lax in the
first and strict in the second. Here every equation's residual is divided by
the sum of its terms' magnitudes, point by point, and the two are reduced as
``rms(residual) / rms(sum |terms|)`` over each region: 10% means the terms
balance to within a tenth of their size there.

The regions are the liquid within one layer length of the interface, between
one and three, and beyond three (the layer is the temperature gate's length,
``model.theta_gate_length``), all of the liquid, and for continuity the vapour.

The points are a regular grid and, optionally, as many again drawn at random
across the channel. A feature thinner than the grid's spacing can fall between
its lines at every frame: on scale-relmulti4-s0 a spike at the pinned root held
the temperature equation at 100% of its terms, read as 11% on the grid alone.
"""

from __future__ import annotations

import numpy as np
import torch

from naviernet.physics.heating import wall_flux_ratio
from naviernet.physics.residuals import (
    DEPTH_PROFILE_CONVECTION,
    _laplacian,
    confinement_drag,
    gradients,
    liquid_pressure,
    mixture,
)

# Points per evaluation batch: second derivatives of every field are taken.
CHUNK = 2048
# The layer length when the model has no temperature gate to take it from.
DEFAULT_LAYER = 0.04

EQUATIONS = ("temperature", "momentum", "continuity", "edge transport")


def _region_masks(liquid_depth: np.ndarray, layer: float) -> dict[str, np.ndarray]:
    return {
        "near edge": (liquid_depth > 0) & (liquid_depth < layer),
        "edge band": (liquid_depth >= layer) & (liquid_depth < 3 * layer),
        "open liquid": liquid_depth >= 3 * layer,
        "all liquid": liquid_depth > 0,
    }


def _distance_field(model, x: torch.Tensor) -> torch.Tensor:
    """``phi``, positive in the vapour: the frozen explicit geometry when there
    is one (its zero is the interface the conditions sit on)."""
    if getattr(model, "front_geometry", False):
        return model._phi_with_frozen_geometry(x, None)
    return model.phi(x)


def _balance(terms: list[torch.Tensor], weight) -> tuple[torch.Tensor, torch.Tensor]:
    """The residual (the terms' sum) and the terms' summed magnitudes."""
    return (weight * sum(terms)).detach(), (weight * sum(t.abs() for t in terms)).detach()


def _terms(model, x: torch.Tensor, groups: dict, inertia: bool) -> dict[str, tuple]:
    """``{equation: (residual, scale)}`` at the points ``x``, for the equations
    this model has."""
    alpha = model.alpha(x)
    u, v = model.velocity(x)
    u_x, u_y, u_t = gradients(u, x)
    v_x, v_y, v_t = gradients(v, x)
    a_x, a_y, a_t = gradients(alpha, x)
    liquid = (1.0 - alpha).detach()
    # Under model.exact_continuity the liquid conserves volume by construction
    # and continuity is the vapour's alone, as the trainer scores it.
    scored = (~model.liquid_side(x)).unsqueeze(1).float() if model.exact_liquid else 1.0
    out = {
        "continuity": _balance([u_x, v_y, -model.source(x)], scored),
        "edge transport": _balance([a_t, u * a_x, v * a_y], 1.0),
    }
    if "T" in model.fields:
        theta = model.temperature(x)
        t_x, t_y, t_t = gradients(theta, x)
        heat = groups["q_wall_star"] * wall_flux_ratio(x, groups) * (1.0 - alpha)
        out["temperature"] = _balance(
            [t_t, u * t_x + v * t_y, -_laplacian(theta, x) / groups["Pe"], -heat], liquid
        )
    if "p" in model.fields:
        p_x, p_y, _ = gradients(liquid_pressure(model, x, None), x)
        drag = confinement_drag(alpha, groups)
        viscosity = mixture(alpha, groups["mu_ratio"]) / groups["Re"]
        along = [p_x, drag * u, -viscosity * _laplacian(u, x)]
        across = [p_y, drag * v, -viscosity * _laplacian(v, x)]
        if inertia:
            density = mixture(alpha, groups["rho_ratio"])
            along += [density * (u_t + DEPTH_PROFILE_CONVECTION * (u * u_x + v * u_y))]
            across += [density * (v_t + DEPTH_PROFILE_CONVECTION * (u * v_x + v * v_y))]
        r_x, s_x = _balance(along, liquid)
        r_y, s_y = _balance(across, liquid)
        out["momentum"] = (torch.hypot(r_x, r_y), torch.hypot(s_x, s_y))
    return out


def _points(data, t: float, nx: int, ny: int, n_random: int, seed: int) -> np.ndarray:
    """An ``nx`` by ``ny`` grid across the channel at ``t``, plus ``n_random``
    points drawn uniformly over it from ``seed``."""
    xs = np.linspace(data.domain.x_min, data.domain.x_max, nx)
    ys = np.linspace(data.y_lo, data.y_hi, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    plane = [np.column_stack([grid_x.ravel(), grid_y.ravel()])]
    if n_random > 0:
        rng = np.random.default_rng(seed)
        low = [data.domain.x_min, data.y_lo]
        high = [data.domain.x_max, data.y_hi]
        plane.append(rng.uniform(low, high, size=(n_random, 2)))
    plane = np.concatenate(plane)
    return np.column_stack([plane, np.full(len(plane), t)])


def relative_errors(
    model,
    groups: dict,
    data,
    t: float,
    nx: int,
    ny: int,
    n_random: int = 0,
    seed: int = 0,
) -> dict:
    """``{equation: {region: relative error}}`` at instant ``t`` on an ``nx`` by
    ``ny`` grid across the channel and ``n_random`` random points drawn from
    ``seed``; also the layer length used, in L_ref."""
    points = _points(data, t, nx, ny, n_random, seed)
    device = next(model.parameters()).device
    inertia = bool(getattr(model.cfg.model, "momentum_inertia", False))
    parts: dict[str, list] = {}
    depth = []
    with torch.enable_grad():
        for start in range(0, len(points), CHUNK):
            x = torch.tensor(
                points[start : start + CHUNK], dtype=torch.float32, device=device
            ).requires_grad_(True)
            for name, (residual, scale) in _terms(model, x, groups, inertia).items():
                parts.setdefault(name, []).append((residual.cpu(), scale.cpu()))
            depth.append(-_distance_field(model, x).detach().cpu())
    liquid_depth = torch.cat(depth).numpy().ravel()
    layer = float(getattr(model, "theta_gate_length", DEFAULT_LAYER))
    regions = _region_masks(liquid_depth, layer)
    result: dict = {"layer": layer}
    for name in EQUATIONS:
        if name not in parts:
            continue
        residual = torch.cat([r for r, _ in parts[name]]).numpy().ravel()
        scale = torch.cat([s for _, s in parts[name]]).numpy().ravel()
        masks = dict(regions)
        if name == "continuity":
            masks["vapour"] = liquid_depth < 0
        result[name] = {
            region: float(
                np.sqrt((residual[m] ** 2).mean()) / max(np.sqrt((scale[m] ** 2).mean()), 1e-12)
            )
            for region, m in masks.items()
            if m.any()
        }
    return result
