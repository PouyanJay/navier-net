"""Wall heating: the cavity heater disk, the uniform background flux, and the
energy-balance estimate of the background flux the observed growth needs.

Two heaters, as in the experiment: a small disk around the nucleation cavity
at the series' flux, and a uniform flux along the whole channel whose value is
either given or fitted so the heat supplied keeps up with the growth. Both are
gated by liquid contact in the energy equation; this module only shapes them.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# The disk's edge is smoothed over this fraction of its radius: a numerical
# choice (a step is a poor target for a network), not a physical parameter.
HEATER_EDGE_FRACTION = 0.1


def heater_footprint(
    x: torch.Tensor, groups: dict[str, float], centre: tuple[float, float]
) -> torch.Tensor:
    """1 inside the cavity heater's disk, 0 outside (smoothly), ``(N, 1)``, and 0
    after ``heater_on_star`` when the heater is gated in time."""
    radius = groups["heater_radius_star"]
    edge = HEATER_EDGE_FRACTION * radius
    distance = torch.hypot(x[:, 0:1] - centre[0], x[:, 1:2] - centre[1])
    # A sigmoid edge integrates to pi (r^2 + pi^2 w^2 / 3), not pi r^2; the
    # rescale keeps the disk's power exactly the configured flux times its area.
    excess = (math.pi * edge) ** 2 / 3.0
    disk = torch.sigmoid((radius - distance) / edge) * (radius**2 / (radius**2 + excess))
    if "heater_on_star" in groups:
        disk = disk * (x[:, 2:3] <= groups["heater_on_star"]).to(x.dtype)
    return disk


def wall_heating(
    x: torch.Tensor,
    groups: dict[str, float],
    centre: tuple[float, float],
    background_scale,
) -> torch.Tensor:
    """``q*(x, y, t)`` before the liquid gate: the background flux times its
    (possibly fitted) scale, plus the cavity disk when the series has one."""
    heat = groups["q_wall_star"] * background_scale * torch.ones_like(x[:, :1])
    if "q_cavity_star" in groups:
        heat = heat + groups["q_cavity_star"] * heater_footprint(x, groups, centre)
    return heat


def estimate_background_flux(data, cfg, groups: dict[str, float]) -> dict[str, float]:
    """The uniform flux the observed growth needs, by a global energy balance.

    Latent power of the measured projected growth (``rho_v h dA/dt h_lv``, the
    bubble taken to fill the depth), minus the cavity disk's input, spread over
    the imaged heated footprint. It ignores the heat carried away by the liquid
    leaving the field of view, so it is a starting value for the fitted flux and
    a reference for its reported value, not the answer.
    """
    fluid, exp = cfg.fluid, cfg.experiment
    length = cfg.scales.L_ref_um * 1e-6
    depth = exp.channel_height_um * 1e-6
    t_ref = groups["t_ref_ms"] * 1e-3
    pixel_area = float(data.x[1] - data.x[0]) * float(data.y[1] - data.y[0]) * length**2
    rows = range(data.n_event)
    areas = np.array([float(((data.alpha[i] > 0.5) & (data.valid[i] > 0)).sum()) for i in rows])
    times = np.array([float(data.t[i]) for i in rows]) * t_ref
    growth = np.gradient(areas * pixel_area, times)
    latent_power = fluid.rho_v * growth * depth * fluid.h_lv
    disk_power = 0.0
    if "q_cavity_star" in groups:
        radius = float(exp.heater_radius_um) * 1e-6
        disk_power = float(exp.q_cavity_W_cm2) * 1e4 * np.pi * radius**2
    footprint = (float(data.x[-1] - data.x[0]) * length) * (exp.channel_width_um * 1e-6)
    needed = max(float(latent_power.mean()) - disk_power, 0.0)
    return {
        "latent_power_mW_mean": float(latent_power.mean()) * 1e3,
        "latent_power_mW_max": float(latent_power.max()) * 1e3,
        "cavity_power_mW": disk_power * 1e3,
        "heated_footprint_mm2": footprint * 1e6,
        "q_wall_estimate_W_cm2": needed / footprint / 1e4,
    }
