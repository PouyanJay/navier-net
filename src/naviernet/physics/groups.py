"""Dimensionless groups derived from the experiment, fluid, and scales.

Every group here is *computed*, never hard-coded. Change the flow rate, the
channel geometry, or the working fluid in the config and each group -- and
therefore each PDE coefficient that uses it -- updates consistently.
"""

from __future__ import annotations

import json
from pathlib import Path

GRAVITY = 9.81  # m/s^2


def reference_time_ms(scales) -> float:
    """Convective reference time L_ref / U_ref, in milliseconds."""
    return scales.L_ref_um * 1e-6 / scales.U_ref * 1e3


def hydraulic_diameter_m(experiment) -> float:
    """Hydraulic diameter of the rectangular channel, in metres."""
    w = experiment.channel_width_um * 1e-6
    h = experiment.channel_height_um * 1e-6
    return 4 * w * h / (2 * (w + h))


def inlet_velocity_m_s(experiment) -> float:
    """Mean liquid inlet velocity from the volumetric flow rate, in m/s."""
    w = experiment.channel_width_um * 1e-6
    h = experiment.channel_height_um * 1e-6
    q = experiment.flow_rate_mL_hr * 1e-6 / 3600.0  # mL/hr -> m^3/s
    return q / (w * h)


def compute_groups(cfg) -> dict[str, float]:
    """All dimensionless groups and reference quantities for a config."""
    exp, fluid, scales = cfg.experiment, cfg.fluid, cfg.scales

    d_h = hydraulic_diameter_m(exp)
    u_in = inlet_velocity_m_s(exp)
    h = exp.channel_height_um * 1e-6
    length = scales.L_ref_um * 1e-6
    u = scales.U_ref
    t_ref = reference_time_ms(scales)

    groups: dict[str, float] = {}

    # Reference quantities
    groups["Dh_um"] = d_h * 1e6
    groups["U_in_m_s"] = u_in
    groups["u_inlet_star"] = u_in / u
    groups["t_ref_ms"] = t_ref
    groups["t_star_per_frame"] = exp.dt_frame_ms / t_ref

    # Momentum
    groups["Re"] = fluid.rho_l * u * length / fluid.mu_l
    groups["Re_in"] = fluid.rho_l * u_in * d_h / fluid.mu_l
    groups["We"] = fluid.rho_l * u**2 * length / fluid.sigma
    groups["Ca"] = fluid.mu_l * u / fluid.sigma
    groups["Bond"] = (fluid.rho_l - fluid.rho_v) * GRAVITY * d_h**2 / fluid.sigma

    # Heat transfer
    groups["Pr"] = fluid.cp_l * fluid.mu_l / fluid.k_l
    groups["Pe"] = groups["Re"] * groups["Pr"]
    groups["Ja_per_5K"] = fluid.cp_l * 5.0 / fluid.h_lv

    # Property ratios entering the mixture rules
    groups["rho_ratio"] = fluid.rho_l / fluid.rho_v
    groups["mu_ratio"] = fluid.mu_l / fluid.mu_v

    # Confinement closures
    # Depth-averaged Hele-Shaw drag coefficient for a channel of height H.
    groups["hele_shaw"] = 12.0 * (length / h) ** 2 / groups["Re"]
    # Bretherton lubrication film left behind an advancing meniscus.
    groups["bretherton_film_um"] = 1.34 * groups["Ca"] ** (2.0 / 3.0) * (h / 2) * 1e6

    return groups


def save_groups(cfg, destination: Path) -> dict[str, float]:
    """Compute the groups and record them with the inputs they came from."""
    from omegaconf import OmegaConf

    groups = compute_groups(cfg)
    payload = {
        "experiment": OmegaConf.to_container(cfg.experiment, resolve=True),
        "fluid": OmegaConf.to_container(cfg.fluid, resolve=True),
        "scales": OmegaConf.to_container(cfg.scales, resolve=True),
        "groups": groups,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    return groups
