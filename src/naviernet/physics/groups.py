"""Dimensionless groups derived from the experiment, fluid, and scales.

Every group here is *computed*, never hard-coded. Change the flow rate, the
channel geometry, or the working fluid in the config and each group -- and
therefore each PDE coefficient that uses it -- updates consistently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

GRAVITY = 9.81  # m/s^2

R_UNIVERSAL = 8.314  # J/mol/K

# Accommodation coefficient in the Schrage kinetic resistance, boiling-calibrated
# (Marek & Straub 2001 survey: measured values span 1e-3 to 1, with boiling data
# clustering near this). The single largest uncertainty in any thermal closure --
# a stated literature choice, deliberately not a per-run tunable.
ACCOMMODATION = 0.035

# Wall roughness below which a film station counts as DRY: the film has merged
# into the surface finish and there is no continuous liquid layer left to
# evaporate. Typical microchannel wall finish, ~0.1 um.
FILM_DRYOUT_ROUGHNESS_M = 1e-7

# The dimensionless groups a joint (transfer-learning) model is conditioned on:
# the regime descriptors that distinguish one dataset's heat-flux condition from
# another's, spanning momentum, heat transfer, property ratios, and confinement.
# Dimensional reference quantities (Dh_um, U_in_m_s, t_ref_ms, dT_ref) are units,
# not regime, so they are excluded. The order is the conditioning-vector layout
# and is part of the checkpoint contract -- append, never reorder.
CONDITIONING_GROUPS = (
    "Re",
    "We",
    "Ca",
    "Bond",
    "Pr",
    "Ja",
    "rho_ratio",
    "mu_ratio",
    "hele_shaw",
    "q_wall_star",
    "t_star_per_frame",
)
N_COND = len(CONDITIONING_GROUPS)


def conditioning_vector(groups: dict[str, float]) -> list[float]:
    """Log10-normalised conditioning vector from a dataset's dimensionless groups.

    In :data:`CONDITIONING_GROUPS` order. Log scale compresses the several-orders-
    of-magnitude spread (Re~1e2, Ca~1e-2) into O(1) inputs the network conditions
    on; the tiny floor guards a non-positive group from breaking the log.
    """
    return [math.log10(max(float(groups[key]), 1e-12)) for key in CONDITIONING_GROUPS]


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

    # Stage-B energy closures.
    # Conduction superheat over the half-height: the temperature scale the wall
    # flux sets by pure conduction across the GAP, and the reference for the
    # superheat theta = (T - T_sat) / dT_ref.
    #
    # NB it does NOT make the depth-averaged wall source O(1), and an earlier
    # comment here claimed it did. Substituting dT_ref below collapses
    # q_wall_star to 2 (L/h)^2 / Pe -- so at Pe ~ 2e3 it is ~4e-3, and that is
    # physics, not a scaling slip: the liquid advects through far faster than the
    # wall can heat it. The consequence is worth knowing before trusting the
    # energy equation to determine anything. A CONSTANT theta kills every other
    # term in it (theta_t, u.grad theta, lap theta all vanish; at saturation the
    # evaporation sink vanishes too), leaving a residual of just q_wall_star --
    # measured, a trained run sits at energy = 1.9e-5 against q_wall_star^2 =
    # 1.6e-5. The energy equation alone cannot pin the temperature; only the
    # evaporation mass closure can, because the source it ties theta to is
    # observed through the bubble's growth.
    q_wall = exp.q_wall_W_cm2 * 1e4  # W/cm^2 -> W/m^2
    groups["dT_ref"] = q_wall * (h / 2.0) / fluid.k_l
    # Stefan/Jakob number at the actual superheat (generalises Ja_per_5K).
    groups["Ja"] = fluid.cp_l * groups["dT_ref"] / fluid.h_lv
    # Depth-averaged wall heat source in the non-dimensional energy equation.
    groups["q_wall_star"] = (
        q_wall / (fluid.rho_l * fluid.cp_l * u * groups["dT_ref"]) * (length / h)
    )

    # Property ratios entering the mixture rules
    groups["rho_ratio"] = fluid.rho_l / fluid.rho_v
    groups["mu_ratio"] = fluid.mu_l / fluid.mu_v

    # Confinement closures
    # Depth-averaged Hele-Shaw drag coefficient for a channel of height H.
    groups["hele_shaw"] = 12.0 * (length / h) ** 2 / groups["Re"]
    # The channel gap, non-dimensionalised: the length the OUT-OF-PLANE (gap)
    # interface curvature 2/H* is set by. A depth-averaged model has no z
    # direction, so this curvature has to be supplied rather than computed --
    # and it is the larger of the two principal curvatures here.
    groups["H_star"] = h / length
    # Bretherton lubrication film left behind an advancing meniscus.
    groups["bretherton_film_um"] = 1.34 * groups["Ca"] ** (2.0 / 3.0) * (h / 2) * 1e6

    # Liquid-film closures (model.liquid_film).
    # The Schrage kinetic resistance at the film's surface, expressed as the
    # EQUIVALENT LIQUID LENGTH k_l * R_i and non-dimensionalised on L_ref. In
    # series with the film's own conduction it is what keeps the evaporative
    # flux bounded as delta -> 0 -- the current |grad alpha| source has no such
    # bound -- and it is genuinely fluid-dependent (~1.1 um FC-72, ~2.4 um
    # water), unlike the kinematic viscosity.
    t_sat_k = fluid.T_sat_C + 273.15
    r_specific = R_UNIVERSAL / fluid.molar_mass
    r_interface = (
        (2.0 - ACCOMMODATION)
        / (2.0 * ACCOMMODATION)
        * t_sat_k
        * math.sqrt(2.0 * math.pi * r_specific * t_sat_k)
        / (fluid.rho_v * fluid.h_lv**2)
    )
    groups["R_gamma_star"] = fluid.k_l * r_interface / length
    # The depletion number: d(delta*)/dt* = -film_depletion * theta / (delta* +
    # R_gamma*). Algebraically exactly Ja/Pe -- conduction across the film per
    # latent heat, on the convective clock.
    groups["film_depletion"] = groups["Ja"] / groups["Pe"]
    groups["film_dryout_star"] = FILM_DRYOUT_ROUGHNESS_M / length

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
