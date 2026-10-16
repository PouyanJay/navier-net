"""Wall heating and the growth it has to pay for.

One wall flux, the series' own, split by phase. Under liquid it heats the
liquid (the energy equation's ``(1 - alpha)`` gate); under the bubble it
evaporates the liquid film the bubble leaves on the heated wall, over the wet
share of the footprint (``model.footprint_evaporation``, the source on the
model). This module holds the energy-balance bookkeeping the trainer's log, the
evaluation report and the audit read: what the measured growth needs against
what the wall flux puts under the footprint.
"""

from __future__ import annotations

import numpy as np


def measured_areas_m2(data, cfg) -> np.ndarray:
    """The projected vapour area of every event frame, in square metres."""
    length = cfg.scales.L_ref_um * 1e-6
    pixel_area = float(data.x[1] - data.x[0]) * float(data.y[1] - data.y[0]) * length**2
    return np.array(
        [
            float(((data.alpha[i] > 0.5) & (data.valid[i] > 0)).sum()) * pixel_area
            for i in range(data.n_event)
        ]
    )


def growth_budget(data, cfg, groups: dict[str, float]) -> dict[str, float | list[float]]:
    """The energy balance of the measured growth, frame by frame.

    Per event frame: the latent power the measured projected growth needs
    (``rho_v h dA/dt h_lv``, the bubble taken to fill the depth), the power the
    wall flux puts under the footprint (``q'' A``), and the wet fraction that
    balance implies, ``min(1, needed / available)``. The implied wet fraction is
    the fit's natural start and the reference its fitted value is reported
    against. It ignores the outline's own conduction flux, which also creates
    vapour, so it is an upper bound on what the film has to supply, not the
    answer.
    """
    fluid, exp = cfg.fluid, cfg.experiment
    depth = exp.channel_height_um * 1e-6
    t_ref = groups["t_ref_ms"] * 1e-3
    areas = measured_areas_m2(data, cfg)
    times = np.array([float(data.t[i]) for i in range(data.n_event)]) * t_ref
    growth = np.gradient(areas, times)
    latent_power = fluid.rho_v * growth * depth * fluid.h_lv
    footprint_power = float(exp.q_wall_W_cm2) * 1e4 * areas
    implied = np.minimum(1.0, latent_power / np.maximum(footprint_power, 1e-12))
    return {
        "latent_power_mW": (latent_power * 1e3).tolist(),
        "footprint_power_mW": (footprint_power * 1e3).tolist(),
        "implied_wet_fraction": implied.tolist(),
        "latent_power_mW_mean": float(latent_power.mean()) * 1e3,
        "latent_power_mW_max": float(latent_power.max()) * 1e3,
        "footprint_power_mW_mean": float(footprint_power.mean()) * 1e3,
        "wet_fraction_estimate": float(implied.mean()),
    }


def budget_summary(budget: dict) -> dict[str, float]:
    """The scalar entries of :func:`growth_budget`, for a log line."""
    return {key: value for key, value in budget.items() if isinstance(value, float)}
