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

from naviernet.data.dataset import INTERFACE_ALPHA

# The step between the two heater zones is smoothed over this width (L_ref
# units, 1.5 um at the 300 um scale): a numerical choice so the network sees a
# differentiable source, far below any physical length here.
EDGE_SMOOTHING = 0.005


def _ratio(x, groups: dict[str, float], sigmoid):
    edge = groups.get("heater_edge_star")
    if edge is None:
        return None
    ratio = groups["q_upstream_ratio"]
    return ratio + (1.0 - ratio) * sigmoid((x - edge) / EDGE_SMOOTHING)


def wall_flux_ratio(x, groups: dict[str, float]):
    """The floor's local flux over ``q_wall`` at points ``x`` ``(N, 3)``, as an
    ``(N, 1)`` tensor: the upstream zone's ratio before the heater edge, 1
    beyond it; 1 everywhere when the floor carries one flux."""
    import torch

    ratio = _ratio(x[:, 0:1], groups, torch.sigmoid)
    return torch.ones_like(x[:, :1]) if ratio is None else ratio


def wall_flux_ratio_np(xs: np.ndarray, groups: dict[str, float]) -> np.ndarray:
    """:func:`wall_flux_ratio` at streamwise positions ``xs`` (L_ref units)."""
    ratio = _ratio(np.asarray(xs, dtype=float), groups, lambda z: 1.0 / (1.0 + np.exp(-z)))
    return np.ones_like(np.asarray(xs, dtype=float)) if ratio is None else ratio


def measured_areas_m2(data, cfg, weight: np.ndarray | None = None) -> np.ndarray:
    """The projected vapour area of every event frame, in square metres; with
    ``weight`` (one value per pixel column) the weighted area."""
    length = cfg.scales.L_ref_um * 1e-6
    pixel_area = float(data.x[1] - data.x[0]) * float(data.y[1] - data.y[0]) * length**2
    columns = np.ones(len(data.x)) if weight is None else weight
    return np.array(
        [
            float((((data.alpha[i] > INTERFACE_ALPHA) & (data.valid[i] > 0)) * columns).sum())
            * pixel_area
            for i in range(data.n_event)
        ]
    )


def growth_budget(data, cfg, groups: dict[str, float]) -> dict[str, float | list[float]]:
    """The energy balance of the measured growth, frame by frame.

    Per event frame: the latent power the measured projected growth needs
    (``rho_v h dA/dt h_lv``, the bubble taken to fill the depth), the power the
    wall flux puts under the footprint (the local flux over its area, both
    heater zones counted), and the wet fraction that
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
    heated = measured_areas_m2(data, cfg, wall_flux_ratio_np(data.x, groups))
    footprint_power = float(exp.q_wall_W_cm2) * 1e4 * heated
    # A frame that shrinks needs no film at all: the share is clipped to [0, 1].
    implied = np.clip(latent_power / np.maximum(footprint_power, 1e-12), 0.0, 1.0)
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
