"""The conversions from model units to physical units, all derived from the
run's own config and dimensionless groups. No literal constant lives anywhere
else in the service: a quantity's unit factor is a named attribute here."""

from __future__ import annotations

from dataclasses import asdict, dataclass

ML_PER_M3 = 1e6
SECONDS_PER_HOUR = 3600.0
UM_PER_M = 1e6
MM_PER_M = 1e3
UM_PER_MM = 1e3
MS_PER_S = 1e3


@dataclass(frozen=True)
class Scales:
    """Multiply a model-unit value by the attribute to get the physical unit."""

    length_um: float  # x*, y*, phi*  -> um
    time_ms: float  # t*             -> ms
    speed_mm_s: float  # u*, v*         -> mm/s (equal to um/ms)
    pressure_Pa: float  # p*             -> Pa  (rho_l U^2)
    rate_per_s: float  # s*, div, vorticity, strain -> 1/s (U/L)
    grad_p_Pa_per_mm: float  # dp*/dx*      -> Pa/mm
    curvature_per_um: float  # kappa*       -> 1/um
    superheat_K: float  # theta          -> K above saturation (dT_ref)
    saturation_C: float  # added to the superheat in C
    temperature_grad_K_per_mm: float  # dtheta/dx*  -> K/mm
    shear_Pa: float  # mu_l du*/dy*   -> Pa
    flux_mL_h: float  # Q* = int u* dy* -> mL/h
    heat_flux_W_m2: float  # g = dtheta/dn  -> W/m^2
    energy_rate_K_s: float  # energy residual -> K/s
    area_mm2: float  # area* -> mm^2
    q_in_star: float  # the inlet volume flux in model units over the wall span

    @classmethod
    def from_run(cls, cfg, groups: dict[str, float], data) -> Scales:
        length_um = float(cfg.scales.L_ref_um)
        length_m = length_um / UM_PER_M
        speed = float(cfg.scales.U_ref)
        t_ref_ms = float(groups["t_ref_ms"])
        depth_m = float(cfg.experiment.channel_height_um) / UM_PER_M
        rho_l, mu_l, k_l = float(cfg.fluid.rho_l), float(cfg.fluid.mu_l), float(cfg.fluid.k_l)
        d_t = float(groups["dT_ref"])
        pressure = rho_l * speed**2
        rate = speed / length_m
        return cls(
            length_um=length_um,
            time_ms=t_ref_ms,
            speed_mm_s=speed * MM_PER_M,
            pressure_Pa=pressure,
            rate_per_s=rate,
            grad_p_Pa_per_mm=pressure / (length_um / UM_PER_MM),
            curvature_per_um=1.0 / length_um,
            superheat_K=d_t,
            saturation_C=float(cfg.experiment.T_sat_C),
            temperature_grad_K_per_mm=d_t / (length_um / UM_PER_MM),
            shear_Pa=mu_l * rate,
            flux_mL_h=speed * length_m * depth_m * ML_PER_M3 * SECONDS_PER_HOUR,
            heat_flux_W_m2=k_l * d_t / length_m,
            energy_rate_K_s=d_t / (t_ref_ms / MS_PER_S),
            area_mm2=(length_um / UM_PER_MM) ** 2,
            q_in_star=float(groups["u_inlet_star"]) * float(data.y_hi - data.y_lo),
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)
