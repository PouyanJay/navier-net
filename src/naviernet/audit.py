"""Acceptance audit of a trained run: the tests a hidden field must pass before
it is trusted (docs/plans/trustworthy-hidden-fields-plan.md, WP6).

The 2026-10 audit of the published checkpoint found hidden fields that looked
plausible and were not solutions of anything: lopsided about a symmetric
channel, different for every random seed, with the governing equations
unenforced after the first half millisecond and an inlet flux that vanished
within one channel width. Each of those failures is now a test, run on every
evaluated checkpoint at every event frame:

1. **Mirror test.** The RMS of each field's symmetry-breaking part about the
   channel centreline (``v`` is odd there, everything else even) over the RMS
   of its variation.
2. **Residuals at every frame.** The RMS of every active collocation equation on
   a uniform grid, with the per-time-chunk table training logged alongside.
3. **Flux balance.** The axial volume flux against the inlet flux plus the
   vapour created upstream, at every station.
4. **Heat budget.** The latent power the source implies, the latent power the
   interface flux implies, the latent power the measured growth needs, and the
   heaters' input, all in milliwatts.
5. **Interface conditions** per side of the front.

The seed-ensemble test needs several runs and lives in
``scripts/audit_ensemble.py``, which reads the field snapshots this module
writes beside the report. Thresholds come from ``cfg.evaluation.acceptance``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from naviernet.physics import heating, mass_balance, registry
from naviernet.physics.groups import compute_groups
from naviernet.physics.residuals import (
    interface_mass_flux,
    kinematic_residual,
    laplace_jump_residual,
)
from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths

log = get_logger(__name__)

FIELDS = ("alpha", "u", "v", "p", "T", "s")
# A mirror-symmetric flow has an odd transverse velocity about the centreline;
# every other field is even.
ODD_FIELDS = frozenset({"v"})
CHUNK = 1024


@dataclass(frozen=True)
class AuditGrid:
    """A uniform grid whose rows are symmetric about the channel centreline, so
    the mirror image of row ``i`` is row ``-1 - i``."""

    xs: np.ndarray
    ys: np.ndarray
    y_center: float


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def mirror_breaking_fraction(field: np.ndarray, odd: bool = False) -> float:
    """RMS of the symmetry-breaking part over the RMS of the field's variation,
    for a field on a grid whose rows are symmetric about the axis."""
    mirrored = field[::-1, :]
    even, odd_part = 0.5 * (field + mirrored), 0.5 * (field - mirrored)
    # For an odd field a mean offset is itself symmetry-breaking, so the whole
    # field is the reference; for an even one only its variation can break.
    breaking, reference = (even, field) if odd else (odd_part, field - field.mean())
    return _rms(breaking) / max(_rms(reference), 1e-12)


def audit_grid(data, nx: int, ny: int) -> AuditGrid:
    """``nx`` stations across the domain and ``ny`` rows centred on the channel
    centreline of the run's wall span, reaching the nearer wall."""
    y_center = 0.5 * (data.y_lo + data.y_hi)
    half = min(y_center - data.y_lo, data.y_hi - y_center)
    return AuditGrid(
        xs=np.linspace(data.domain.x_min, data.domain.x_max, nx),
        ys=y_center + np.linspace(-half, half, ny),
        y_center=float(y_center),
    )


def _points(grid: AuditGrid, t: float) -> np.ndarray:
    grid_x, grid_y = np.meshgrid(grid.xs, grid.ys)
    return np.column_stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, t)]).astype(
        np.float32
    )


def fields_on_grid(model, grid: AuditGrid, t: float) -> dict[str, np.ndarray]:
    """Every field the model carries, as ``(ny, nx)`` arrays at one instant.

    Evaluated with autograd on: a thermal source differentiates the temperature
    at the front to get its flux, so it cannot be read under ``no_grad``.
    """
    points = torch.tensor(_points(grid, t))
    shape = (len(grid.ys), len(grid.xs))
    with torch.enable_grad():
        u, v = model.velocity(points)
        out = {"alpha": model.alpha(points), "u": u, "v": v, "s": model.source(points)}
        if "p" in model.fields:
            out["p"] = model.pressure(points)
        if "T" in model.fields:
            out["T"] = model.temperature(points)
    return {name: value.detach().cpu().numpy().reshape(shape) for name, value in out.items()}


def residual_rms_on_grid(
    model, groups, equations, grid: AuditGrid, t: float
) -> dict[str, float]:
    """RMS of each collocation equation's residual over the grid at ``t``."""
    points = _points(grid, t)
    sums = dict.fromkeys((eq.id for eq in equations), 0.0)
    with torch.enable_grad():
        for start in range(0, len(points), CHUNK):
            chunk = torch.tensor(points[start : start + CHUNK], requires_grad=True)
            ctx = registry.LossContext(model, chunk, groups=groups)
            for eq in equations:
                sums[eq.id] += float(eq.pointwise(ctx).detach().sum())
    return {name: float(np.sqrt(total / len(points))) for name, total in sums.items()}


def flux_balance(cfg, model, data, groups, t: float) -> dict:
    """The normalised axial-flux defect at every station at ``t``, and its worst
    value upstream of the root, where the flux must simply equal the inlet's."""
    plan = mass_balance.build_plan(
        data.domain,
        (data.y_lo, data.y_hi),
        groups,
        cfg.training,
        "cpu",
        at_times=torch.tensor([t], dtype=torch.float32),
    )
    with torch.enable_grad():  # the thermal source needs autograd for its flux
        defect = mass_balance.axial_flux_defect(model, plan).detach().cpu().numpy()[0]
    stations = plan.stations.cpu().numpy()
    upstream = stations < float(data.pin_anchor[0])
    return {
        "stations": stations.tolist(),
        "defect": defect.tolist(),
        "max_abs": float(np.abs(defect).max()),
        "upstream_max_abs": float(np.abs(defect[upstream]).max()) if upstream.any() else None,
    }


def _integral(values: np.ndarray, grid: AuditGrid) -> float:
    return float(np.trapezoid(np.trapezoid(values, grid.xs, axis=1), grid.ys))


def _measured_growth_rate(data, row: int) -> float:
    """``dA/dt`` of the measured vapour area at ``row``, in ``L_ref^2 / t_ref``."""
    pixel = float(data.x[1] - data.x[0]) * float(data.y[1] - data.y[0])
    areas = {
        r: float(((data.alpha[r] > 0.5) & (data.valid[r] > 0)).sum()) * pixel
        for r in (row - 1, row, row + 1)
        if 0 <= r < data.n_event
    }
    rows = sorted(areas)
    first, last = rows[0], rows[-1]
    return (areas[last] - areas[first]) / (float(data.t[last]) - float(data.t[first]))


def heat_budget(cfg, model, data, groups, row: int, fields: dict, grid: AuditGrid) -> dict:
    """Latent power implied by the source, by the interface flux (sharp thermal
    model), and needed by the measured growth; the heaters' input into the
    liquid; and the spread between the latent estimates. All in milliwatts."""
    fluid, exp = cfg.fluid, cfg.experiment
    length = cfg.scales.L_ref_um * 1e-6
    depth = exp.channel_height_um * 1e-6
    speed = cfg.scales.U_ref
    t_ref = length / speed
    liquid = 1.0 - fields["alpha"]

    specific_volume = 1.0 / fluid.rho_v - 1.0 / fluid.rho_l
    mass_from_source = _integral(fields["s"], grid) * length * depth * speed / specific_volume
    growth = _measured_growth_rate(data, row) * length**2 / t_ref
    scale = model.q_wall_scale
    scale = float(scale) if isinstance(scale, float) else float(scale.detach())
    background = float(exp.q_wall_W_cm2) * 1e4 * scale * _integral(liquid, grid) * length**2
    budget = {
        "latent_from_source_mW": mass_from_source * fluid.h_lv * 1e3,
        "latent_for_growth_mW": fluid.rho_v * growth * depth * fluid.h_lv * 1e3,
        "background_heater_mW": background * 1e3,
    }
    if "q_cavity_star" in groups and model.heater_centre is not None:
        points = torch.tensor(_points(grid, float(data.t[row])))
        disk = heating.heater_footprint(points, groups, model.heater_centre).numpy()
        disk = disk.reshape(len(grid.ys), len(grid.xs))
        cavity = float(exp.q_cavity_W_cm2) * 1e4 * _integral(disk * liquid, grid) * length**2
        budget["cavity_heater_mW"] = cavity * 1e3
    if getattr(model, "thermal_model", "legacy") == "sharp":
        flux, weights, _ = model._front_flux(torch.tensor(float(data.t[row])), None)
        along_front = float((flux * weights).sum().detach())
        budget["latent_from_flux_mW"] = fluid.k_l * groups["dT_ref"] * depth * along_front * 1e3
    latent = [budget[k] for k in budget if k.startswith("latent")]
    reference = max(abs(budget["latent_for_growth_mW"]), 1e-9)
    budget["spread"] = float((max(latent) - min(latent)) / reference)
    return budget


def front_conditions(model, groups, t: float) -> dict | None:
    """Kinematic and Young-Laplace residual RMS on the upper and lower halves of
    the explicit front, or ``None`` for a run without one."""
    if not getattr(model, "front_geometry", False) or not getattr(
        model, "sharp_interface", False
    ):
        return None
    model_cfg = model.cfg.model
    with torch.enable_grad():
        front = model.front(
            torch.tensor([[t]], dtype=torch.float32),
            int(model_cfg.front_body_samples),
            int(model_cfg.front_cap_samples),
        )
        mass_flux = (
            interface_mass_flux(model, front, groups)
            if getattr(model, "thermal_model", "legacy") == "sharp"
            else None
        )
        kinematic = (
            kinematic_residual(model, front, mass_flux=mass_flux).detach().numpy().ravel()
        )
        laplace = laplace_jump_residual(model, front, groups).detach().numpy().ravel()
    upper = front.side.detach().numpy().ravel() > 0
    return {
        "kinematic_rms_upper": _rms(kinematic[upper]),
        "kinematic_rms_lower": _rms(kinematic[~upper]),
        "laplace_rms_upper": _rms(laplace[upper]),
        "laplace_rms_lower": _rms(laplace[~upper]),
    }


def audit_frame(
    cfg, model, data, groups, equations, grid: AuditGrid, row: int
) -> tuple[dict, dict]:
    """One event frame's audit record and its field snapshot."""
    t = float(data.t[row])
    fields = fields_on_grid(model, grid, t)
    record = {
        "frame": int(data.frame_numbers[row]),
        "t_star": t,
        "mirror": {
            name: mirror_breaking_fraction(values, odd=name in ODD_FIELDS)
            for name, values in fields.items()
        },
        "residual_rms": residual_rms_on_grid(model, groups, equations, grid, t),
        "flux": flux_balance(cfg, model, data, groups, t),
        "heat": heat_budget(cfg, model, data, groups, row, fields, grid),
        "front": front_conditions(model, groups, t),
    }
    return record, fields


def _verdicts(frames: list[dict], acceptance) -> dict:
    """Pass/fail against the configured thresholds, with the worst value behind
    each verdict so a failure says by how much."""
    hidden = [name for name in FIELDS if name != "alpha"]
    mirror_worst = max(
        f["mirror"][name] for f in frames for name in hidden if name in f["mirror"]
    )
    residual_worst = max((v for f in frames for v in f["residual_rms"].values()), default=0.0)
    flux_worst = max(f["flux"]["max_abs"] for f in frames)
    heat_worst = max(f["heat"]["spread"] for f in frames)
    return {
        "mirror": {
            "worst": mirror_worst,
            "max": acceptance.mirror_max,
            "pass": mirror_worst <= acceptance.mirror_max,
        },
        "residuals": {
            "worst": residual_worst,
            "max": acceptance.residual_rms_max,
            "pass": residual_worst <= acceptance.residual_rms_max,
        },
        "flux": {
            "worst": flux_worst,
            "max": acceptance.flux_tol,
            "pass": flux_worst <= acceptance.flux_tol,
        },
        "heat": {
            "worst": heat_worst,
            "max": acceptance.heat_budget_tol,
            "pass": heat_worst <= acceptance.heat_budget_tol,
        },
    }


def audit_run(cfg, model, data, paths: RunPaths) -> dict:
    """Audit every event frame, write ``audit.json`` and the field snapshots
    (``audit_fields.npz``) the seed-ensemble test reads, and return the report."""
    evaluation = cfg.evaluation
    groups = compute_groups(cfg)
    equations = registry.collocation_equations(
        registry.enabled_equations(cfg.model.fields, **registry.flags_from(cfg.model))
    )
    grid = audit_grid(data, int(evaluation.audit_nx), int(evaluation.audit_ny))
    frames, snapshots = [], {"xs": grid.xs, "ys": grid.ys}
    for row in range(data.n_event):
        record, fields = audit_frame(cfg, model, data, groups, equations, grid, row)
        frames.append(record)
        for name, values in fields.items():
            snapshots[f"{record['frame']}/{name}"] = values
    report = {
        "run_name": cfg.run_name,
        "dataset": cfg.dataset,
        "y_center": grid.y_center,
        "thresholds": {
            "seed_corr_min": float(evaluation.acceptance.seed_corr_min),
            "mirror_max": float(evaluation.acceptance.mirror_max),
            "residual_rms_max": float(evaluation.acceptance.residual_rms_max),
            "flux_tol": float(evaluation.acceptance.flux_tol),
            "heat_budget_tol": float(evaluation.acceptance.heat_budget_tol),
        },
        "verdicts": _verdicts(frames, evaluation.acceptance),
        "frames": frames,
    }
    paths.audit_json.write_text(json.dumps(report, indent=2))
    np.savez_compressed(paths.audit_fields_npz, **snapshots)
    log.info(
        "audit: %s",
        " ".join(
            f"{name}={'pass' if v['pass'] else 'FAIL'}({v['worst']:.3g})"
            for name, v in report["verdicts"].items()
        ),
    )
    return report
