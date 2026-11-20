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
4. **Heat budget.** The latent power the source implies (the footprint film
   source and the outline flux, and their sum), the latent power the measured
   growth needs, and the wall flux's input under the liquid and under the
   footprint, all in milliwatts.
5. **Interface conditions** per side of the front.
6. **Outline.** The IoU against the mask at every frame, held above a floor
   and within a tolerance of the warm-up's own fit (recorded in the checkpoint
   at the step the physics engaged), and the nose position against the
   measured nose. The images are the measurement: a run that fails here is
   rejected whatever its residuals say.

The seed-ensemble test needs several runs and lives in
``scripts/audit_ensemble.py``, which reads the field snapshots this module
writes beside the report. Thresholds come from ``cfg.evaluation.acceptance``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from naviernet.physics import depth_drain, mass_balance, registry
from naviernet.physics.groups import compute_groups
from naviernet.physics.growth_closure import measured_growth_rate
from naviernet.physics.heating import wall_flux_ratio_np
from naviernet.physics.residuals import (
    IN_PLANE_WEIGHT,
    front_arc_lengths,
    interface_mass_flux,
    interface_shear_residual,
    kinematic_residual,
    laplace_jump_residual,
)
from naviernet.relative_error import relative_errors
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


# Open liquid for the temperature checks: farther than this from the front
# (L_ref units, 20 um at the 300 um scale), past the thermal layer the
# saturated front holds against the liquid (about 7 um at Pe 2000), where a
# steep gradient is physics.
FRONT_MARGIN_STAR = 0.067


def _near(mask: np.ndarray, cells_x: int, cells_y: int) -> np.ndarray:
    """``mask`` grown by ``cells_x`` columns and ``cells_y`` rows."""
    grown = mask.copy()
    for _ in range(cells_x):
        grown[:, 1:] |= grown[:, :-1]
        grown[:, :-1] |= grown[:, 1:]
    for _ in range(cells_y):
        grown[1:, :] |= grown[:-1, :]
        grown[:-1, :] |= grown[1:, :]
    return grown


def temperature_record(fields: dict, grid: AuditGrid, groups: dict, l_ref_um: float) -> dict:
    """The liquid temperature's sanity on the grid, in kelvin: the largest
    streamwise gradient between neighbouring cells of open liquid (K/mm, with
    where it sits in microns) and the deepest subcooling below the inlet over
    all the liquid. Empty when the model carries no temperature."""
    if "T" not in fields:
        return {}
    dt_ref = float(groups["dT_ref"])
    dx, dy = float(grid.xs[1] - grid.xs[0]), float(abs(grid.ys[1] - grid.ys[0]))
    vapour = fields["alpha"] >= 0.5
    open_liquid = ~_near(
        vapour, int(np.ceil(FRONT_MARGIN_STAR / dx)), int(np.ceil(FRONT_MARGIN_STAR / dy))
    )
    both = open_liquid[:, 1:] & open_liquid[:, :-1]
    theta = fields["T"]
    gradient = (
        np.where(both, np.abs(np.diff(theta, axis=1)), 0.0) * dt_ref / (dx * l_ref_um * 1e-3)
    )
    row, col = np.unravel_index(int(np.argmax(gradient)), gradient.shape)
    liquid = theta[~vapour]
    coldest = float(liquid.min()) if liquid.size else float(groups["theta_inlet"])
    return {
        "gradient_K_mm": float(gradient[row, col]),
        "gradient_at_um": [
            float(0.5 * (grid.xs[col] + grid.xs[col + 1]) * l_ref_um),
            float(grid.ys[row] * l_ref_um),
        ],
        "subcooling_K": max(0.0, float(groups["theta_inlet"]) - coldest) * dt_ref,
    }


# The strip at the channel's end reported on its own: the last 75 um held 60% of
# the momentum error at frame 10 of ghost5-15-s0 and at most 4% at frames 3 and 7.
OUTLET_STRIP_UM = 75.0


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
    """``nx`` stations across the camera's view and ``ny`` rows centred on the
    channel centreline of the run's wall span, reaching the nearer wall. An
    outlet extension past the view is left out, so every number stays about
    the imaged channel and comparable with runs that end at the view."""
    y_center = 0.5 * (data.y_lo + data.y_hi)
    half = min(y_center - data.y_lo, data.y_hi - y_center)
    return AuditGrid(
        xs=np.linspace(data.domain.x_min, data.domain.x_view, nx),
        ys=y_center + np.linspace(-half, half, ny),
        y_center=float(y_center),
    )


def _device(model) -> torch.device:
    """Where the model's parameters live: the audit's tensors go there too, so
    a checkpoint trained on a GPU is audited on it."""
    return next(model.parameters()).device


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
    points = torch.tensor(_points(grid, t), device=_device(model))
    shape = (len(grid.ys), len(grid.xs))
    with torch.enable_grad():
        u, v = model.velocity(points)
        out = {"alpha": model.alpha(points), "u": u, "v": v, "s": model.source(points)}
        if "p" in model.fields:
            out["p"] = model.pressure(points)
        if "T" in model.fields:
            out["T"] = model.temperature(points)
    return {name: value.detach().cpu().numpy().reshape(shape) for name, value in out.items()}


def _squared_residuals(
    model, groups, equations, grid: AuditGrid, t: float
) -> dict[str, np.ndarray]:
    """Each collocation equation's squared residual at the grid's points."""
    points = _points(grid, t)
    parts: dict[str, list[np.ndarray]] = {eq.id: [] for eq in equations}
    with torch.enable_grad():
        for start in range(0, len(points), CHUNK):
            chunk = torch.tensor(
                points[start : start + CHUNK], device=_device(model), requires_grad=True
            )
            ctx = registry.LossContext(model, chunk, groups=groups)
            for eq in equations:
                parts[eq.id].append(eq.pointwise(ctx).detach().cpu().numpy().ravel())
    return {name: np.concatenate(values) for name, values in parts.items()}


def _rms_of_squares(squared: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.sqrt(values.mean())) for name, values in squared.items()}


def residual_rms_on_grid(
    model, groups, equations, grid: AuditGrid, t: float
) -> dict[str, float]:
    """RMS of each collocation equation's residual over the grid at ``t``."""
    return _rms_of_squares(_squared_residuals(model, groups, equations, grid, t))


def outlet_strip_record(
    squared: dict[str, np.ndarray], grid: AuditGrid, strip_star: float
) -> dict[str, dict[str, float]]:
    """Each equation's residual in the last ``strip_star`` of the channel against
    the rest, and the strip's share of its squared error. By the last frame the
    nose is about 100 um from the outlet and most of the momentum error sits
    here, so a whole-grid RMS says little about the rest of the channel."""
    columns = np.tile(grid.xs, len(grid.ys))
    in_strip = columns >= float(grid.xs[-1]) - strip_star
    return {
        name: {
            "rms_strip": float(np.sqrt(values[in_strip].mean())),
            "rms_rest": float(np.sqrt(values[~in_strip].mean())),
            "share_of_error": float(values[in_strip].sum() / max(values.sum(), 1e-30)),
        }
        for name, values in squared.items()
    }


def flux_balance(cfg, model, data, groups, t: float) -> dict:
    """The normalised axial-flux defect at every station at ``t``, and its worst
    value upstream of the root, where the flux must simply equal the inlet's."""
    device = _device(model)
    plan = mass_balance.build_plan(
        data.domain.in_view(),
        (data.y_lo, data.y_hi),
        groups,
        cfg.training,
        device,
        at_times=torch.tensor([t], dtype=torch.float32, device=device),
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


def _scalar(value) -> float:
    return float(value) if isinstance(value, float) else float(value.detach())


def wetting_record(model) -> dict:
    """How the footprint is wetted and its fitted or declared value: the
    uniform wet share, or the microlayer's film thickness."""
    if getattr(model, "footprint_wetting", "uniform") == "microlayer":
        return {
            "footprint_wetting": "microlayer",
            "microlayer_um": _scalar(model.microlayer_um),
            "microlayer_mode": model.microlayer_mode,
        }
    return {
        "footprint_wetting": "uniform",
        "wet_fraction": _scalar(model.wet_fraction),
        "wet_fraction_mode": model.wet_fraction_mode,
    }


def heat_budget(cfg, model, data, groups, row: int, fields: dict, grid: AuditGrid) -> dict:
    """The latent power the source implies, its two parts under the sharp
    thermal model (the footprint film source and the outline flux), the latent
    power the measured growth needs, the wall flux's input under the liquid and
    under the footprint, and the spread between the estimates of the latent
    power: the source's integral, the growth's need and the parts' sum. All in
    milliwatts."""
    fluid, exp = cfg.fluid, cfg.experiment
    length = cfg.scales.L_ref_um * 1e-6
    depth = exp.channel_height_um * 1e-6
    speed = cfg.scales.U_ref
    t_ref = length / speed
    liquid = 1.0 - fields["alpha"]

    specific_volume = 1.0 / fluid.rho_v - 1.0 / fluid.rho_l
    # A dimensionless source integral to the latent power its vapour volume cost.
    to_latent_mW = length * depth * speed / specific_volume * fluid.h_lv * 1e3
    growth = measured_growth_rate(data, row) * length**2 / t_ref
    # The floor's local flux over each grid column (both heater zones counted).
    wall = float(exp.q_wall_W_cm2) * 1e4 * wall_flux_ratio_np(grid.xs, groups)[None, :]
    budget = {
        "latent_from_source_mW": _integral(fields["s"], grid) * to_latent_mW,
        "latent_for_growth_mW": fluid.rho_v * growth * depth * fluid.h_lv * 1e3,
        "wall_into_liquid_mW": _integral(wall * liquid, grid) * length**2 * 1e3,
        "wall_under_footprint_mW": _integral(wall * fields["alpha"], grid) * length**2 * 1e3,
    }
    estimates = [budget["latent_from_source_mW"], budget["latent_for_growth_mW"]]
    if getattr(model, "thermal_model", "legacy") == "sharp":
        t_row = torch.tensor(float(data.t[row]), device=_device(model))
        flux, weights, _ = model._front_flux(t_row, None)
        along_front = float((flux * weights).sum().detach())
        budget["latent_from_flux_mW"] = fluid.k_l * groups["dT_ref"] * depth * along_front * 1e3
        budget["latent_from_footprint_mW"] = 0.0
        if getattr(model, "footprint_evaporation", False):
            points = torch.tensor(_points(grid, float(data.t[row])), device=_device(model))
            footprint = model.footprint_source(points).detach().cpu().numpy()
            footprint = footprint.reshape(len(grid.ys), len(grid.xs))
            budget["latent_from_footprint_mW"] = _integral(footprint, grid) * to_latent_mW
            budget.update(wetting_record(model))
        estimates.append(budget["latent_from_flux_mW"] + budget["latent_from_footprint_mW"])
    reference = max(abs(budget["latent_for_growth_mW"]), 1e-9)
    budget["spread"] = float((max(estimates) - min(estimates)) / reference)
    return budget


def outline_agreement(cfg, model, data, row: int) -> dict:
    """The frame's IoU against its mask, and the nose position of the predicted
    and the measured outline with their distance, in L_ref units. A predicted
    outline with no vapour has no nose; its error is reported infinite."""
    from naviernet.evaluation import frame_iou, front_position, predict_alpha

    stride, threshold = int(cfg.evaluation.stride), float(cfg.evaluation.threshold)
    xs = data.x[::stride]
    x_anchor = float(data.pin_anchor[0])
    predicted = predict_alpha(model, data, float(data.t[row]), stride) > threshold
    measured = data.alpha[row, ::stride, ::stride] > threshold
    nose_model = front_position(predicted, xs, x_anchor)
    nose_measured = front_position(measured, xs, x_anchor)
    error = abs(nose_model - nose_measured)
    # None, never inf: JSON has no infinity, and the report must stay readable.
    # The verdict counts a missing nose as the whole channel length off.
    return {
        "iou": frame_iou(cfg, model, data, row),
        "nose_model": None if np.isnan(nose_model) else float(nose_model),
        "nose_measured": None if np.isnan(nose_measured) else float(nose_measured),
        "nose_error": None if np.isnan(error) else float(error),
    }


# The front's four segments, as the sampler orders them and the jump map names them.
SEGMENTS = ("upper", "lower", "root", "nose")
ROOT_SEGMENT = SEGMENTS.index("root")

# The explicit front the audit samples for its jump and kinematic checks. The
# Young-Laplace residual converges at 128 body and 64 cap points (within a
# percent beyond that); the training's 64 and 16 under-resolve the root and
# nose, so the audit reads the front at this converged resolution.
AUDIT_FRONT_BODY_SAMPLES = 128
AUDIT_FRONT_CAP_SAMPLES = 64


def _segment_codes(front) -> np.ndarray:
    """0 upper body, 1 lower body, 2 root cap, 3 nose cap, per front sample."""
    side = front.side.detach().cpu().numpy().ravel()
    cap = front.on_cap.detach().cpu().numpy().ravel() > 0.5
    spine = front.u.detach().cpu().numpy().ravel()
    codes = np.where(side > 0, 0, 1)
    codes[cap & (spine < 0.5)] = 2
    codes[cap & (spine >= 0.5)] = 3
    return codes


def jump_shape_pixels(
    laplace: np.ndarray, kappa_par: np.ndarray, groups, pixel: float, width: float
) -> np.ndarray:
    """The Young-Laplace mismatch at each front sample as the shape change that
    would remove it, in camera pixels.

    A local displacement of the outline by ``d`` across a feature of radius
    ``R`` changes its in-plane curvature by ``d / R^2``, and so the jump by
    ``(pi/4) d / (R^2 We)``. ``R`` is the local radius of curvature, capped at
    half the channel width (a flat stretch of the body still bends over the
    channel's own scale). The gap-direction curvature reads the front's speed,
    not its shape, and is left out of the yardstick.
    """
    kappa = np.maximum(np.abs(kappa_par), 2.0 / width)
    per_pixel = IN_PLANE_WEIGHT * kappa**2 * pixel / groups["We"]
    return np.abs(laplace) / per_pixel


def _by_length(
    laplace: np.ndarray, codes: np.ndarray, lengths: np.ndarray, scale: float
) -> dict:
    """The jump's error in pascals with each sample counted by the length of front
    it stands for: the root cap is 10 to 17% of the samples but 4 to 8% of the
    outline. Also with the root cap left out, where the jump's curvature formula
    does not apply, and the root cap on its own."""
    root = codes == ROOT_SEGMENT

    def weighted(values: np.ndarray, keep: np.ndarray) -> float:
        return float(np.sum(values[keep] * lengths[keep]) / max(np.sum(lengths[keep]), 1e-30))

    everywhere = np.ones_like(root)
    return {
        "rms": float(np.sqrt(weighted(laplace**2, everywhere))) * scale,
        "mean": weighted(laplace, everywhere) * scale,
        "rms_without_root": float(np.sqrt(weighted(laplace**2, ~root))) * scale,
        "root_mean": weighted(laplace, root) * scale,
        "root_rms": float(np.sqrt(weighted(laplace**2, root))) * scale,
    }


def _laplace_in_pascals(
    model, laplace: np.ndarray, codes: np.ndarray, t: float, lengths: np.ndarray
) -> dict:
    """The Young-Laplace jump around the front in pascals: the vapour pressure
    (gauge), the jump's mean error (zero when imposed on average), and its
    scatter about that mean, overall and per segment: the part a uniform vapour
    pressure cannot satisfy, which is the 2D model's form error at the front."""
    scale = float(model.cfg.fluid.rho_l) * float(model.cfg.scales.U_ref) ** 2
    instant = torch.tensor([[t]], dtype=torch.float32, device=_device(model))
    vapour = float(model.p_vapor(instant).detach()) * scale
    mean = float(laplace.mean())
    scatter = (laplace - mean) * scale
    return {
        "vapour_pressure_Pa": vapour,
        "laplace_rms_Pa": _rms(laplace) * scale,
        "laplace_mean_Pa": mean * scale,
        "laplace_scatter_Pa": _rms(scatter),
        "laplace_scatter_Pa_by_segment": {
            name: _rms(scatter[codes == k])
            for k, name in enumerate(SEGMENTS)
            if (codes == k).any()
        },
        "laplace_by_length_Pa": _by_length(laplace, codes, lengths, scale),
    }


def _shear_in_pascals(model, front, groups, upper: np.ndarray) -> dict:
    """The tangential stress the liquid puts on the front (``model.interface_
    shear``), RMS in pascals on each half: what a shear-free edge would make zero."""
    scale = float(model.cfg.fluid.rho_l) * float(model.cfg.scales.U_ref) ** 2
    with torch.enable_grad():
        shear = interface_shear_residual(model, front, groups).detach().cpu().numpy().ravel()
    return {
        "shear_rms_upper_Pa": _rms(shear[upper]) * scale,
        "shear_rms_lower_Pa": _rms(shear[~upper]) * scale,
    }


def front_conditions(
    model, groups, t: float, pixel: float = 0.0, width: float = 1.0
) -> tuple[dict | None, dict[str, np.ndarray]]:
    """Kinematic and Young-Laplace residual RMS on the upper and lower halves of
    the explicit front, and (``pixel`` > 0) the jump mismatch per segment as the
    shape change in pixels that would remove it; with the per-sample arrays the
    jump map is drawn from. ``(None, {})`` for a run without the sharp front."""
    if not getattr(model, "front_geometry", False) or not getattr(
        model, "sharp_interface", False
    ):
        return None, {}
    model_cfg = model.cfg.model
    with torch.enable_grad():
        front = model.front(
            torch.tensor([[t]], dtype=torch.float32, device=_device(model)),
            AUDIT_FRONT_BODY_SAMPLES,
            AUDIT_FRONT_CAP_SAMPLES,
        )
        mass_flux = (
            interface_mass_flux(model, front, groups)
            if getattr(model, "thermal_model", "legacy") == "sharp"
            else None
        )
        kinematic = (
            kinematic_residual(model, front, mass_flux=mass_flux).detach().cpu().numpy().ravel()
        )
        laplace = laplace_jump_residual(model, front, groups).detach().cpu().numpy().ravel()
    upper = front.side.detach().cpu().numpy().ravel() > 0
    record = {
        "kinematic_rms_upper": _rms(kinematic[upper]),
        "kinematic_rms_lower": _rms(kinematic[~upper]),
        "laplace_rms_upper": _rms(laplace[upper]),
        "laplace_rms_lower": _rms(laplace[~upper]),
    }
    if bool(getattr(model_cfg, "interface_shear", False)):
        record.update(_shear_in_pascals(model, front, groups, upper))
    if pixel <= 0.0:
        return record, {}
    kappa = front.kappa_par.detach().cpu().numpy().ravel()
    pixels = jump_shape_pixels(laplace, kappa, groups, pixel, width)
    lengths = front_arc_lengths(front).detach().cpu().numpy().ravel()
    record.update(
        _laplace_in_pascals(model, laplace, codes := _segment_codes(front), t, lengths)
    )
    record["jump_px_median"] = float(np.median(pixels))
    record["jump_px_p90"] = float(np.percentile(pixels, 90))
    record["jump_px_by_segment"] = {
        name: float(np.median(pixels[codes == k]))
        for k, name in enumerate(SEGMENTS)
        if (codes == k).any()
    }
    xy = front.points.detach().cpu().numpy()[:, :2]
    return record, {
        "front_x": xy[:, 0],
        "front_y": xy[:, 1],
        "front_jump_px": pixels,
        "front_segment": codes,
    }


def depth_drain_record(model, groups, t: float, view_x_max: float | None) -> dict | None:
    """How the edge rule's mismatch at instant ``t`` is split under
    ``model.depth_drain``: the force (each part's size summed along the edge
    and across the channel's depth, micronewtons) kept in the 2D solution,
    drained to the depth and answered by the root's law, each as a share of
    the three; the size of the drain the loss charges (RMS along a half,
    pascals); and how far the bubble rebuilt from the forces sits from the
    real outline (RMS, microns, the worse half) with nothing drained and with
    the drain; and the pressure the outline forces on the liquid turning
    around the head (pascals) with the share of the drain's price the instant
    is charged under ``model.depth_drain_ease``. ``None`` without the flag or
    before the first image."""
    if not getattr(model, "depth_drain", False):
        return None
    instant = torch.tensor([[t]], dtype=torch.float32, device=_device(model))
    with torch.enable_grad():
        front = model.front(instant, AUDIT_FRONT_BODY_SAMPLES, AUDIT_FRONT_CAP_SAMPLES)
        split = depth_drain.depth_split(model, front, groups, view_x_max)
    if len(split.move_before) == 0:
        return None
    length = float(model.cfg.scales.L_ref_um) * 1e-6
    pressure = float(model.cfg.fluid.rho_l) * float(model.cfg.scales.U_ref) ** 2
    micronewtons = pressure * length * groups["H_star"] * length * 1e6
    record = {
        f"{name}_force_uN": force * micronewtons
        for name, force in depth_drain.part_forces(split).items()
    }
    record.update(
        {f"{name}_share": share for name, share in depth_drain.force_shares(split).items()}
    )
    record["drain_rms_Pa"] = float(split.needed.detach()) * pressure
    record["edge_move_undrained_um"] = float(split.move_before.max()) * length * 1e6
    record["edge_move_um"] = float(split.move_after.max()) * length * 1e6
    record["turnaround_Pa"] = float(split.turnaround.max()) * pressure
    record["drain_price_share"] = float(split.price_share.min())
    return record


def audit_frame(
    cfg, model, data, groups, equations, grid: AuditGrid, row: int
) -> tuple[dict, dict]:
    """One event frame's audit record and its field snapshot."""
    t = float(data.t[row])
    fields = fields_on_grid(model, grid, t)
    squared = _squared_residuals(model, groups, equations, grid, t)
    record = {
        "frame": int(data.frame_numbers[row]),
        # A ghost frame is a rule's outline before the first image: audited like
        # any instant, but no verdict is taken on it.
        "ghost": row < int(getattr(data, "n_ghost", 0)),
        "t_star": t,
        "mirror": {
            name: mirror_breaking_fraction(values, odd=name in ODD_FIELDS)
            for name, values in fields.items()
        },
        "residual_rms": _rms_of_squares(squared),
        "outlet_strip": outlet_strip_record(
            squared, grid, OUTLET_STRIP_UM / float(cfg.scales.L_ref_um)
        ),
        "flux": flux_balance(cfg, model, data, groups, t),
        "heat": heat_budget(cfg, model, data, groups, row, fields, grid),
        "outline": outline_agreement(cfg, model, data, row),
        "temperature": temperature_record(fields, grid, groups, float(cfg.scales.L_ref_um)),
    }
    nx = int(getattr(cfg.evaluation, "relative_error_nx", 0))
    if nx > 0:
        record["relative_error"] = relative_errors(
            model,
            groups,
            data,
            t,
            nx,
            int(cfg.evaluation.relative_error_ny),
            int(getattr(cfg.evaluation, "relative_error_random", 0)),
            int(cfg.training.seed),
        )
    pixel = float(data.x[1] - data.x[0])
    record["front"], front = front_conditions(
        model, groups, t, pixel, float(data.y_hi - data.y_lo)
    )
    drain = depth_drain_record(model, groups, t, float(data.domain.x_view))
    if drain is not None:
        record["depth_drain"] = drain
    return record, {**fields, **front}


def _verdicts(
    frames: list[dict],
    acceptance,
    warmup_iou: dict[str, float] | None,
    nose_missing: float,
) -> dict:
    """Pass/fail against the configured thresholds, with the worst value behind
    each verdict so a failure says by how much. A verdict carries ``max`` when
    the worst value must stay below it and ``min`` when it must stay above.
    ``nose_missing`` is the error booked for a frame whose predicted outline
    has no nose (the domain's length: as far off as a nose can be)."""
    hidden = [name for name in FIELDS if name != "alpha"]
    mirror_worst = max(
        f["mirror"][name] for f in frames for name in hidden if name in f["mirror"]
    )
    residual_worst = max((v for f in frames for v in f["residual_rms"].values()), default=0.0)
    flux_worst = max(f["flux"]["max_abs"] for f in frames)
    heat_worst = max(f["heat"]["spread"] for f in frames)
    verdicts = _outline_verdicts(frames, acceptance, warmup_iou, nose_missing)
    verdicts.update(
        _verdicts_of_fields(mirror_worst, residual_worst, flux_worst, heat_worst, acceptance)
    )
    verdicts.update(_temperature_verdicts(frames, acceptance))
    return verdicts


def _temperature_verdicts(frames: list[dict], acceptance) -> dict:
    """The liquid temperature's sanity: the steepest streamwise gradient in the
    open liquid and the deepest subcooling below the inlet, each at its worst
    frame; nothing for a model without a temperature."""
    records = [f["temperature"] for f in frames if f.get("temperature")]
    if not records:
        return {}
    return {
        "liquid_gradient": _at_most(
            max(r["gradient_K_mm"] for r in records), acceptance.liquid_gradient_max_K_mm
        ),
        "subcooling": _at_most(
            max(r["subcooling_K"] for r in records), acceptance.subcooling_max_K
        ),
    }


def _outline_verdicts(
    frames: list[dict], acceptance, warmup_iou: dict[str, float] | None, nose_missing: float
) -> dict:
    """The outline against the masks: the lowest per-frame IoU above the floor,
    the largest drop below the warm-up's fit of the same frame (when the run
    recorded one), and the largest nose error."""
    iou_worst = min(f["outline"]["iou"] for f in frames)
    errors = [f["outline"]["nose_error"] for f in frames]
    nose_worst = max(nose_missing if e is None else e for e in errors)
    verdicts = {
        "outline": {
            "worst": iou_worst,
            "min": float(acceptance.iou_min),
            "pass": iou_worst >= float(acceptance.iou_min),
        },
        "nose": {
            "worst": nose_worst,
            "max": float(acceptance.nose_tol),
            "pass": nose_worst <= float(acceptance.nose_tol),
        },
    }
    if warmup_iou:
        drops = [
            float(warmup_iou[str(f["frame"])]) - f["outline"]["iou"]
            for f in frames
            if str(f["frame"]) in warmup_iou
        ]
        if drops:
            drop_worst = max(drops)
            verdicts["outline_drop"] = {
                "worst": drop_worst,
                "max": float(acceptance.iou_drop_max),
                "pass": drop_worst <= float(acceptance.iou_drop_max),
            }
    return verdicts


def _verdicts_of_fields(
    mirror_worst, residual_worst, flux_worst, heat_worst, acceptance
) -> dict:
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


def time_scan(cfg, model, data, groups, equations, n_times: int) -> dict | None:
    """The residuals, the flux balance and the jump condition at ``n_times``
    instants spread over the whole window, camera frames or not, on a grid
    half the audit's in each direction; ``None`` when the scan is off."""
    if n_times <= 0:
        return None
    evaluation = cfg.evaluation
    grid = audit_grid(
        data, int(evaluation.audit_nx) // 2 + 1, int(evaluation.audit_ny) // 2 + 1
    )
    times = np.linspace(data.domain.t_min, data.domain.t_max, n_times)
    scan: dict[str, list] = {"t_star": times.tolist(), "flux_max_abs": [], "laplace_rms": []}
    residuals: dict[str, list[float]] = {eq.id: [] for eq in equations}
    for t in times:
        for name, value in residual_rms_on_grid(model, groups, equations, grid, t).items():
            residuals[name].append(value)
        scan["flux_max_abs"].append(flux_balance(cfg, model, data, groups, float(t))["max_abs"])
        front, _ = front_conditions(model, groups, float(t))
        scan["laplace_rms"].append(
            None
            if front is None
            else float(
                np.hypot(front["laplace_rms_upper"], front["laplace_rms_lower"]) / np.sqrt(2)
            )
        )
    scan["residual_rms"] = residuals
    return scan


def _scan_verdicts(scan: dict | None, acceptance, t_first_image: float) -> dict:
    """The equations and the flux balance held at every scanned instant from
    the first image on (the ghost frames before it carry no verdict)."""
    if scan is None:
        return {}
    judged = [k for k, t in enumerate(scan["t_star"]) if t >= t_first_image]
    residual_worst = max(
        (v[k] for v in scan["residual_rms"].values() for k in judged), default=0.0
    )
    flux_worst = max(scan["flux_max_abs"][k] for k in judged)
    return {
        "residuals_in_time": {
            "worst": residual_worst,
            "max": acceptance.residual_rms_max,
            "pass": residual_worst <= acceptance.residual_rms_max,
        },
        "flux_in_time": {
            "worst": flux_worst,
            "max": acceptance.flux_tol,
            "pass": flux_worst <= acceptance.flux_tol,
        },
    }


def _jump_verdict(frames: list[dict], acceptance) -> dict:
    """The jump mismatch two ways: as shape change, the worst frame's 90th
    percentile in pixels against ``jump_px_max``; and as pressure, the worst
    frame's RMS in pascals against ``jump_pa_max``. Nothing for a run without
    the sharp front."""
    fronts = [f["front"] for f in frames if f["front"] and "jump_px_p90" in f["front"]]
    if not fronts:
        return {}
    return {
        "jump_shape": _at_most(max(f["jump_px_p90"] for f in fronts), acceptance.jump_px_max),
        "jump_pressure": _at_most(
            max(f["laplace_rms_Pa"] for f in fronts), acceptance.jump_pa_max
        ),
    }


def _worst_of(frames: list[dict]) -> dict:
    """The worst value of each scored quantity over ``frames``."""
    fronts = [
        f["front"] for f in frames if f.get("front") and "laplace_by_length_Pa" in f["front"]
    ]
    worst = {
        "residual": max(max(f["residual_rms"].values(), default=0.0) for f in frames),
        "flux": max(f["flux"]["max_abs"] for f in frames),
        "heat": max(f["heat"]["spread"] for f in frames),
    }
    if fronts:
        lengths = [f["laplace_by_length_Pa"] for f in fronts]
        worst.update(
            jump_rms_Pa=max(f["laplace_rms_Pa"] for f in fronts),
            jump_rms_by_length_Pa=max(v["rms"] for v in lengths),
            jump_rms_by_length_without_root_Pa=max(v["rms_without_root"] for v in lengths),
            root_rms_Pa=max(v["root_rms"] for v in lengths),
        )
    return worst


def scoring_summary(frames: list[dict]) -> dict:
    """The worst values over the judged frames, and without the last of them, which
    the outlet distorts. A report beside the verdicts: it changes none of them."""
    summary = {"all_frames": _worst_of(frames)}
    if len(frames) > 1:
        summary["without_last_frame"] = _worst_of(frames[:-1])
    return summary


def _at_most(worst: float, limit) -> dict:
    return {"worst": worst, "max": float(limit), "pass": worst <= float(limit)}


def warmup_fit(paths: RunPaths) -> dict[str, float] | None:
    """The per-frame IoU the run reached at the end of its warm-up, as the
    trainer recorded it in the checkpoint at the step the physics engaged;
    ``None`` for a run without that record (no warm-up, or an older run)."""
    if not paths.checkpoint.exists():
        return None
    state = torch.load(paths.checkpoint, map_location="cpu", weights_only=False).get(
        "state", {}
    )
    recorded = state.get("warmup_iou")
    return {str(k): float(v) for k, v in recorded.items()} if recorded else None


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
    warmup = warmup_fit(paths)
    scan = time_scan(cfg, model, data, groups, equations, int(evaluation.audit_scan_times))
    # The verdicts are about the images: ghost frames are reported, not judged.
    judged = [f for f in frames if not f["ghost"]]
    verdicts = _verdicts(
        judged,
        evaluation.acceptance,
        warmup,
        float(data.domain.x_view - data.domain.x_min),
    )
    verdicts.update(_scan_verdicts(scan, evaluation.acceptance, judged[0]["t_star"]))
    verdicts.update(_jump_verdict(judged, evaluation.acceptance))
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
            "iou_min": float(evaluation.acceptance.iou_min),
            "iou_drop_max": float(evaluation.acceptance.iou_drop_max),
            "nose_tol": float(evaluation.acceptance.nose_tol),
            "jump_px_max": float(evaluation.acceptance.jump_px_max),
            "jump_pa_max": float(evaluation.acceptance.jump_pa_max),
            "liquid_gradient_max_K_mm": float(evaluation.acceptance.liquid_gradient_max_K_mm),
            "subcooling_max_K": float(evaluation.acceptance.subcooling_max_K),
        },
        "warmup_iou": warmup,
        "ghost_frames": len(frames) - len(judged),
        "scoring": scoring_summary(judged),
        "verdicts": verdicts,
        "frames": frames,
        "scan": scan,
    }
    # allow_nan=False: a report a browser cannot parse is no report.
    paths.audit_json.write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(paths.audit_fields_npz, **snapshots)
    log.info(
        "audit: %s",
        " ".join(
            f"{name}={'pass' if v['pass'] else 'FAIL'}({v['worst']:.3g})"
            for name, v in report["verdicts"].items()
        ),
    )
    return report
