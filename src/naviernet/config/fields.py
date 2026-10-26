"""The user-facing inputs, declared once (trustworthy-fields WP7).

Every value a person may set from the app is one entry here: a series'
operating conditions, the physics choices and training switches of a run, and
the thresholds a run is accepted against. Each entry names its Hydra key, its
label and unit, its kind (a number, a switch, or a choice among options), its
bounds, who owns it, whether it is baked into the tensors, and whether it is an
advanced dial rather than a form control.

The API serves the table (``GET /api/config/fields``) and generates its
condition and override allow-lists from it; the web renders its forms from the
exported copy (``apps/web/src/lib/fieldTable.json``, written by
``scripts/export_field_table.py``); ``tests/test_field_table.py`` holds the
table, the schema dataclasses and the exported copy in step. Adding an input is
one entry here, plus the request-model field that carries it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

from naviernet.config import schema

Kind = Literal["float", "int", "bool", "enum"]
Section = Literal["conditions", "physics", "training", "acceptance"]
# Who sets the value: "series" lives in the series' conditions file and travels
# with every run on it; "form" is a launch-form control; "override" is reachable
# through the advanced overrides (and the CLI) only.
Owner = Literal["series", "form", "override"]

# The schema groups a key may start with, and the dataclass each resolves to.
_GROUPS: dict[str, type] = {
    "experiment": schema.ExperimentConfig,
    "scales": schema.ScalesConfig,
    "model": schema.ModelConfig,
    "training": schema.TrainingConfig,
    "evaluation": schema.EvaluationConfig,
}

# Generous physical sanity limits, not experiment tuning (SECURITY.md section 4:
# the values come from the network).
_TEMPERATURE_C = (-273.15, 1000.0)
_LENGTH_UM = (1.0, 1e6)
_FLUX_W_CM2 = (1e-3, 1e4)
_WEIGHT = (0.0, 1e4)
_FRACTION = (0.0, 1.0)
_TOLERANCE = (0.0, 10.0)


@dataclass(frozen=True)
class FieldSpec:
    """One user-facing input."""

    key: str  # the Hydra path, e.g. "experiment.T_inlet_C"
    label: str
    kind: Kind
    section: Section
    owner: Owner
    help: str = ""
    unit: str = ""
    options: tuple[str, ...] = ()  # the admitted values of an enum
    lo: float | None = None
    hi: float | None = None
    optional: bool = False  # None is a value: "not set, take the derived default"
    baked: bool = False  # written into the tensors; an edit needs a re-preprocess
    advanced: bool = False  # reachable, but not a form control

    @property
    def name(self) -> str:
        """The short name the API and the web use: the key's last segment."""
        return self.key.rsplit(".", 1)[1]

    @property
    def default(self) -> Any:
        """The schema's default for this field (None when the series config
        must supply it)."""
        return schema_default(self.key)

    def as_dict(self) -> dict[str, Any]:
        """The JSON shape the API serves and the web imports."""
        entry = dataclasses.asdict(self)
        entry["options"] = list(self.options)
        entry["name"] = self.name
        entry["default"] = self.default
        return entry


_SERIES = ("conditions", "series")

# The operating conditions every series carries (the hand-written forms).
GEOMETRY_AND_FLOW: tuple[FieldSpec, ...] = (
    FieldSpec(
        "experiment.dt_frame_ms",
        "Frame interval",
        "float",
        *_SERIES,
        help="time between camera frames; sets the time axis of the tensors",
        unit="ms",
        lo=1e-6,
        hi=1e4,
        baked=True,
    ),
    FieldSpec(
        "experiment.channel_width_um",
        "Channel width",
        "float",
        *_SERIES,
        help="wall to wall in the imaged direction; calibrates um/px from the walls",
        unit="µm",
        lo=_LENGTH_UM[0],
        hi=_LENGTH_UM[1],
        baked=True,
    ),
    FieldSpec(
        "experiment.channel_height_um",
        "Channel height",
        "float",
        *_SERIES,
        help="the depth of the channel; the heated wall is below",
        unit="µm",
        lo=_LENGTH_UM[0],
        hi=_LENGTH_UM[1],
    ),
    FieldSpec(
        "experiment.flow_rate_mL_hr",
        "Flow rate",
        "float",
        *_SERIES,
        help="volumetric inlet flow rate",
        unit="mL/h",
        lo=1e-3,
        hi=1e6,
    ),
    FieldSpec(
        "experiment.q_wall_W_cm2",
        "Wall heat flux",
        "float",
        *_SERIES,
        help=(
            "the series' wall flux, uniform over the heated wall: it heats the liquid "
            "where there is liquid and evaporates the film under the bubble"
        ),
        unit="W/cm²",
        lo=_FLUX_W_CM2[0],
        hi=_FLUX_W_CM2[1],
    ),
    FieldSpec(
        "scales.U_ref",
        "Reference velocity",
        "float",
        *_SERIES,
        help="measured nose-speed scale; sets the reference time",
        unit="m/s",
        lo=1e-6,
        hi=1e3,
        baked=True,
    ),
)

# The thermal conditions and heaters the sharp thermal model reads.
THERMAL: tuple[FieldSpec, ...] = (
    FieldSpec(
        "experiment.T_inlet_C",
        "Inlet temperature",
        "float",
        *_SERIES,
        help="the liquid entering the channel; not set means saturation",
        unit="°C",
        lo=_TEMPERATURE_C[0],
        hi=_TEMPERATURE_C[1],
        optional=True,
    ),
    FieldSpec(
        "experiment.T_init_C",
        "Initial liquid temperature",
        "float",
        *_SERIES,
        help=(
            "the whole liquid before the first frame; its excess over saturation "
            "is the superheat scale"
        ),
        unit="°C",
        lo=_TEMPERATURE_C[0],
        hi=_TEMPERATURE_C[1],
        optional=True,
    ),
    FieldSpec(
        "experiment.T_init_layer_um",
        "Initial thermal layer",
        "float",
        *_SERIES,
        help=(
            "the initial temperature holds only farther than this from the first frame's "
            "outline; inside it the heat equation forms the layer the bubble already has"
        ),
        unit="µm",
        lo=0.0,
        hi=1e4,
    ),
    FieldSpec(
        "experiment.q_upstream_W_cm2",
        "Upstream heater flux",
        "float",
        *_SERIES,
        help=(
            "the floor's flux from the inlet edge of the view to the heater edge; not set "
            "means one flux over the whole floor"
        ),
        unit="W/cm²",
        lo=0.0,
        hi=_FLUX_W_CM2[1],
        optional=True,
    ),
    FieldSpec(
        "experiment.heater_edge_um",
        "Heater edge",
        "float",
        *_SERIES,
        help="where the upstream heater zone ends, from the inlet edge of the view",
        unit="µm",
        lo=0.0,
        hi=_LENGTH_UM[1],
        optional=True,
    ),
    FieldSpec(
        "experiment.side_wall_thermal",
        "Side walls",
        "enum",
        *_SERIES,
        help="adiabatic walls pass no heat; isothermal walls hold the wall temperature",
        options=("adiabatic", "isothermal"),
    ),
    FieldSpec(
        "experiment.T_wall_C",
        "Side-wall temperature",
        "float",
        *_SERIES,
        help="read only when the side walls are isothermal",
        unit="°C",
        lo=_TEMPERATURE_C[0],
        hi=_TEMPERATURE_C[1],
        optional=True,
    ),
    FieldSpec(
        "experiment.wet_fraction_mode",
        "Wet fraction",
        "enum",
        *_SERIES,
        help=(
            "the wet share of the footprint the wall flux evaporates: fixed at the "
            "value below, or fitted as one bounded scalar that starts there"
        ),
        options=("fixed", "fit"),
    ),
    FieldSpec(
        "experiment.wet_fraction",
        "Wet fraction value",
        "float",
        *_SERIES,
        help=(
            "the share of the bubble's footprint still covered by liquid film; the "
            "fixed value, or the fit's start (strictly inside 0 and 1)"
        ),
        lo=_FRACTION[0],
        hi=_FRACTION[1],
    ),
    FieldSpec(
        "experiment.footprint_wetting",
        "Footprint wetting",
        "enum",
        *_SERIES,
        help=(
            "uniform: the wet share above for the whole event; microlayer: each spot the "
            "bubble covers stays wet until the floor's flux has evaporated its film"
        ),
        options=("uniform", "microlayer"),
    ),
    FieldSpec(
        "experiment.microlayer_mode",
        "Microlayer thickness",
        "enum",
        *_SERIES,
        help="the film thickness: fixed at the value below, or fitted from it",
        options=("fixed", "fit"),
    ),
    FieldSpec(
        "experiment.microlayer_um",
        "Microlayer thickness value",
        "float",
        *_SERIES,
        help="the film the bubble leaves on the floor it covers; the fixed value or the fit's start",
        unit="µm",
        lo=0.01,
        hi=50.0,
    ),
    FieldSpec(
        "experiment.footprint_age_ms",
        "First footprint age",
        "float",
        *_SERIES,
        help="how long before the first frame its footprint was covered; 0 is freshly covered",
        unit="ms",
        lo=0.0,
        hi=100.0,
    ),
    FieldSpec(
        "scales.dT_ref_K",
        "Superheat scale",
        "float",
        *_SERIES,
        help=(
            "not set means the initial superheat when the initial temperature is "
            "given, else the conduction scale"
        ),
        unit="K",
        lo=1e-6,
        hi=1e4,
        optional=True,
        advanced=True,
    ),
)

# The physics choices of a run, owned by the launch form.
PHYSICS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "model.walls_at",
        "Walls",
        "enum",
        "physics",
        "form",
        help=(
            "the side walls at the crop edges, or at the detected wall rows (needs "
            "tensors preprocessed with wall positions)"
        ),
        options=("crop", "physical"),
    ),
    FieldSpec(
        "model.wall_condition",
        "Wall condition",
        "enum",
        "physics",
        "form",
        help=(
            "no slip holds the liquid still at the walls; no penetration only stops "
            "it crossing them"
        ),
        options=("no_slip", "no_penetration"),
    ),
    FieldSpec(
        "model.momentum_closure",
        "Momentum closure",
        "enum",
        "physics",
        "form",
        help="Darcy drag alone, or Darcy with the Brinkman term that makes no slip admissible",
        options=("darcy", "brinkman"),
    ),
    FieldSpec(
        "model.hard_kinematics",
        "Exact kinematic condition",
        "bool",
        "physics",
        "override",
        help=(
            "the liquid moves across the bubble edge exactly at the edge's own speed, on the "
            "whole edge, by construction instead of as a penalty; sliding along it stays free"
        ),
        advanced=True,
    ),
    FieldSpec(
        "model.hard_walls",
        "No flow through the walls, exactly",
        "bool",
        "physics",
        "override",
        help=(
            "the flow across the channel is zero at both side walls by construction "
            "instead of as a penalty; the flow along them stays a penalty"
        ),
        advanced=True,
    ),
    FieldSpec(
        "model.vapour_potential_region",
        "Curl-free vapour, where",
        "enum",
        "physics",
        "override",
        help=(
            "where the vapour's curl-free condition is scored: band (weighted by the smoothed "
            "interface, so it reaches into the liquid beside it) or vapour (inside the bubble only)"
        ),
        options=("band", "vapour"),
        advanced=True,
    ),
    FieldSpec(
        "model.kinematic_side",
        "Exact kinematic side",
        "enum",
        "physics",
        "override",
        help=(
            "where the exact kinematic correction acts: the liquid only, or both sides of "
            "the bubble edge (singular where the bubble necks)"
        ),
        options=("liquid", "both"),
        advanced=True,
    ),
    FieldSpec(
        "model.exact_continuity",
        "Exact continuity in the liquid",
        "bool",
        "physics",
        "override",
        help=(
            "the liquid's velocity is built so no volume is created or lost in it, it moves "
            "with the bubble edge and passes nothing through the walls; once the physics "
            "starts, on the frozen outline. Replaces the exact kinematic condition"
        ),
        advanced=True,
    ),
    FieldSpec(
        "model.exact_continuity_body_sources",
        "Exact continuity: sources per side",
        "int",
        "physics",
        "override",
        help="points per body side the liquid's potential flow is built from",
        lo=4,
        hi=1000,
        advanced=True,
    ),
    FieldSpec(
        "model.exact_continuity_cap_sources",
        "Exact continuity: sources per cap",
        "int",
        "physics",
        "override",
        help="points per end cap the liquid's potential flow is built from",
        lo=4,
        hi=400,
        advanced=True,
    ),
    FieldSpec(
        "model.exact_continuity_offset",
        "Exact continuity: source depth",
        "float",
        "physics",
        "override",
        help="how far inside the bubble the sources sit, in spacings between them",
        lo=0.05,
        hi=4.0,
        advanced=True,
    ),
    FieldSpec(
        "model.kinematic_blend_length",
        "Kinematic blend length",
        "float",
        "physics",
        "override",
        help="how far from the edge the exact kinematic correction fades out, in L_ref units",
        lo=1e-4,
        hi=1.0,
        advanced=True,
    ),
    FieldSpec(
        "model.laplace_condition",
        "Young-Laplace condition",
        "enum",
        "physics",
        "override",
        help=(
            "the pressure jump across the bubble edge imposed at every edge point, or on "
            "average around the edge, which sets the vapour pressure and reports the rest"
        ),
        options=("pointwise", "mean"),
        advanced=True,
    ),
    FieldSpec(
        "model.momentum_inertia",
        "Inertia in the momentum law",
        "bool",
        "physics",
        "override",
        help=(
            "adds the liquid's depth-averaged inertia to the Brinkman law, making it the "
            "depth-averaged Navier-Stokes; needed when the liquid accelerates around the bubble"
        ),
        advanced=True,
    ),
    FieldSpec(
        "model.outlet_condition",
        "Outlet",
        "enum",
        "physics",
        "form",
        help=(
            "nothing imposed at the downstream edge, or a zero gradient for a channel "
            "that continues past the field of view"
        ),
        options=("none", "zero_gradient"),
    ),
    FieldSpec(
        "model.outlet_pressure_Pa",
        "Outlet pressure",
        "float",
        "physics",
        "override",
        help=(
            "the liquid pressure at the downstream edge, above atmospheric; not set leaves "
            "the pressure level free"
        ),
        unit="Pa",
        lo=-1e5,
        hi=1e5,
        optional=True,
        advanced=True,
    ),
    FieldSpec(
        "model.inlet_profile",
        "Inlet profile",
        "enum",
        "physics",
        "override",
        help=(
            "the inlet velocity across the width: the mean everywhere, or the developed "
            "profile that falls to zero at the side walls"
        ),
        options=("plug", "developed"),
        advanced=True,
    ),
    FieldSpec(
        "model.pressure_representation",
        "Pressure",
        "enum",
        "physics",
        "form",
        help=(
            "one network for both phases, or the uniform vapour pressure inside the "
            "bubble and the liquid network outside, with the jump built in"
        ),
        options=("single", "two_phase"),
    ),
    FieldSpec(
        "model.thermal_model",
        "Thermal model",
        "enum",
        "physics",
        "form",
        help=(
            "the legacy diffuse energy equation, or the liquid-only equation with the "
            "interface at saturation and the series' thermal conditions"
        ),
        options=("legacy", "sharp"),
    ),
    FieldSpec(
        "model.source_mode",
        "Vapour source",
        "enum",
        "physics",
        "form",
        help=(
            "a free source network pulled toward the evaporation flux, or the flux "
            "itself spread over the front"
        ),
        options=("free", "thermal"),
    ),
    FieldSpec(
        "model.footprint_evaporation",
        "Film evaporation under the bubble",
        "bool",
        "physics",
        "form",
        help=(
            "the wall flux under the footprint evaporates the liquid film at the wet "
            "fraction and creates vapour there: the growth mechanism of confined boiling"
        ),
    ),
    FieldSpec(
        "model.mirror_symmetry",
        "Mirror symmetry",
        "bool",
        "physics",
        "form",
        help=(
            "build the symmetry about the channel centreline into every field; on "
            "in the trusted recipe"
        ),
    ),
)

# The physics dials reachable through the advanced overrides only.
_DIAL = ("physics", "override")
PHYSICS_DIALS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "model.interface_temperature",
        "Interface temperature",
        "enum",
        *_DIAL,
        help=(
            "saturation is conduction-limited evaporation; the kinetic route is not "
            "built under the sharp model yet"
        ),
        options=("saturation", "kinetic"),
        advanced=True,
    ),
    FieldSpec(
        "model.theta_parameterization",
        "Temperature parameterisation",
        "enum",
        *_DIAL,
        help=(
            "gated multiplies the network by the distance into the liquid, so the "
            "front is exactly at saturation; free is not built yet"
        ),
        options=("gated", "free"),
        advanced=True,
    ),
    FieldSpec(
        "model.theta_gate_width",
        "Temperature gate width",
        "float",
        *_DIAL,
        help="smoothing of the distance gate; keep well below the thermal layer",
        unit="L_ref",
        lo=1e-5,
        hi=1.0,
        advanced=True,
    ),
    FieldSpec(
        "model.theta_gate",
        "Temperature gate shape",
        "enum",
        *_DIAL,
        help=(
            "linear multiplies the network by the distance into the liquid; saturating levels "
            "off beyond the thermal layer, so the far field does not move with the bubble"
        ),
        options=("linear", "saturating"),
        advanced=True,
    ),
    FieldSpec(
        "model.theta_gate_length",
        "Temperature gate length",
        "float",
        *_DIAL,
        help="where the saturating gate levels off: about the thermal layer's thickness",
        unit="L_ref",
        lo=1e-4,
        hi=10.0,
        advanced=True,
    ),
    FieldSpec(
        "model.source_kernel_width",
        "Source kernel width",
        "float",
        *_DIAL,
        help=(
            "width of the band the front flux is spread over; not set means the "
            "interface half-thickness"
        ),
        unit="L_ref",
        lo=1e-5,
        hi=10.0,
        optional=True,
        advanced=True,
    ),
)

# The training switches of a run.
TRAINING: tuple[FieldSpec, ...] = (
    FieldSpec(
        "training.mass_balance",
        "Cross-section mass balance",
        "bool",
        "training",
        "form",
        help=(
            "the volume flux through every cross-section equals the inlet flux plus "
            "the vapour created upstream"
        ),
    ),
    FieldSpec(
        "training.mass_balance_weight",
        "Mass balance weight",
        "float",
        "training",
        "form",
        help="weight of the mass balance term; it sits outside the rebalancer",
        lo=_WEIGHT[0],
        hi=_WEIGHT[1],
    ),
    FieldSpec(
        "training.growth_closure_weight",
        "Growth closure weight",
        "float",
        "training",
        "override",
        help=(
            "the vapour the heat makes equals the bubble's measured growth at every frame; "
            "trains the footprint's wetting, never the temperature; 0 is off"
        ),
        lo=_WEIGHT[0],
        hi=_WEIGHT[1],
        advanced=True,
    ),
    FieldSpec(
        "training.growth_closure_outline_gradient",
        "Growth closure trains the temperature",
        "bool",
        "training",
        "override",
        help=(
            "let the vapour the bubble edge makes carry the closure's gradient, so the "
            "measured growth also adjusts the temperature; off, it adjusts the film only"
        ),
        advanced=True,
    ),
    FieldSpec(
        "training.growth_closure_stride",
        "Growth closure stride",
        "int",
        "training",
        "override",
        help="every n-th pixel of the covered floor the vapour total is summed over",
        lo=1,
        hi=64,
        advanced=True,
    ),
    FieldSpec(
        "training.collocation_time_bins",
        "Collocation time bins",
        "int",
        "training",
        "form",
        help="distinct instants per collocation batch; the thermal source needs at least one",
        lo=0,
        hi=1024,
    ),
    FieldSpec(
        "training.causal_gate_min_weight",
        "Causal convergence gate",
        "float",
        "training",
        "form",
        help=(
            "the causal schedule tightens only once every time chunk's weight clears "
            "this; not set keeps the fixed annealing"
        ),
        lo=_FRACTION[0],
        hi=_FRACTION[1],
        optional=True,
    ),
    FieldSpec(
        "training.log_chunk_residuals",
        "Log per-chunk residuals",
        "bool",
        "training",
        "form",
        help=(
            "record each equation's residual per time chunk, so a late time left "
            "unenforced is visible in the run's history"
        ),
    ),
    FieldSpec(
        "training.rebalance_max_ratio",
        "Physics weight cap",
        "float",
        "training",
        "form",
        help=(
            "no rebalanced physics weight may exceed this multiple of the data weight; "
            "0 keeps the fixed ceiling alone"
        ),
        lo=_WEIGHT[0],
        hi=_WEIGHT[1],
    ),
    FieldSpec(
        "training.grad_clip",
        "Gradient clip",
        "float",
        "training",
        "form",
        help="bound on the total gradient norm per step; 0 is no clipping",
        lo=_WEIGHT[0],
        hi=_WEIGHT[1],
    ),
    FieldSpec(
        "training.outline_guard_every",
        "Outline guard",
        "int",
        "training",
        "form",
        help=(
            "measure the fit of the masks every this many steps after the warm-up and "
            "stop a run that has drifted; 0 is off"
        ),
        lo=0,
        hi=100000,
    ),
    FieldSpec(
        "training.outline_guard_drop",
        "Outline guard drop",
        "float",
        "training",
        "form",
        help="the mean IoU may fall at most this far below the warm-up's fit",
        lo=_FRACTION[0],
        hi=_FRACTION[1],
    ),
    FieldSpec(
        "training.freeze_geometry",
        "Freeze the outline after the warm-up",
        "bool",
        "training",
        "form",
        help=(
            "two-stage training: the bubble is fixed at the warm-up's fit of the images and "
            "the physics stage trains only the velocity, pressure and temperature"
        ),
    ),
    FieldSpec(
        "model.smooth_apex",
        "Smooth cap apex",
        "bool",
        "physics",
        "override",
        help=(
            "the field around the bubble has no corner on the line through the root's "
            "apex; the outline and the pinned root are unchanged"
        ),
        advanced=True,
    ),
    FieldSpec(
        "training.edge_transport_after_freeze",
        "Edge transport after the freeze",
        "bool",
        "training",
        "override",
        help=(
            "keep the edge-transport term once the outline is frozen; off leaves the "
            "velocity to the kinematic condition on the front"
        ),
        advanced=True,
    ),
    FieldSpec(
        "training.continuity_weight_after_freeze",
        "Continuity weight after the freeze",
        "float",
        "training",
        "override",
        help="the continuity weight from the physics stage on; unset keeps the configured one",
        lo=0.0,
        hi=1e4,
        optional=True,
        advanced=True,
    ),
    FieldSpec(
        "training.energy_relative",
        "Relative temperature loss",
        "bool",
        "training",
        "override",
        help=(
            "train the temperature equation on its error relative to the size of its own "
            "terms, so the open liquid counts as much as the edge of the bubble"
        ),
        advanced=True,
    ),
    FieldSpec(
        "training.energy_relative_floor",
        "Relative temperature loss floor",
        "float",
        "training",
        "override",
        help="the smallest term size the relative temperature loss divides by",
        lo=1e-6,
        hi=10.0,
        advanced=True,
    ),
    FieldSpec(
        "training.energy_relative_mode",
        "Relative temperature loss form",
        "enum",
        "training",
        "override",
        help=(
            "relative: the error relative to the terms alone; mixed: the absolute error "
            "plus the relative one, so no point counts less than under the absolute loss"
        ),
        options=("relative", "mixed"),
        advanced=True,
    ),
    FieldSpec(
        "training.self_adaptive",
        "Self-adaptive weights",
        "enum",
        "training",
        "override",
        help=(
            "a map over space and time that raises a term's weight where its residual "
            "stays high: on the liquid energy equation, or on every collocation equation"
        ),
        options=("off", "energy", "all"),
        advanced=True,
    ),
    FieldSpec(
        "training.self_adaptive_ratio",
        "Self-adaptive weight range",
        "float",
        "training",
        "override",
        help="the largest self-adaptive weight over the smallest",
        lo=1.01,
        hi=1e3,
        advanced=True,
    ),
    FieldSpec(
        "training.energy_relative_cap",
        "Relative temperature loss cap",
        "float",
        "training",
        "override",
        help=(
            "the largest term size the relative temperature loss divides by; above it a "
            "point counts at its absolute error, so a spike cannot hide behind its own size"
        ),
        lo=1e-3,
        hi=1e4,
        advanced=True,
    ),
    FieldSpec(
        "training.frozen_weighting",
        "Weights after the freeze",
        "enum",
        "training",
        "override",
        help=(
            "once the outline is frozen: each physics term at its configured weight, or "
            "rebalanced to equal gradient contributions"
        ),
        options=("fixed", "equalised"),
        advanced=True,
    ),
    FieldSpec(
        "training.one_way_advection",
        "One-way heat advection",
        "bool",
        "training",
        "override",
        help=(
            "the temperature equation trains the temperature only, so its error cannot bend "
            "the flow; the liquid carries the heat, the heat does not move the liquid"
        ),
        advanced=True,
    ),
    FieldSpec(
        "training.warmup_cache",
        "Reuse the warm-up",
        "bool",
        "training",
        "override",
        help=(
            "start from the outline fit an earlier run with the same images, settings and "
            "code already trained; it is retrained only when one of those changes"
        ),
        advanced=True,
    ),
    FieldSpec(
        "training.physics_lr_halflife",
        "Physics-stage learning-rate halving",
        "int",
        "training",
        "override",
        help=(
            "steps per halving of the learning rate after the warm-up; for a physics stage "
            "of fewer, larger steps. 0 keeps the warm-up's halving period"
        ),
        lo=0,
        hi=1_000_000,
        advanced=True,
    ),
    FieldSpec(
        "training.temperature_stage_from",
        "Temperature-only stage from step",
        "int",
        "training",
        "override",
        help=(
            "sequential solve: after this step the outline and flow are locked and only the "
            "temperature trains, on its own equation and conditions; 0 keeps the coupled stage"
        ),
        lo=0,
        hi=1_000_000,
        advanced=True,
    ),
    FieldSpec(
        "training.quadrature_jitter",
        "Jitter the quadrature times",
        "bool",
        "training",
        "override",
        help=(
            "draw the front times and the mass-balance instants afresh each step "
            "instead of fixed grids, so the physics is scored across the window"
        ),
        advanced=True,
    ),
    FieldSpec(
        "training.mass_balance_stations",
        "Mass balance stations",
        "int",
        "training",
        "override",
        help="x-stations of the cumulative source integral",
        lo=2,
        hi=1024,
        advanced=True,
    ),
    FieldSpec(
        "training.mass_balance_times",
        "Mass balance time slices",
        "int",
        "training",
        "override",
        help="fixed time slices across the window",
        lo=1,
        hi=256,
        advanced=True,
    ),
    FieldSpec(
        "training.mass_balance_grid",
        "Mass balance quadrature",
        "int",
        "training",
        "override",
        help="transverse quadrature points, wall to wall",
        lo=2,
        hi=1024,
        advanced=True,
    ),
)

# The acceptance thresholds, reachable through the advanced overrides.
_GATE = ("acceptance", "override")
ACCEPTANCE: tuple[FieldSpec, ...] = (
    FieldSpec(
        "evaluation.audit",
        "Acceptance audit",
        "bool",
        *_GATE,
        help="write the acceptance audit beside the metrics after evaluation",
    ),
    FieldSpec(
        "evaluation.audit_nx",
        "Audit grid, along the channel",
        "int",
        *_GATE,
        lo=9,
        hi=1025,
        advanced=True,
    ),
    FieldSpec(
        "evaluation.audit_ny",
        "Audit grid, across the channel",
        "int",
        *_GATE,
        lo=5,
        hi=513,
        advanced=True,
    ),
    FieldSpec(
        "evaluation.relative_error_nx",
        "Relative error grid, along",
        "int",
        *_GATE,
        help="points along the channel each equation's error relative to its terms is taken on; 0 skips",
        lo=0,
        hi=2049,
        advanced=True,
    ),
    FieldSpec(
        "evaluation.relative_error_ny",
        "Relative error grid, across",
        "int",
        *_GATE,
        help="points across the channel the relative equation errors are taken on",
        lo=3,
        hi=1025,
        advanced=True,
    ),
    FieldSpec(
        "evaluation.relative_error_random",
        "Relative error random points",
        "int",
        *_GATE,
        help=(
            "random points added to the grid, so a feature thinner than its spacing "
            "cannot fall between its lines; 0 keeps the grid alone"
        ),
        lo=0,
        hi=4_000_000,
        advanced=True,
    ),
    FieldSpec(
        "evaluation.audit_scan_times",
        "Audit scan instants",
        "int",
        *_GATE,
        help="instants across the whole event the residuals and the flux are scanned at; 0 skips",
        lo=0,
        hi=1001,
        advanced=True,
    ),
    FieldSpec(
        "evaluation.acceptance.seed_corr_min",
        "Seed agreement",
        "float",
        *_GATE,
        help="minimum correlation of each hidden field across seeds",
        lo=_FRACTION[0],
        hi=_FRACTION[1],
    ),
    FieldSpec(
        "evaluation.acceptance.mirror_max",
        "Mirror breaking",
        "float",
        *_GATE,
        help="maximum symmetry-breaking fraction of a hidden field",
        lo=_TOLERANCE[0],
        hi=_TOLERANCE[1],
    ),
    FieldSpec(
        "evaluation.acceptance.residual_rms_max",
        "Residual RMS",
        "float",
        *_GATE,
        help="maximum RMS of any collocation residual on the audit grid",
        lo=_WEIGHT[0],
        hi=_WEIGHT[1],
    ),
    FieldSpec(
        "evaluation.acceptance.flux_tol",
        "Flux balance",
        "float",
        *_GATE,
        help="maximum cross-section flux defect, relative to the inlet flux",
        lo=_TOLERANCE[0],
        hi=_TOLERANCE[1],
    ),
    FieldSpec(
        "evaluation.acceptance.heat_budget_tol",
        "Heat budget",
        "float",
        *_GATE,
        help="maximum spread between the latent-power estimates",
        lo=_TOLERANCE[0],
        hi=_TOLERANCE[1],
    ),
    FieldSpec(
        "evaluation.acceptance.iou_min",
        "Outline floor",
        "float",
        *_GATE,
        help="minimum IoU against the mask at every event frame",
        lo=_FRACTION[0],
        hi=_FRACTION[1],
    ),
    FieldSpec(
        "evaluation.acceptance.iou_drop_max",
        "Outline drop",
        "float",
        *_GATE,
        help="how far below the warm-up's fit a frame's IoU may end",
        lo=_FRACTION[0],
        hi=_FRACTION[1],
    ),
    FieldSpec(
        "evaluation.acceptance.nose_tol",
        "Nose tolerance",
        "float",
        *_GATE,
        help="maximum distance between the predicted and the measured nose at any frame",
        unit="L_ref",
        lo=_TOLERANCE[0],
        hi=_TOLERANCE[1],
    ),
    FieldSpec(
        "evaluation.acceptance.jump_px_max",
        "Shape change the jump asks for",
        "float",
        *_GATE,
        help=(
            "the Young-Laplace mismatch on the front as the shape change that would remove "
            "it; its 90th percentile may not exceed this many pixels"
        ),
        unit="px",
        lo=0.0,
        hi=1e3,
    ),
)

FIELD_TABLE: tuple[FieldSpec, ...] = (
    *GEOMETRY_AND_FLOW,
    *THERMAL,
    *PHYSICS,
    *PHYSICS_DIALS,
    *TRAINING,
    *ACCEPTANCE,
)

_BY_KEY: dict[str, FieldSpec] = {spec.key: spec for spec in FIELD_TABLE}


def by_key(key: str) -> FieldSpec | None:
    """The entry for a Hydra key, or None when the key is not user-facing."""
    return _BY_KEY.get(key)


def specs(section: Section | None = None, owner: Owner | None = None) -> list[FieldSpec]:
    """The entries matching every given filter, in table order."""
    return [
        spec
        for spec in FIELD_TABLE
        if (section is None or spec.section == section)
        and (owner is None or spec.owner == owner)
    ]


def schema_field(key: str) -> dataclasses.Field:
    """The schema dataclass field a Hydra key resolves to (KeyError if none)."""
    group, _, rest = key.partition(".")
    owner: type = _GROUPS[group]
    segments = rest.split(".")
    for segment in segments[:-1]:
        nested = _field_of(owner, segment)
        owner = getattr(schema, str(nested.type))  # PEP 563: the type is its name
    return _field_of(owner, segments[-1])


def _field_of(owner: type, name: str) -> dataclasses.Field:
    for candidate in dataclasses.fields(owner):
        if candidate.name == name:
            return candidate
    raise KeyError(f"{owner.__name__} has no field {name!r}")


def schema_default(key: str) -> Any:
    """The schema's default for a key, or None when it has none (MISSING)."""
    default = schema_field(key).default
    if default is dataclasses.MISSING or default == schema.MISSING:
        return None
    return default


def hydra_literal(value: Any) -> str:
    """A value as a Hydra override literal: null, true/false, a number, a word."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return repr(value)
    return str(value)


def export() -> list[dict[str, Any]]:
    """The whole table in its JSON shape, in table order."""
    return [spec.as_dict() for spec in FIELD_TABLE]
