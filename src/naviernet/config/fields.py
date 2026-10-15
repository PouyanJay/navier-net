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
            "the uniform background flux along the channel: the setpoint, or the "
            "initial guess when it is fitted"
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
        "experiment.q_cavity_W_cm2",
        "Cavity heater flux",
        "float",
        *_SERIES,
        help="the disk heater around the nucleation cavity; not set means no disk",
        unit="W/cm²",
        lo=0.0,
        hi=_FLUX_W_CM2[1],
        optional=True,
    ),
    FieldSpec(
        "experiment.heater_radius_um",
        "Cavity heater radius",
        "float",
        *_SERIES,
        help="radius of the heated disk around the cavity",
        unit="µm",
        lo=1e-3,
        hi=_LENGTH_UM[1],
        optional=True,
    ),
    FieldSpec(
        "experiment.heater_on_ms",
        "Heater on for",
        "float",
        *_SERIES,
        help=(
            "how long the cavity heater stays on after the first frame; not set "
            "means the whole event"
        ),
        unit="ms",
        lo=0.0,
        hi=1e6,
        optional=True,
    ),
    FieldSpec(
        "experiment.cavity_y_fraction",
        "Cavity position across the channel",
        "float",
        *_SERIES,
        help="fraction of the wall-to-wall span from the lower wall; 0.5 is the centreline",
        lo=_FRACTION[0],
        hi=_FRACTION[1],
    ),
    FieldSpec(
        "experiment.q_wall_mode",
        "Background flux",
        "enum",
        *_SERIES,
        help=(
            "fixed at the wall heat flux, or fitted as one bounded scalar constrained "
            "by the observed growth"
        ),
        options=("fixed", "fit"),
    ),
    FieldSpec(
        "experiment.q_wall_max_W_cm2",
        "Background flux cap",
        "float",
        *_SERIES,
        help="upper bound of the fitted background flux",
        unit="W/cm²",
        lo=_FLUX_W_CM2[0],
        hi=_FLUX_W_CM2[1],
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
        "model.mirror_symmetry",
        "Mirror symmetry",
        "bool",
        "physics",
        "form",
        help=(
            "build the symmetry about the channel centreline into every field; for "
            "after the physics passes its tests without it"
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
