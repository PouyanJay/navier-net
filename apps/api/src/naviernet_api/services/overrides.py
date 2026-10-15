"""Strict validation of free-form Hydra overrides sent from the platform.

The Solver's advanced-overrides field is the bridge to the schema's CLI-only
dials (``rba_gamma``, ``kin_grid``, ``fv_smooth_px``, ...), so a UI run and a
CLI run stop being different config universes. It is also text from the
network headed for Hydra compose, so everything about it is allowlisted
(SECURITY.md section 4):

- only top-level ``training.*`` / ``model.*`` keys that exist on the typed
  schema dataclasses -- never paths, hydra internals, datasets, or run names;
- only fields declared ``int`` / ``float`` / ``bool`` -- strings, lists, and
  nested groups are refused, which structurally rules out every enum the form
  owns and every per-series weight table -- unless the field table
  (``naviernet.config.fields``) declares the key as an override dial, in which
  case its declared kind, options and bounds are the gate (that is how the
  acceptance thresholds and the named physics choices stay reachable);
- never a key the launch form, the series' conditions or the series' saved
  model config already sets: one value must have one home, and a silent
  precedence fight is a config bug waiting to be reproduced.

Every rejection names the offending token and where the value belongs.
"""

from __future__ import annotations

import dataclasses
import math

from naviernet.config import fields as field_table
from naviernet.config.schema import ModelConfig, TrainingConfig

# Hard caps on volume, not values: the field is a handful of expert dials, not
# a config file. Anything larger belongs in a YAML under configs/.
MAX_OVERRIDES = 16
MAX_TOKEN_LENGTH = 128

# |value| cap for numeric overrides. Generous by orders of magnitude over any
# real dial (the largest schema default is 1e3-scale) while refusing the
# pathological (SECURITY.md section 4: the values come from the network).
MAX_MAGNITUDE = 1e9

# Keys the launch form owns (mirrors the override list `_configure` builds in
# run_manager.py -- keep the two in sync; the field table's form-owned entries
# are added below, as `_form_overrides` emits them from the same table).
# Overriding one here would silently fight the form control that claims to set
# it.
_HAND_WRITTEN_FORM_KEYS = frozenset(
    {
        "training.steps",
        "training.lr",
        "training.lr_halflife",
        "training.n_data",
        "training.n_coll",
        "training.n_bc",
        "training.holdout_frame",
        "training.val_fraction",
        "training.val_strategy",
        "training.rebalance_every",
        "training.log_every",
        "training.seed",
        "training.device",
        "training.weighting",
        "training.causal_weighting",
        "training.causal_mode",
        "training.adaptive_collocation",
        "training.root_supervision",
        "training.nose_supervision",
        "training.evap_closure_two_way",
        "training.alpha_eps_anneal_steps",
        "training.alpha_eps_final",
        "training.kinematics",
        "training.kin_margin_frac",
        "training.kin_weight_mono",
        "training.kin_weight_balance",
        "training.kin_weight_evap",
        "training.front_velocity",
        "training.fv_weight",
        "training.fv_apex_weight",
        "model.hard_pin",
        "model.pin_d_ref",
        "model.front_geometry",
        "model.allow_pinch",
        "model.evolving_width",
        "model.cap_freedom",
        "model.cap_delta",
        "model.nucleation_root",
        "model.nose_shape",
        "model.sharp_interface",
        "model.film_pressure",
        "model.depletable_superheat",
        "model.liquid_film",
        "model.root_attachment",
        "model.receding_cap",
    }
)
FORM_OWNED_KEYS = _HAND_WRITTEN_FORM_KEYS | {
    spec.key for spec in field_table.specs(owner="form")
}

# Keys the series' saved model config owns (datasets.MODEL_ARCH_FIELDS): the
# Physics & model stage is their home, and series_overrides() composes them
# into every run. The Stage-B weight table and per-field architecture are
# nested keys, refused structurally below.
SERIES_OWNED_KEYS = frozenset(
    {
        "model.hidden",
        "model.layers",
        "model.fourier_feats",
        "model.fourier_scale",
        "model.alpha_eps",
    }
)

_SCALAR_TYPES = ("int", "float", "bool")

# schema.py uses `from __future__ import annotations` (PEP 563), so field.type
# is the literal source string ("int", "float", "list[float]", ...) and the
# scalar check below is a plain string match. That fails CLOSED: a field ever
# declared with an aliased or verbose spelling ("Optional[float]") becomes
# non-overridable rather than wrongly accepted.
_GROUPS = {
    "training": {field.name: field.type for field in dataclasses.fields(TrainingConfig)},
    "model": {field.name: field.type for field in dataclasses.fields(ModelConfig)},
}


class OverrideRejected(ValueError):
    """An advanced override that cannot be honored, with the reason."""


def validate_overrides(tokens: list[str]) -> list[str]:
    """The overrides in canonical ``key=value`` form, or `OverrideRejected`.

    Canonical means the value is re-emitted from its parsed form (booleans
    lowercased, numbers as Python renders them), so what reaches Hydra is what
    validation actually saw, never the raw token.
    """
    if len(tokens) > MAX_OVERRIDES:
        raise OverrideRejected(
            f"{len(tokens)} overrides; at most {MAX_OVERRIDES} are accepted -- "
            "a larger configuration belongs in a config file, not the launch form"
        )
    canonical: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key, value = _parse_token(token)
        if key in seen:
            raise OverrideRejected(f"override key {key!r} appears more than once")
        seen.add(key)
        canonical.append(f"{key}={_parse_value(key, value, _field_type(key))}")
    return canonical


def _parse_token(token: str) -> tuple[str, str]:
    """Split ``key=value``, rejecting anything else."""
    if len(token) > MAX_TOKEN_LENGTH:
        raise OverrideRejected(
            f"override {token[:32]!r}... is longer than {MAX_TOKEN_LENGTH} characters"
        )
    key, eq, value = token.partition("=")
    key, value = key.strip(), value.strip()
    if not eq or not key or not value:
        raise OverrideRejected(f"override {token!r} is not of the form key=value")
    return key, value


def _field_type(key: str) -> str:
    """The schema type of ``key``, after every allowlist gate."""
    spec = field_table.by_key(key)
    if spec is not None:
        _reject_if_declared_elsewhere(spec)
        return spec.kind
    group, dot, field = key.partition(".")
    if not dot or group not in _GROUPS:
        raise OverrideRejected(
            f"override key {key!r} is outside training.* / model.*; only the "
            "typed training and model schemas can be overridden here"
        )
    if "." in field:
        raise OverrideRejected(
            f"override key {key!r} is nested; only top-level fields are "
            "overridable (the weight table and per-field architecture have "
            "their own homes on the form and the Physics & model stage)"
        )
    _reject_if_owned(key)
    declared = _GROUPS[group].get(field)
    if declared is None:
        raise OverrideRejected(
            f"unknown override key {key!r}: no such field on the {group} schema"
        )
    if declared not in _SCALAR_TYPES:
        raise OverrideRejected(
            f"{key!r} is not a numeric or boolean field ({declared}); only "
            "scalar numeric/boolean overrides are accepted"
        )
    return declared


def _reject_if_declared_elsewhere(spec: field_table.FieldSpec) -> None:
    """A field-table entry is overridable only when the overrides are its home."""
    if spec.owner == "form":
        raise OverrideRejected(
            f"{spec.key!r} is set by the launch form; change it there instead of "
            "overriding it, so the form always says what the run does"
        )
    if spec.owner == "series":
        raise OverrideRejected(
            f"{spec.key!r} belongs to the series' operating conditions; edit it on "
            "the Datasets stage, so every run on the series sees one value"
        )


def _reject_if_owned(key: str) -> None:
    """Refuse a key that already has a home: one value, one owner."""
    if key in FORM_OWNED_KEYS:
        raise OverrideRejected(
            f"{key!r} is set by the launch form; change it there instead of "
            "overriding it, so the form always says what the run does"
        )
    if key in SERIES_OWNED_KEYS:
        raise OverrideRejected(
            f"{key!r} belongs to the series' saved model configuration; edit "
            "it on the Physics & model stage"
        )


def _parse_value(key: str, value: str, declared: str) -> str:
    """The value in canonical form, typed against the schema's declaration."""
    if declared == "enum":
        return _parse_choice(key, value)
    if declared == "bool":
        lowered = value.lower()
        if lowered not in ("true", "false"):
            raise OverrideRejected(f"{key!r} is a boolean field; {value!r} is not true/false")
        return lowered
    try:
        number = int(value, 10) if declared == "int" else float(value)
    except ValueError:
        raise OverrideRejected(
            f"{key!r} is declared {declared}; {value!r} does not parse as one"
        ) from None
    if not math.isfinite(number) or abs(number) > MAX_MAGNITUDE:
        raise OverrideRejected(
            f"{key!r}={value!r} is outside the accepted magnitude (|v| <= {MAX_MAGNITUDE:g})"
        )
    _check_declared_bounds(key, number)
    return repr(number)


def _check_declared_bounds(key: str, number: float) -> None:
    """A field-table dial also carries its own [lo, hi]."""
    spec = field_table.by_key(key)
    if spec is None or spec.lo is None or spec.hi is None:
        return
    if not spec.lo <= number <= spec.hi:
        raise OverrideRejected(
            f"{key!r} must be within [{spec.lo:g}, {spec.hi:g}], got {number!r}"
        )


def _parse_choice(key: str, value: str) -> str:
    """One of the options the field table declares for ``key``, verbatim."""
    options = field_table.by_key(key).options
    if value not in options:
        raise OverrideRejected(f"{key!r} must be one of {', '.join(options)}; got {value!r}")
    return value
