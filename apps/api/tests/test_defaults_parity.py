"""RunLaunchRequest defaults vs configs/training/stage_a.yaml: one configuration.

The launch request's docstring promises its numeric defaults map 1:1 onto
``cfg.training`` as the CLI would compose it, and the web form's defaults chain
off the same file -- but nothing ever checked the promise, so the copies could
drift silently. This is that check, over every field the two actually share.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args, get_origin

import pytest
import yaml
from naviernet_api.models import ConditionsUpdate, OperatingConditions, RunLaunchRequest

from naviernet.config.fields import schema_default, specs

REPO_ROOT = Path(__file__).parents[3]

# The launch fields whose defaults claim to mirror the CLI's Stage-A training
# config. Fields the platform deliberately owns differently (val_fraction is a
# platform axis the YAML does not carry; device is forced to cpu server-side)
# are not shared and are not listed.
SHARED_FIELDS = (
    "steps",
    "lr",
    "lr_halflife",
    "n_data",
    "n_coll",
    "n_bc",
    "holdout_frame",
    "rebalance_every",
    "log_every",
    "seed",
)


def test_launch_defaults_match_the_cli_training_config():
    config = yaml.safe_load((REPO_ROOT / "configs" / "training" / "stage_a.yaml").read_text())
    request = RunLaunchRequest(dataset="any")

    # Bare == on the floats is deliberate: both sides are hand-written literals
    # parsed independently, and the test exists to catch drift between exactly
    # those literals -- approx would let one side be rounded without notice.
    for field in SHARED_FIELDS:
        assert getattr(request, field) == config[field], (
            f"RunLaunchRequest.{field} default {getattr(request, field)!r} has "
            f"drifted from configs/training/stage_a.yaml's {config[field]!r}"
        )
    for term in ("data", "vof", "div", "src", "bc"):
        assert getattr(request.weights, term) == config["weights"][term], (
            f"weights.{term} default has drifted from stage_a.yaml"
        )


# --- the field table vs the request models ----------------------------------


def _literal_options(model: type, name: str) -> tuple[str, ...] | None:
    """The options of a `Literal[...]` (possibly `| None`) annotation, else None."""
    annotation = model.model_fields[name].annotation
    for candidate in (annotation, *get_args(annotation)):
        if get_origin(candidate) is Literal:
            return tuple(get_args(candidate))
    return None


@pytest.mark.parametrize("spec", specs(owner="form"), ids=lambda spec: spec.key)
def test_a_form_owned_choice_defaults_to_the_schema_value(spec):
    """Every form-owned entry of the field table is a launch-request field whose
    default is the schema's, so an untouched request composes byte-for-byte."""
    request = RunLaunchRequest(dataset="any")

    assert spec.name in RunLaunchRequest.model_fields, f"RunLaunchRequest lacks {spec.name}"
    assert getattr(request, spec.name) == schema_default(spec.key)
    if spec.kind == "enum":
        assert _literal_options(RunLaunchRequest, spec.name) == spec.options


@pytest.mark.parametrize("spec", specs(section="conditions"), ids=lambda spec: spec.key)
def test_a_series_condition_is_editable_and_reported(spec):
    """Every condition of the field table can be sent on a conditions edit and
    comes back on the operating conditions (U_ref under its unit)."""
    reported = "U_ref_m_s" if spec.name == "U_ref" else spec.name

    assert spec.name in ConditionsUpdate.model_fields, f"ConditionsUpdate lacks {spec.name}"
    assert reported in OperatingConditions.model_fields, f"OperatingConditions lacks {reported}"
    if spec.kind == "enum":
        assert _literal_options(ConditionsUpdate, spec.name) == spec.options
