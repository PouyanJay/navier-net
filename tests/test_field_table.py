"""The field table: one declaration per user-facing input, held against the
schema dataclasses it names and the copy the web renders from."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from naviernet.config import fields, schema
from naviernet.config.fields import FIELD_TABLE, FieldSpec, hydra_literal
from naviernet.data.preprocess import BAKED_CONDITION_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_COPY = REPO_ROOT / "apps" / "web" / "src" / "lib" / "fieldTable.json"

# Schema fields that are not user inputs: identity, derived from the fluid,
# fixed by the imaging, or structural.
_NOT_INPUTS = {
    "experiment": {
        "name",
        "fluid",
        "T_sat_C",
        "flow_direction",
        "n_frames_raw",
        "n_frames_usable",
        "n_frames_event",
        "excluded_frames",
        "notes",
    },
    "scales": {"L_ref_um"},
    "evaluation": {"stride", "n_traj_points", "threshold", "acceptance"},
}

# String-valued model and training fields the launch request already carries
# as its own typed choices (or that the server fixes), so they need no entry.
_CHOICES_ELSEWHERE = {
    "model.fields",
    "training.val_strategy",
    "training.causal_mode",
    "training.weighting",
    "training.device",
}

_KINDS = {"float": "float", "int": "int", "bool": "bool", "enum": "str"}


def _declared_type(spec: FieldSpec) -> str:
    return str(fields.schema_field(spec.key).type)


def test_every_key_and_name_is_unique():
    keys = [spec.key for spec in FIELD_TABLE]
    names = [spec.name for spec in FIELD_TABLE]

    assert len(set(keys)) == len(keys)
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("spec", FIELD_TABLE, ids=lambda spec: spec.key)
def test_every_entry_names_a_schema_field_of_its_kind(spec: FieldSpec):
    declared = _declared_type(spec)

    expected = _KINDS[spec.kind]
    assert declared == (f"Optional[{expected}]" if spec.optional else expected), (
        f"{spec.key} is declared {spec.kind}{' optional' if spec.optional else ''} "
        f"but the schema says {declared}"
    )


@pytest.mark.parametrize(
    "spec", [s for s in FIELD_TABLE if s.kind == "enum"], ids=lambda s: s.key
)
def test_an_enum_offers_its_schema_default_among_at_least_two_options(spec: FieldSpec):
    assert len(spec.options) >= 2
    assert spec.default in spec.options


@pytest.mark.parametrize(
    "spec", [s for s in FIELD_TABLE if s.kind in ("float", "int")], ids=lambda s: s.key
)
def test_a_number_has_bounds_that_admit_its_default(spec: FieldSpec):
    assert spec.lo is not None and spec.hi is not None and spec.lo < spec.hi
    if spec.default is not None:
        assert spec.lo <= spec.default <= spec.hi


def _user_facing(group: str) -> set[str]:
    owner = fields._GROUPS[group]
    return {f.name for f in dataclasses.fields(owner)} - _NOT_INPUTS.get(group, set())


@pytest.mark.parametrize("group", ["experiment", "scales", "evaluation"])
def test_every_user_facing_field_of_the_conditions_and_evaluation_groups_is_declared(group):
    declared = {spec.name for spec in FIELD_TABLE if spec.key.startswith(f"{group}.")}

    assert _user_facing(group) <= declared, sorted(_user_facing(group) - declared)


def test_every_acceptance_threshold_is_declared():
    declared = {
        spec.name for spec in FIELD_TABLE if spec.key.startswith("evaluation.acceptance.")
    }

    assert {f.name for f in dataclasses.fields(schema.AcceptanceConfig)} == declared


@pytest.mark.parametrize("group", ["model", "training"])
def test_every_string_choice_of_the_model_and_training_schemas_is_reachable(group):
    """A string-valued option is refused by the overrides gate unless the table
    declares it, so every such schema field must be in the table (or carried
    by the request model's own typed choices)."""
    owner = fields._GROUPS[group]
    strings = {f"{group}.{f.name}" for f in dataclasses.fields(owner) if str(f.type) == "str"}
    declared = {spec.key for spec in FIELD_TABLE}

    assert strings - _CHOICES_ELSEWHERE <= declared, sorted(
        strings - _CHOICES_ELSEWHERE - declared
    )


def test_the_baked_fields_are_the_tables_baked_conditions():
    assert (
        tuple(spec.name for spec in fields.specs(section="conditions") if spec.baked)
        == BAKED_CONDITION_FIELDS
    )


def test_hydra_literals_compose_back_to_the_value():
    from omegaconf import OmegaConf

    composed = OmegaConf.from_dotlist(
        [
            f"a={hydra_literal(None)}",
            f"b={hydra_literal(True)}",
            f"c={hydra_literal(0.5)}",
            f"d={hydra_literal(3)}",
            f"e={hydra_literal('two_phase')}",
        ]
    )

    assert OmegaConf.to_container(composed) == {
        "a": None,
        "b": True,
        "c": 0.5,
        "d": 3,
        "e": "two_phase",
    }


def test_the_web_copy_of_the_table_is_current():
    """The web renders from a committed copy; a stale copy fails here, naming
    the script that refreshes it."""
    assert WEB_COPY.is_file(), "run scripts/export_field_table.py"

    assert json.loads(WEB_COPY.read_text()) == fields.export(), (
        "apps/web/src/lib/fieldTable.json is stale: run scripts/export_field_table.py"
    )
