"""The equation registry: the single declarative source of the active PDE terms.

The trainer, the API and the UI all read this table instead of hardcoding
physics. These tests pin the Stage-A behaviour the trainer must reproduce
byte-for-byte and the presence (but not-yet-implemented status) of Stage B.
"""

from __future__ import annotations

from naviernet.physics import registry

STAGE_A_FIELDS = ("phi", "u", "v", "s")


def test_stage_a_fields_enable_exactly_the_stage_a_equations():
    ids = [e.id for e in registry.enabled_equations(STAGE_A_FIELDS)]
    assert ids == ["vof", "div", "src", "bc"]


def test_rebalanced_terms_match_the_stage_a_convention():
    eqs = registry.enabled_equations(STAGE_A_FIELDS)
    assert registry.rebalanced_terms(eqs) == ("vof", "div", "bc")


def test_src_is_a_penalty_not_rebalanced():
    src = next(e for e in registry.REGISTRY if e.id == "src")
    assert src.rebalanced is False


def test_stage_b_equations_are_present_but_unimplemented():
    by_id = {e.id: e for e in registry.REGISTRY}
    assert {"mom", "energy"} <= set(by_id)
    for eid in ("mom", "energy"):
        assert by_id[eid].stage == "B"
        assert by_id[eid].implemented is False, "Stage-B lands in a later task"


def test_stage_b_stays_disabled_until_implemented_even_with_its_fields():
    ids = [e.id for e in registry.enabled_equations(("phi", "u", "v", "s", "p", "T"))]
    assert "mom" not in ids and "energy" not in ids


def test_momentum_and_energy_declare_the_fields_they_unlock():
    by_id = {e.id: e for e in registry.REGISTRY}
    assert by_id["mom"].fields_added == ("p",)
    assert by_id["energy"].fields_added == ("T",)


def test_every_equation_carries_ui_metadata():
    for e in registry.REGISTRY:
        assert e.tex, f"{e.id} has no TeX"
        assert e.weight_key
        assert e.stage in ("A", "B")
        assert e.name
