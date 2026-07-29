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


def test_stage_b_equations_are_present_and_implemented():
    by_id = {e.id: e for e in registry.REGISTRY}
    assert {"mom", "energy", "evap"} <= set(by_id)
    for eid in ("mom", "energy", "evap"):
        assert by_id[eid].stage == "B"
        assert by_id[eid].implemented is True


def test_full_stage_b_fields_enable_every_equation():
    ids = [e.id for e in registry.enabled_equations(("phi", "u", "v", "s", "p", "T"))]
    assert ids == ["vof", "div", "src", "bc", "mom", "energy", "evap"]


def test_stage_b_rebalances_momentum_and_energy_but_not_evaporation():
    eqs = registry.enabled_equations(("phi", "u", "v", "s", "p", "T"))
    assert registry.rebalanced_terms(eqs) == ("vof", "div", "bc", "mom", "energy")


def test_stage_b_terms_are_exactly_the_warmup_gated_physics():
    """The terms the in-run warm-up holds off are the Stage-B equations, and only
    those -- the Stage-A objective is never gated."""
    eqs = registry.enabled_equations(("phi", "u", "v", "s", "p", "T"))
    assert registry.stage_b_terms(eqs) == ("mom", "energy", "evap")
    # A Stage-A-only model has nothing to gate.
    assert registry.stage_b_terms(registry.enabled_equations(("phi", "u", "v", "s"))) == ()


def test_momentum_needs_pressure_and_can_be_enabled_without_temperature():
    """The toggles are independent: pressure unlocks momentum, temperature unlocks
    energy + evaporation. Enabling one must not require the other."""
    ids = [e.id for e in registry.enabled_equations(("phi", "u", "v", "s", "p"))]
    assert "mom" in ids
    assert "energy" not in ids and "evap" not in ids


def test_energy_needs_temperature_and_can_be_enabled_without_pressure():
    ids = [e.id for e in registry.enabled_equations(("phi", "u", "v", "s", "T"))]
    assert "energy" in ids and "evap" in ids
    assert "mom" not in ids


def test_momentum_and_energy_declare_the_fields_they_unlock():
    by_id = {e.id: e for e in registry.REGISTRY}
    assert by_id["mom"].fields_added == ("p",)
    assert by_id["energy"].fields_added == ("T",)


def test_stage_a_equations_are_core_and_stage_b_are_not():
    """`core` drives the UI's locked-on state: the Stage-A objective is always on."""
    by_id = {e.id: e for e in registry.REGISTRY}
    assert all(by_id[eid].core for eid in ("vof", "div", "src", "bc"))
    assert not any(by_id[eid].core for eid in ("mom", "energy", "evap"))


def test_every_equation_carries_ui_metadata():
    for e in registry.REGISTRY:
        assert e.tex, f"{e.id} has no TeX"
        assert e.weight_key
        assert e.stage in ("A", "B")
        assert e.name
