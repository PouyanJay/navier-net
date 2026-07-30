"""The equation registry: the single declarative source of the active PDE terms.

The trainer, the API and the UI all read this table instead of hardcoding
physics. These tests pin the Stage-A behaviour the trainer must reproduce
byte-for-byte and the presence (but not-yet-implemented status) of Stage B.
"""

from __future__ import annotations

import pytest

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


def test_collocation_terms_expose_per_point_residuals_that_mean_to_the_term():
    """Every collocation equation exposes a per-point squared residual (`pointwise`)
    whose mean is exactly the scalar `term` -- the accessor RBA (per-point attention)
    and RAR (residual-adaptive resampling) build on. bc, evaluated on boundary batches,
    has no pointwise residual."""
    import torch

    from naviernet.models.pinn import BubblePINN
    from naviernet.physics.groups import compute_groups

    from .conftest import make_config

    cfg = make_config(
        [
            "model=stage_b",
            "model.hidden=8",
            "model.layers=2",
            "model.fourier_feats=4",
            "model.per_field.p.hidden=8",
            "model.per_field.p.layers=2",
            "model.per_field.T.hidden=8",
            "model.per_field.T.layers=2",
        ]
    )
    torch.manual_seed(0)
    model = BubblePINN(cfg)
    groups = compute_groups(cfg)
    n = 12
    x = torch.rand(n, 3, requires_grad=True)
    ctx = registry.LossContext(model, x, groups=groups)

    equations = registry.enabled_equations(cfg.model.fields)
    coll = registry.collocation_equations(equations)
    assert {e.id for e in coll} == {"vof", "div", "src", "mom", "energy", "evap"}

    by_id = {e.id: e for e in coll}
    for eid in ("vof", "div", "src", "mom", "energy", "evap"):  # explicit so a failure names it
        e = by_id[eid]
        pw = e.pointwise(ctx)
        assert pw.shape == (n, 1), f"{eid} pointwise must be per-point, got {tuple(pw.shape)}"
        assert torch.all(pw >= 0), f"{eid} pointwise must be a squared residual (>= 0)"
        assert e.term(ctx).item() == pytest.approx(pw.mean().item(), rel=1e-6), (
            f"{eid}: term must equal mean(pointwise)"
        )

    bc = next(e for e in equations if e.id == "bc")
    assert bc.pointwise is None, "bc is a boundary term, not a per-point collocation residual"
