"""Fluid configs and the T_sat-derives-from-fluid contract.

Saturation temperature is a property of the working fluid at the operating
pressure (atmospheric here), not a free operating input. The fluid group is the
single source of truth; the experiment derives its ``T_sat_C`` from it.
"""

from __future__ import annotations

import pytest

from .conftest import make_config


def test_fluid_carries_its_own_saturation_temperature(cfg):
    """T_sat_C lives on the fluid group (FC-72 saturates at 56.6 C, 1 atm)."""
    assert cfg.fluid.T_sat_C == pytest.approx(56.6)


def test_experiment_saturation_temperature_derives_from_fluid(cfg):
    """The experiment's T_sat is the fluid's T_sat, not an independent value."""
    assert cfg.experiment.T_sat_C == pytest.approx(cfg.fluid.T_sat_C)


# (id, display name, atmospheric saturation temperature in C)
EXPECTED_FLUIDS = [
    ("fc72", "FC-72", 56.6),
    ("fc77", "FC-77", 97.0),
    ("hfe7100", "HFE-7100", 61.0),
    ("novec649", "Novec 649", 49.0),
    ("water", "Water", 100.0),
]


@pytest.mark.parametrize(("fluid_id", "name", "t_sat"), EXPECTED_FLUIDS)
def test_each_fluid_composes_with_its_saturation_temperature(fluid_id, name, t_sat):
    cfg = make_config([f"fluid={fluid_id}"])
    assert cfg.fluid.name == name
    assert cfg.fluid.T_sat_C == pytest.approx(t_sat)
    assert cfg.experiment.T_sat_C == pytest.approx(t_sat)


@pytest.mark.parametrize("fluid_id", [f[0] for f in EXPECTED_FLUIDS])
def test_dimensionless_groups_are_finite_for_every_fluid(fluid_id):
    """Every characterised fluid yields physically sane (finite, positive)
    dimensionless groups, not NaN/inf from a missing or zero property."""
    import math

    from naviernet.physics.groups import compute_groups

    groups = compute_groups(make_config([f"fluid={fluid_id}"]))
    for key, value in groups.items():
        assert math.isfinite(value), f"{fluid_id}: {key} is not finite"
    for key in ("Re", "We", "Pr", "Ca", "rho_ratio", "mu_ratio"):
        assert groups[key] > 0, f"{fluid_id}: {key} must be positive"


def test_selecting_a_fluid_updates_the_derived_saturation_temperature():
    """Overriding the fluid group flows through to the experiment's T_sat."""
    cfg = make_config(["fluid=water"])
    assert cfg.fluid.name == "Water"
    assert cfg.fluid.T_sat_C == pytest.approx(100.0)
    assert cfg.experiment.T_sat_C == pytest.approx(100.0)
    # The label the UI and video title read follows the chosen group.
    assert cfg.experiment.fluid == "Water"
