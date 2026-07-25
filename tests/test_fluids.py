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


def test_selecting_a_fluid_updates_the_derived_saturation_temperature():
    """Overriding the fluid group flows through to the experiment's T_sat."""
    cfg = make_config(["fluid=water"])
    assert cfg.fluid.name == "Water"
    assert cfg.fluid.T_sat_C == pytest.approx(100.0)
    assert cfg.experiment.T_sat_C == pytest.approx(100.0)
    # The label the UI and video title read follows the chosen group.
    assert cfg.experiment.fluid == "Water"
