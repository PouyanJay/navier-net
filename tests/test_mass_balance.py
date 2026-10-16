"""The cross-section mass balance functional."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from naviernet.data.dataset import Domain
from naviernet.physics.mass_balance import axial_flux_defect, build_plan, mass_balance_loss
from naviernet.training import train

from .conftest import staged_run

U_IN = 0.15
SPAN = (0.0, 1.0)
DOMAIN = Domain(
    x_min=0.0,
    x_max=2.0,
    y_min=0.1,
    y_max=0.9,
    t_min=0.0,
    t_max=1.0,
    x_pin=0.2,
    y_wall_lo=SPAN[0],
    y_wall_hi=SPAN[1],
)
QUADRATURE = SimpleNamespace(
    mass_balance_times=3, mass_balance_stations=9, mass_balance_grid=17
)


class _Flow:
    """A stand-in for the model with a given axial velocity and source."""

    def __init__(self, u, s) -> None:
        self._u, self._s = u, s

    def velocity(self, x, c=None):
        return self._u(x), torch.zeros_like(x[:, :1])

    def source(self, x, c=None):
        return self._s(x)


def _plan():
    return build_plan(DOMAIN, SPAN, {"u_inlet_star": U_IN}, QUADRATURE, "cpu")


def test_plan_spans_wall_to_wall_at_fixed_stations_and_times():
    plan = _plan()

    assert plan.points.shape == (3 * 9 * 17, 3)
    assert float(plan.points[:, 1].min()) == pytest.approx(SPAN[0])
    assert float(plan.points[:, 1].max()) == pytest.approx(SPAN[1])
    assert plan.q_in == pytest.approx(U_IN * (SPAN[1] - SPAN[0]))
    assert torch.equal(plan.stations, torch.linspace(DOMAIN.x_min, DOMAIN.x_max, 9))


def test_defect_vanishes_when_the_flux_grows_with_the_source():
    """A uniform source ``c`` creates ``c * width * (x - x_min)`` of volume upstream
    of ``x``, so ``u = u_in + c (x - x_min)`` conserves volume at every station."""
    rate = 0.3
    flow = _Flow(
        lambda x: U_IN + rate * (x[:, 0:1] - DOMAIN.x_min),
        lambda x: torch.full_like(x[:, :1], rate),
    )

    defect = axial_flux_defect(flow, _plan())

    assert defect.shape == (3, 9)
    assert defect.abs().max().item() < 1e-5


def test_defect_measures_a_lost_inlet_flux():
    flow = _Flow(
        lambda x: torch.full_like(x[:, :1], 0.9 * U_IN), lambda x: torch.zeros_like(x[:, :1])
    )

    loss, record = mass_balance_loss(flow, _plan())

    assert loss.item() == pytest.approx(0.01, rel=1e-4)
    assert record["mass_bal_max"] == pytest.approx(0.1, rel=1e-4)
    assert set(record) == {"mass_bal", "mass_bal_max"}


def test_a_run_with_the_mass_balance_logs_the_term(tmp_path):
    cfg, paths = staged_run(tmp_path, ["training.mass_balance=true", "training.log_every=1"])

    _, _, state = train(cfg, paths)

    assert all(
        "mass_bal" in record and np.isfinite(record["mass_bal"]) for record in state["hist"]
    )
