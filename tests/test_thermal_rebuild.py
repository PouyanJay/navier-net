"""The sharp thermal model: conditions, heating, the gated temperature, and the
continuity source taken from the evaporation flux through the front."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BubblePINN
from naviernet.physics import registry
from naviernet.physics.groups import compute_groups
from naviernet.physics.heating import estimate_background_flux, heater_footprint
from naviernet.physics.residuals import (
    ThermalBatches,
    energy_liquid_residual,
    thermal_boundary_losses,
)
from naviernet.training import _geometry_priors, train

from .conftest import make_config, staged_capsule_run

STAGE_B_FIELDS = ("phi", "u", "v", "s", "p", "T")
THERMAL_SERIES = [
    "experiment.T_inlet_C=56.6",
    "experiment.T_init_C=58.0",
    "experiment.q_cavity_W_cm2=2.0",
    "experiment.heater_radius_um=100",
]
SHARP_THERMAL = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.thermal_model=sharp",
    *THERMAL_SERIES,
]


def test_groups_take_the_superheat_scale_from_the_initial_liquid():
    groups = compute_groups(make_config(THERMAL_SERIES))

    assert groups["dT_ref"] == pytest.approx(58.0 - 56.6)
    assert groups["theta_inlet"] == pytest.approx(0.0)
    assert groups["theta_init"] == pytest.approx(1.0)
    assert groups["heater_radius_star"] == pytest.approx(100.0 / 300.0)
    assert groups["q_cavity_star"] > 0
    assert groups["mass_flux_to_source"] == pytest.approx(
        groups["Ja"] / groups["Pe"] * (groups["rho_ratio"] - 1.0)
    )


def test_a_configured_superheat_scale_overrides_the_initial_liquid():
    groups = compute_groups(make_config([*THERMAL_SERIES, "scales.dT_ref_K=2.0"]))

    assert groups["dT_ref"] == pytest.approx(2.0)
    assert groups["theta_init"] == pytest.approx(0.7)


def test_a_legacy_series_keeps_the_conduction_scale_and_no_initial_condition(cfg):
    groups = compute_groups(cfg)

    assert groups["dT_ref"] == pytest.approx(28.74, rel=1e-3)
    assert groups["theta_inlet"] == 0.0
    assert "theta_init" not in groups and "q_cavity_star" not in groups


def test_heater_disk_integrates_to_its_area_and_switches_off_after_its_time():
    radius = 0.2
    groups = {"heater_radius_star": radius, "heater_on_star": 0.5}
    axis = torch.linspace(-0.5, 0.5, 401)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    cell = float(axis[1] - axis[0]) ** 2

    def at(time: float) -> torch.Tensor:
        points = torch.stack(
            [grid_x.ravel(), grid_y.ravel(), torch.full_like(grid_x.ravel(), time)], dim=1
        )
        return heater_footprint(points, groups, (0.0, 0.0))

    assert float(at(0.1).sum()) * cell == pytest.approx(np.pi * radius**2, rel=0.02)
    assert float(at(0.9).abs().max()) == 0.0, "gated off after heater_on_star"


def _model(tmp_path, overrides):
    cfg, paths = staged_capsule_run(tmp_path, overrides)
    data = BubbleDataset(cfg, paths)
    torch.manual_seed(0)
    return cfg, data, BubblePINN(cfg, geometry=_geometry_priors(cfg, data))


def test_gated_temperature_is_saturated_in_the_vapour_and_on_the_front(tmp_path):
    _, data, model = _model(tmp_path, SHARP_THERMAL)
    t = float(data.t[1])
    front = model.front(torch.tensor([[t]]), 16, 4)
    net = model.nets["T"]

    on_front = model.temperature(front.points.detach())
    spine = torch.tensor([[0.5, 0.5, t]])  # the capsule's axis, radius 0.12 away from the front
    liquid = torch.tensor([[0.05, 0.05, t]])  # the far corner

    assert (
        on_front.abs().max() <= model.theta_gate_width * net(front.points.detach()).abs().max()
    )
    assert model.temperature(spine).abs().item() < 1e-3 * net(spine).abs().item()
    expected = -model.phi(liquid) * net(liquid)
    assert model.temperature(liquid).item() == pytest.approx(expected.item(), rel=1e-3)


def test_temperature_gate_cannot_move_the_geometry(tmp_path):
    _, data, model = _model(tmp_path, SHARP_THERMAL)
    torch.manual_seed(1)
    x = torch.rand(32, 3) * torch.tensor([2.0, 1.0, 0.5])

    model.temperature(x).sum().backward()

    geometry = list(model.nets["phi"].parameters())
    assert all(p.grad is None or torch.all(p.grad == 0) for p in geometry)
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in model.nets["T"].parameters()
    )


def test_thermal_source_integrates_to_the_flux_along_the_front(tmp_path):
    cfg, data, model = _model(tmp_path, [*SHARP_THERMAL, "model.source_mode=thermal"])
    groups = compute_groups(cfg)
    t = torch.tensor(float(data.t[1]))
    flux, weights, _ = model._front_flux(t, None)
    expected = groups["mass_flux_to_source"] * float((flux * weights).sum().detach())
    xs, ys = torch.linspace(0.0, 2.0, 241), torch.linspace(0.0, 1.0, 121)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    points = torch.stack(
        [grid_x.ravel(), grid_y.ravel(), torch.full_like(grid_x.ravel(), float(t))], dim=1
    )

    source = model.source(points).detach().reshape(241, 121)
    integral = float(torch.trapezoid(torch.trapezoid(source, ys, dim=1), xs))

    assert integral == pytest.approx(expected, rel=0.03)
    assert torch.allclose(model.source(points), model.thermal_source(points))


def test_thermal_source_is_concentrated_on_the_interface_band(tmp_path):
    _, data, model = _model(tmp_path, [*SHARP_THERMAL, "model.source_mode=thermal"])
    t = float(data.t[1])
    far = torch.tensor([[0.05, 0.05, t], [1.9, 0.95, t]])

    assert model.source(far).abs().max().item() < 1e-6


class _LiquidField:
    """A liquid-only stand-in with a given temperature and velocity."""

    heater_centre = (0.0, 0.0)
    q_wall_scale = 1.0

    def __init__(self, theta, u=None, v=None) -> None:
        self._theta = theta
        self._u = u or (lambda x: 0.0 * x[:, 0:1])
        self._v = v or (lambda x: 0.0 * x[:, 1:2])

    def alpha(self, x, c=None):
        return torch.zeros_like(x[:, :1])

    def temperature(self, x, c=None):
        return self._theta(x)

    def velocity(self, x, c=None):
        return self._u(x), self._v(x)


def test_liquid_energy_residual_vanishes_for_pure_conduction(cfg):
    groups = {**compute_groups(cfg), "q_wall_star": 0.0}
    pe = groups["Pe"]
    conduction = _LiquidField(lambda x: torch.sin(x[:, 0:1]) * torch.exp(-x[:, 2:3] / pe))
    torch.manual_seed(0)
    x = torch.rand(64, 3, requires_grad=True)

    residual = energy_liquid_residual(conduction, x, groups)

    assert residual.abs().max().item() < 1e-5


def test_thermal_boundary_losses_score_inlet_initial_and_walls():
    uniform = _LiquidField(lambda x: 0.3 + 0.0 * x[:, 0:1])
    torch.manual_seed(0)
    batches = ThermalBatches(torch.rand(16, 3), torch.rand(16, 3), torch.rand(16, 3))

    adiabatic = thermal_boundary_losses(
        uniform, {"theta_inlet": 0.3, "theta_init": 1.0}, batches
    )
    isothermal = thermal_boundary_losses(
        uniform, {"theta_inlet": 0.3, "theta_init": 1.0, "theta_wall": 0.5}, batches
    )

    assert adiabatic.item() == pytest.approx(0.49), "only the initial condition is missed"
    assert isothermal.item() == pytest.approx(0.49 + 0.04)


def test_sharp_thermal_swaps_the_energy_pair_for_the_liquid_model():
    def ids(**flags):
        return {
            e.id
            for e in registry.enabled_equations(STAGE_B_FIELDS, sharp_interface=True, **flags)
        }

    base, sharp = ids(), ids(sharp_thermal=True)
    thermal = ids(sharp_thermal=True, thermal_source=True)

    assert {"energy", "evap"} <= base and not {"energy", "evap"} & sharp
    assert {"energy_liquid", "theta_bc", "evap_flux", "src"} <= sharp
    assert {"energy_liquid", "theta_bc"} <= thermal
    assert not {"evap_flux", "src"} & thermal, "a thermal source has no free source to pull"


def test_flags_from_reads_every_registry_gate():
    cfg = make_config(
        [
            *SHARP_THERMAL,
            "model.momentum_closure=brinkman",
            "model.pressure_representation=two_phase",
            "model.source_mode=thermal",
        ]
    )

    flags = registry.flags_from(cfg.model)

    assert flags == {
        "sharp_interface": True,
        "liquid_film": False,
        "momentum_closure": "brinkman",
        "two_phase_pressure": True,
        "sharp_thermal": True,
        "thermal_source": True,
    }


def test_background_flux_estimate_is_the_energy_balance_of_the_growth(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, THERMAL_SERIES)
    data = BubbleDataset(cfg, paths)

    estimate = estimate_background_flux(data, cfg, compute_groups(cfg))

    assert estimate["latent_power_mW_mean"] > 0, "the capsule grows"
    assert estimate["cavity_power_mW"] == pytest.approx(2.0 * 1e4 * np.pi * (100e-6) ** 2 * 1e3)
    assert estimate["q_wall_estimate_W_cm2"] >= 0


def test_fitted_background_flux_starts_at_the_configured_value_and_stays_bounded(tmp_path):
    fit = [
        "experiment.q_wall_mode=fit",
        "experiment.q_wall_W_cm2=0.5",
        "experiment.q_wall_max_W_cm2=5.0",
    ]
    _, _, model = _model(tmp_path, [*SHARP_THERMAL, *fit])

    assert float(model.q_wall_scale) == pytest.approx(1.0, abs=1e-6)
    with torch.no_grad():
        model._q_wall_scale_raw.fill_(100.0)
        assert float(model.q_wall_scale) <= 10.0
        model._q_wall_scale_raw.fill_(-100.0)
        assert float(model.q_wall_scale) >= 0.0


def test_fitted_background_flux_needs_the_sharp_thermal_model(tmp_path):
    with pytest.raises(ValueError, match="q_wall_mode=fit"):
        _model(tmp_path, ["model=stage_b", "experiment.q_wall_mode=fit"])


def test_sharp_thermal_requires_the_front_and_the_t_field(tmp_path):
    with pytest.raises(ValueError, match="thermal_model=sharp"):
        _model(tmp_path, ["model.thermal_model=sharp"])


def test_a_sharp_thermal_run_trains_and_logs_its_terms(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            *SHARP_THERMAL,
            "model.source_mode=thermal",
            "experiment.q_wall_mode=fit",
            "experiment.q_wall_W_cm2=0.5",
            "training.collocation_time_bins=2",
            "training.log_every=1",
        ],
    )

    _, _, state = train(cfg, paths)

    record = state["hist"][-1]
    assert {"energy_liquid", "theta_bc"} <= record.keys()
    assert not {"energy", "evap", "src", "evap_flux"} & record.keys()
    assert all(np.isfinite(v) for v in record.values())


def test_thermal_source_requires_binned_collocation_times(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [*SHARP_THERMAL, "model.source_mode=thermal", "training.collocation_time_bins=0"],
    )

    with pytest.raises(ValueError, match="collocation_time_bins"):
        train(cfg, paths)


def test_the_thermal_source_waits_for_the_physics_to_engage(tmp_path):
    """Through the warm-up the free source carries continuity; the thermal
    source takes over at the step the Stage-B physics engages, and evaluation
    (no trainer) reads the thermal source from the start."""
    from naviernet.training import _engage_thermal_source

    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            *SHARP_THERMAL,
            *THERMAL_SERIES,
            "model.source_mode=thermal",
            "training.stage_b_warmup_steps=3",
        ],
    )
    data = BubbleDataset(cfg, paths)
    torch.manual_seed(0)
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    x = torch.rand(16, 3, requires_grad=True)
    assert model.thermal_source_engaged, "evaluation reads the thermal source"
    with torch.enable_grad():
        thermal = model.thermal_source(x).detach()

    _engage_thermal_source(model, 3, cfg.training, ("energy_liquid",))
    held_back = model.source(x).detach()
    _engage_thermal_source(model, 4, cfg.training, ("energy_liquid",))
    with torch.enable_grad():
        engaged = model.source(x).detach()

    assert not torch.allclose(held_back, thermal), "the warm-up source is the free net"
    assert torch.allclose(held_back, model._even(model.nets["s"], x, None).detach())
    assert torch.allclose(engaged, thermal)
    _engage_thermal_source(model, 1, cfg.training, ())  # no Stage-B terms: no warm-up
    assert model.thermal_source_engaged
