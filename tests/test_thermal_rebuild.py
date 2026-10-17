"""The sharp thermal model: conditions, the wall flux split by phase, the gated
temperature, the continuity source taken from the evaporation flux through the
front, and the film evaporation under the footprint."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BubblePINN
from naviernet.physics import registry
from naviernet.physics.groups import compute_groups
from naviernet.physics.heating import growth_budget
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
]
FOOTPRINT = [
    "model.footprint_evaporation=true",
    "experiment.wet_fraction_mode=fit",
    "experiment.wet_fraction=0.8",
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
    assert groups["mass_flux_to_source"] == pytest.approx(
        groups["Ja"] / groups["Pe"] * (groups["rho_ratio"] - 1.0)
    )


def test_the_footprint_source_coefficient_is_the_wall_flux_over_the_latent_heat():
    """``q'' (1/rho_v - 1/rho_l) L / (h_lv H U)``: the vapour volume the wall flux
    under the bubble creates per unit footprint, as a dilatation on U/L."""
    cfg = make_config(["experiment.q_wall_W_cm2=2.0"])
    fluid, exp, scales = cfg.fluid, cfg.experiment, cfg.scales

    groups = compute_groups(cfg)

    expected = (
        2.0e4
        * (1.0 / fluid.rho_v - 1.0 / fluid.rho_l)
        * scales.L_ref_um
        * 1e-6
        / (fluid.h_lv * exp.channel_height_um * 1e-6 * scales.U_ref)
    )
    assert groups["footprint_source_star"] == pytest.approx(expected)
    assert groups["wet_fraction_init"] == 1.0, "the schema's intact film"


def test_a_fitted_wet_fraction_must_start_strictly_inside_the_interval():
    with pytest.raises(ValueError, match="strictly inside"):
        compute_groups(make_config(["experiment.wet_fraction_mode=fit"]))
    with pytest.raises(ValueError, match="wet_fraction=1.5"):
        compute_groups(make_config(["experiment.wet_fraction=1.5"]))
    with pytest.raises(ValueError, match="wet_fraction_mode"):
        compute_groups(make_config(["experiment.wet_fraction_mode=guess"]))


def test_a_configured_superheat_scale_overrides_the_initial_liquid():
    groups = compute_groups(make_config([*THERMAL_SERIES, "scales.dT_ref_K=2.0"]))

    assert groups["dT_ref"] == pytest.approx(2.0)
    assert groups["theta_init"] == pytest.approx(0.7)


def test_a_legacy_series_keeps_the_conduction_scale_and_no_initial_condition(cfg):
    groups = compute_groups(cfg)

    assert groups["dT_ref"] == pytest.approx(28.74, rel=1e-3)
    assert groups["theta_inlet"] == 0.0
    assert "theta_init" not in groups


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


def test_growth_budget_is_the_energy_balance_of_the_growth(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*THERMAL_SERIES, "experiment.q_wall_W_cm2=2.0"])
    data = BubbleDataset(cfg, paths)

    budget = growth_budget(data, cfg, compute_groups(cfg))

    assert budget["latent_power_mW_mean"] > 0, "the capsule grows"
    areas = [
        float(((data.alpha[i] > 0.5) & (data.valid[i] > 0)).sum()) for i in range(data.n_event)
    ]
    pixel = float(data.x[1] - data.x[0]) * float(data.y[1] - data.y[0]) * (300e-6) ** 2
    assert budget["footprint_power_mW"] == pytest.approx(
        [2.0e4 * a * pixel * 1e3 for a in areas], rel=1e-6
    )
    assert all(0.0 <= f <= 1.0 for f in budget["implied_wet_fraction"])
    assert 0.0 < budget["wet_fraction_estimate"] <= 1.0


def test_fitted_wet_fraction_starts_at_its_declared_value_and_stays_bounded(tmp_path):
    _, _, model = _model(tmp_path, [*SHARP_THERMAL, *FOOTPRINT])

    assert float(model.wet_fraction.detach()) == pytest.approx(0.8, abs=1e-6)
    with torch.no_grad():
        model._wet_fraction_raw.fill_(100.0)
        assert float(model.wet_fraction) <= 1.0
        model._wet_fraction_raw.fill_(-100.0)
        assert float(model.wet_fraction) >= 0.0


def test_a_fixed_wet_fraction_is_the_declared_value_and_trains_nothing(tmp_path):
    fixed = ["model.footprint_evaporation=true", "experiment.wet_fraction=0.5"]
    _, _, model = _model(tmp_path, [*SHARP_THERMAL, *fixed])

    assert model.wet_fraction == 0.5
    assert not hasattr(model, "_wet_fraction_raw")


def test_footprint_evaporation_needs_the_sharp_thermal_model(tmp_path):
    with pytest.raises(ValueError, match="footprint_evaporation"):
        _model(tmp_path, ["model=stage_b", "model.footprint_evaporation=true"])


def test_footprint_source_is_the_wall_flux_under_the_bubble(tmp_path):
    """``footprint_source_star * f_wet * alpha``: the source integrates to the
    coefficient times the wet fraction times the vapour area, it is zero in the
    liquid, and the thermal source carries it."""
    cfg, data, model = _model(
        tmp_path,
        [
            *SHARP_THERMAL,
            "model.source_mode=thermal",
            "model.footprint_evaporation=true",
            "experiment.wet_fraction=0.5",
        ],
    )
    groups = compute_groups(cfg)
    t = float(data.t[1])
    xs, ys = torch.linspace(0.0, 2.0, 241), torch.linspace(0.0, 1.0, 121)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    points = torch.stack(
        [grid_x.ravel(), grid_y.ravel(), torch.full_like(grid_x.ravel(), t)], dim=1
    )

    footprint = model.footprint_source(points).detach().reshape(241, 121)
    vapour = model.alpha(points).detach().reshape(241, 121)
    integral = float(torch.trapezoid(torch.trapezoid(footprint, ys, dim=1), xs))
    area = float(torch.trapezoid(torch.trapezoid(vapour, ys, dim=1), xs))

    assert integral == pytest.approx(groups["footprint_source_star"] * 0.5 * area, rel=1e-4)
    # The far liquid: alpha's sigmoid tail, not the source, sets the residue.
    far = torch.tensor([[0.05, 0.05, t]])
    assert model.footprint_source(far).item() == pytest.approx(
        groups["footprint_source_star"] * 0.5 * model.alpha(far).item(), rel=1e-5
    )
    assert model.alpha(far).item() < 1e-3
    inside = torch.tensor([[0.5, 0.5, t]])  # the capsule's axis, 2.4 eps from the front
    assert model.footprint_source(inside).item() == pytest.approx(
        groups["footprint_source_star"] * 0.5 * model.alpha(inside).item(), rel=1e-5
    )
    assert model.alpha(inside).item() > 0.9
    assert torch.allclose(model.source(points), model.thermal_source(points))


def test_footprint_source_cannot_move_the_geometry_but_trains_the_wet_fraction(tmp_path):
    _, data, model = _model(tmp_path, [*SHARP_THERMAL, *FOOTPRINT])
    torch.manual_seed(1)
    x = torch.rand(32, 3) * torch.tensor([2.0, 1.0, 0.5])

    model.footprint_source(x).sum().backward()

    geometry = list(model.nets["phi"].parameters())
    assert all(p.grad is None or torch.all(p.grad == 0) for p in geometry)
    assert model._wet_fraction_raw.grad is not None and model._wet_fraction_raw.grad.item() != 0


def test_sharp_thermal_requires_the_front_and_the_t_field(tmp_path):
    with pytest.raises(ValueError, match="thermal_model=sharp"):
        _model(tmp_path, ["model.thermal_model=sharp"])


def test_a_sharp_thermal_run_trains_and_logs_its_terms(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            *SHARP_THERMAL,
            "model.source_mode=thermal",
            *FOOTPRINT,
            "training.collocation_time_bins=2",
            "training.log_every=1",
        ],
    )

    _, _, state = train(cfg, paths)

    record = state["hist"][-1]
    assert {"energy_liquid", "theta_bc", "grad_norm"} <= record.keys()
    assert not {"energy", "evap", "src", "evap_flux"} & record.keys()
    assert all(np.isfinite(v) for v in record.values())
    assert record["grad_norm"] > 0


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


def test_one_way_advection_trains_the_temperature_and_leaves_the_flow_alone(tmp_path):
    """With the velocity detached in the advection term, the liquid energy
    residual sends no gradient to the velocity networks and still trains the
    temperature; without it, it reaches both."""
    from naviernet.physics.residuals import energy_liquid_residual

    cfg, data, model = _model(tmp_path, SHARP_THERMAL)
    groups = compute_groups(cfg)
    torch.manual_seed(2)
    x = (torch.rand(64, 3) * torch.tensor([2.0, 1.0, 0.5])).requires_grad_(True)
    velocity = [*model.nets["u"].parameters(), *model.nets["v"].parameters()]
    temperature = list(model.nets["T"].parameters())

    def reach(one_way: bool) -> tuple[float, float]:
        loss = (energy_liquid_residual(model, x, groups, one_way=one_way) ** 2).mean()
        grads = torch.autograd.grad(loss, velocity + temperature, allow_unused=True)
        norm = [0.0 if g is None else float(g.norm()) for g in grads]
        return sum(norm[: len(velocity)]), sum(norm[len(velocity) :])

    flow_two_way, heat_two_way = reach(False)
    flow_one_way, heat_one_way = reach(True)

    assert flow_two_way > 0, "the two-way residual reaches the flow"
    assert flow_one_way == 0.0, "the one-way residual never does"
    assert heat_one_way == pytest.approx(heat_two_way, rel=1e-6), (
        "the temperature trains the same"
    )


def test_the_saturating_gate_reads_the_network_far_from_the_bubble(tmp_path):
    """Zero on the front like the linear gate, but equal to the network beyond a
    few gate lengths, so the far field does not scale with the distance."""
    cfg, data, model = _model(
        tmp_path,
        [*SHARP_THERMAL, "model.theta_gate=saturating", "model.theta_gate_length=0.02"],
    )
    t = float(data.t[1])
    front = model.front(torch.tensor([[t]]), 16, 4)
    far = torch.tensor([[0.05, 0.05, t], [1.9, 0.95, t]])
    net = model._even(model.nets["T"], far, None)

    # On the front the smoothed distance is half the gate's smoothing width, so
    # the gate reads tanh(width / 2 / length) there: small, and set by config.
    floor = float(np.tanh(model.theta_gate_width / 2 / model.theta_gate_length))
    on_front = model.temperature(front.points.detach()).abs()
    assert torch.all(
        on_front
        <= floor * model._even(model.nets["T"], front.points.detach(), None).abs() * 1.01 + 1e-7
    )
    assert torch.allclose(model.temperature(far), net, rtol=1e-4, atol=1e-6)


def test_an_unknown_temperature_gate_is_refused(tmp_path):
    with pytest.raises(ValueError, match="theta_gate"):
        _model(tmp_path, [*SHARP_THERMAL, "model.theta_gate=cubic"])


@pytest.mark.parametrize("gate", ["linear", "saturating"])
def test_the_interface_flux_is_the_liquid_side_gradient(tmp_path, gate):
    """The flux read on the front equals the temperature gradient just past the
    gate's smoothing, not the half-slope in the middle of it."""
    _, data, model = _model(
        tmp_path, [*SHARP_THERMAL, f"model.theta_gate={gate}", "model.theta_gate_length=0.2"]
    )
    t = float(data.t[1])
    front = model.front(torch.tensor([[t]]), 32, 8)
    flux = model.interface_flux_at(front).detach().ravel()
    leaf = front.points.detach().clone()
    leaf[:, :2] = leaf[:, :2] + 5 * model.theta_gate_width * front.normal.detach()
    leaf.requires_grad_(True)
    grad = torch.autograd.grad(model.temperature(leaf).sum(), leaf)[0][:, :2]
    inside = (grad * front.normal.detach()).sum(dim=1)

    ratio = float((inside / flux).median())
    assert ratio == pytest.approx(1.0, abs=0.1), "the flux is the liquid-side gradient"


def test_the_initial_temperature_spares_the_layer_the_bubble_already_has(tmp_path):
    """With T_init_layer_um set, starting points within that distance of the
    first-frame outline carry no initial condition; farther ones still do."""
    cfg, data, model = _model(tmp_path, [*SHARP_THERMAL, "experiment.T_init_layer_um=20"])
    groups = compute_groups(cfg)
    assert groups["init_layer_star"] == pytest.approx(20.0 / 300.0)
    t0 = float(data.t[0])
    front = model.front(torch.tensor([[t0]]), 32, 8)
    near = front.points.detach().clone()
    near[:, :2] = near[:, :2] + 0.5 * groups["init_layer_star"] * front.normal.detach()
    far = torch.tensor([[1.9, 0.05, t0], [1.9, 0.95, t0]])
    walls = torch.tensor([[0.5, 0.0, t0]])

    def initial_part(points):
        batches = ThermalBatches(torch.tensor([[0.0, 0.5, t0]]), points, walls)
        with_init = thermal_boundary_losses(model, groups, batches)
        without = thermal_boundary_losses(
            model, {k: v for k, v in groups.items() if k != "theta_init"}, batches
        )
        return float(with_init - without)

    assert initial_part(near) == pytest.approx(0.0, abs=1e-9), "inside the layer: no condition"
    assert initial_part(far) > 0.0, "beyond it: the 58 C start still holds"


def test_a_negative_initial_layer_is_refused():
    with pytest.raises(ValueError, match="T_init_layer_um"):
        compute_groups(make_config([*THERMAL_SERIES, "experiment.T_init_layer_um=-1"]))
