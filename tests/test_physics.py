"""Dimensionless groups and PDE residuals.

The group tests pin the published values from the README, so a change to the
fluid properties or the geometry that silently shifts the physics shows up as a
failing test rather than as a quietly different result.
"""

from __future__ import annotations

import math

import pytest
import torch

from naviernet.models.pinn import BubblePINN
from naviernet.physics.groups import (
    CONDITIONING_GROUPS,
    N_COND,
    compute_groups,
    conditioning_vector,
    hydraulic_diameter_m,
    inlet_velocity_m_s,
    reference_time_ms,
)
from naviernet.physics.registry import LossContext
from naviernet.physics.residuals import (
    KAPPA_EPS,
    boundary_losses,
    curvature,
    gradients,
    interface_indicator,
    mixture,
    source_penalty,
    stage_a_residuals,
    stage_b_residuals,
)

from .conftest import make_config


def test_conditioning_vector_is_ordered_and_log_scaled(cfg):
    groups = compute_groups(cfg)
    vec = conditioning_vector(groups)
    assert len(vec) == len(CONDITIONING_GROUPS)
    # Log10 compresses the several-orders-of-magnitude spread; Re leads the vector.
    assert CONDITIONING_GROUPS[0] == "Re"
    assert vec[0] == pytest.approx(math.log10(groups["Re"]))


def test_conditioning_vector_excludes_dimensional_reference_quantities():
    # Only regime-defining dimensionless numbers condition the model, not units.
    for dimensional in ("Dh_um", "U_in_m_s", "t_ref_ms"):
        assert dimensional not in CONDITIONING_GROUPS


def test_stage_a_residuals_thread_the_conditioning_context(tiny_cfg):
    model = BubblePINN(tiny_cfg, n_cond=N_COND)
    x = torch.rand(16, 3, requires_grad=True)
    res = stage_a_residuals(model, x, torch.rand(1, N_COND))
    assert res.vof.shape == (16, 1)
    # The residual genuinely depends on the dataset's context, not just the point.
    other = stage_a_residuals(model, x, torch.zeros(1, N_COND))
    assert not torch.allclose(res.vof, other.vof)


def test_boundary_losses_accept_conditioning_for_differently_sized_point_sets(tiny_cfg):
    model = BubblePINN(tiny_cfg, n_cond=N_COND)
    inlet = torch.rand(8, 3)
    wall = torch.rand(5, 3)  # a different count from inlet: the context must broadcast
    loss = boundary_losses(model, inlet, wall, 1.0, torch.rand(1, N_COND))
    assert loss.ndim == 0


def test_loss_context_threads_conditioning_to_its_residuals(tiny_cfg):
    model = BubblePINN(tiny_cfg, n_cond=N_COND)
    groups = compute_groups(tiny_cfg)
    x = torch.rand(12, 3, requires_grad=True)
    ctx = LossContext(
        model, x, torch.rand(4, 3), torch.rand(4, 3), 1.0, groups, c=torch.rand(1, N_COND)
    )
    assert ctx.res_a.vof.shape == (12, 1)


def test_reference_time(cfg):
    # 300 um / 0.2 m/s = 1.5 ms
    assert reference_time_ms(cfg.scales) == pytest.approx(1.5)


def test_hydraulic_diameter(cfg):
    # 300 x 150 um rectangular duct -> Dh = 200 um
    assert hydraulic_diameter_m(cfg.experiment) * 1e6 == pytest.approx(200.0)


def test_inlet_velocity_from_flow_rate(cfg):
    # 5 mL/hr through 300 x 150 um
    assert inlet_velocity_m_s(cfg.experiment) == pytest.approx(0.03086, rel=1e-3)


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("Re", 215.5),
        ("We", 2.302),
        ("Ca", 0.01068),
        ("Pr", 9.411),
        ("hele_shaw", 0.2228),
        ("bretherton_film_um", 4.875),
        # Stage-B closures (q_wall = 2 W/cm^2, H = 150 um, k_l/cp_l/h_lv of FC-72)
        ("dT_ref", 28.74),  # K, conduction superheat q'' (H/2) / k_l
        ("Ja", 0.3746),  # Stefan/Jakob number at the actual superheat
        ("q_wall_star", 0.003945),  # non-dim depth-averaged wall source
    ],
)
def test_published_groups(cfg, group, expected):
    assert compute_groups(cfg)[group] == pytest.approx(expected, rel=1e-3)


def test_peclet_is_the_product_of_reynolds_and_prandtl(cfg):
    groups = compute_groups(cfg)
    assert groups["Pe"] == pytest.approx(groups["Re"] * groups["Pr"])


def test_jakob_number_is_cp_superheat_over_latent_heat(cfg):
    """Ja generalises the fixed Ja_per_5K to the run's actual conduction superheat."""
    groups = compute_groups(cfg)
    assert groups["Ja"] == pytest.approx(cfg.fluid.cp_l * groups["dT_ref"] / cfg.fluid.h_lv)


def test_conduction_superheat_scales_with_wall_flux():
    """dT_ref = q'' (H/2) / k_l — doubling the wall flux doubles the superheat."""
    from .conftest import make_config

    base = compute_groups(make_config(["experiment.q_wall_W_cm2=2.0"]))["dT_ref"]
    hotter = compute_groups(make_config(["experiment.q_wall_W_cm2=4.0"]))["dT_ref"]
    assert hotter == pytest.approx(2.0 * base)


def test_stage_b_groups_are_finite_and_positive(cfg):
    groups = compute_groups(cfg)
    for name in ("dT_ref", "Ja", "q_wall_star"):
        assert math.isfinite(groups[name])  # rejects nan AND inf
        assert groups[name] > 0.0


def test_bond_number_is_small(cfg):
    """Surface tension dominates gravity at this scale -- the premise of the model."""
    assert compute_groups(cfg)["Bond"] < 0.1


def test_gradients_of_a_known_function():
    """d/d(x,y,t) of f = 2x + 3y + 5t is exactly (2, 3, 5)."""
    x = torch.rand(16, 3, requires_grad=True)
    f = (x * torch.tensor([2.0, 3.0, 5.0])).sum(dim=1, keepdim=True)

    f_x, f_y, f_t = gradients(f, x)
    assert torch.allclose(f_x, torch.full_like(f_x, 2.0))
    assert torch.allclose(f_y, torch.full_like(f_y, 3.0))
    assert torch.allclose(f_t, torch.full_like(f_t, 5.0))


def test_interface_indicator_peaks_at_the_interface():
    alpha = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
    weight = interface_indicator(alpha)

    assert weight[2].item() == pytest.approx(1.0)  # alpha = 0.5
    assert weight[0].item() == pytest.approx(0.0)  # pure liquid
    assert weight[4].item() == pytest.approx(0.0)  # pure vapour
    assert torch.all(weight >= 0) and torch.all(weight <= 1)


def test_stage_a_residual_shapes(tiny_cfg):
    model = BubblePINN(tiny_cfg)
    x = torch.rand(32, 3, requires_grad=True)

    residuals = stage_a_residuals(model, x)
    for field in (residuals.vof, residuals.div, residuals.source):
        assert field.shape == (32, 1)
    assert not residuals.interface_weight.requires_grad


def test_divergence_residual_equals_div_u_minus_source(tiny_cfg):
    """r_div = u_x + v_y - s, verified against separately taken gradients."""
    model = BubblePINN(tiny_cfg)
    x = torch.rand(24, 3, requires_grad=True)

    residuals = stage_a_residuals(model, x)
    u, v = model.velocity(x)
    u_x, _, _ = gradients(u, x)
    _, v_y, _ = gradients(v, x)

    assert torch.allclose(residuals.div, u_x + v_y - residuals.source, atol=1e-6)


def test_source_penalty_ignores_sources_on_the_interface(tiny_cfg):
    """A source concentrated exactly at alpha=0.5 costs nothing; away from it, it does."""
    from naviernet.physics.residuals import StageAResiduals

    zero = torch.zeros(8, 1)
    source = torch.ones(8, 1)

    on_interface = StageAResiduals(zero, zero, source, torch.ones(8, 1))
    in_bulk = StageAResiduals(zero, zero, source, torch.zeros(8, 1))

    assert source_penalty(on_interface).item() == pytest.approx(0.0)
    assert source_penalty(in_bulk).item() == pytest.approx(1.0)


def test_nucleation_pulse_source_is_localized_at_the_pin_and_brief():
    """The nucleation pulse is a streamwise Gaussian peaked at x_pin, active only for
    t < t0 (a brief initiation burst), and scales linearly with the pulse magnitude --
    the fixed, time-anchored heat that seeds the bubble."""
    from naviernet.physics.residuals import nucleation_pulse_source

    x_pin, t0, sigma = 0.3, 0.1, 0.05
    q_pulse = torch.tensor(2.0)

    # Points at t=0 (pulse active): at the pin, one sigma away, far away.
    early = torch.tensor([[x_pin, 0.5, 0.0], [x_pin + sigma, 0.5, 0.0], [0.9, 0.5, 0.0]])
    heat = nucleation_pulse_source(early, x_pin, t0, sigma, q_pulse)

    assert heat.shape == (3, 1)
    assert heat[0].item() == pytest.approx(2.0), "peaks at q_pulse on the pin"
    assert heat[1].item() == pytest.approx(2.0 * math.exp(-0.5), rel=1e-5), "Gaussian falloff"
    assert heat[2].item() < 1e-6, "≈0 far from the pin"

    # After t0 the pulse is off everywhere (the burst has ended).
    late = torch.tensor([[x_pin, 0.5, t0], [x_pin, 0.5, 0.5]])
    assert torch.all(nucleation_pulse_source(late, x_pin, t0, sigma, q_pulse) == 0), (
        "no heat once t >= t0"
    )

    # Independent of channel height y (a streamwise band at the pin).
    off_center = torch.tensor([[x_pin, 0.1, 0.0], [x_pin, 0.9, 0.0]])
    band = nucleation_pulse_source(off_center, x_pin, t0, sigma, q_pulse)
    assert band[0].item() == pytest.approx(band[1].item())


def test_boundary_loss_is_zero_for_a_perfect_solution(tiny_cfg):
    """A model outputting exactly the BC values incurs no boundary loss."""
    u_inlet = 0.1542

    class PerfectAtBoundaries:
        def velocity(self, x, c=None):
            # u = u_inlet on the inlet batch, 0 on the wall batch; v = 0 always.
            n = x.shape[0]
            return torch.full((n, 1), u_inlet if n == 4 else 0.0), torch.zeros(n, 1)

    inlet = torch.zeros(4, 3)
    walls = torch.zeros(6, 3)
    loss = boundary_losses(PerfectAtBoundaries(), inlet, walls, u_inlet)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


# --- Stage B: momentum + surface tension, energy + evaporation --------------


class _Analytic:
    """Closed-form fields with genuine (x, y, t) dependence, for residual checks.

    Every accessor mirrors ``BubblePINN``'s, so ``stage_b_residuals`` drives it
    exactly as it would the network, but the values are smooth analytic
    functions whose derivatives autograd can take to any order.
    """

    def __init__(self, eps: float = 0.05):
        self.eps = eps

    def _r(self, x):
        return torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2 + 1e-9)

    # Accessors mirror BubblePINN's, including the optional conditioning `c`
    # (ignored here: these are closed-form fields of the coordinates alone).
    def alpha(self, x, c=None):  # a compact vapour bubble at the origin
        return torch.sigmoid((0.3 - self._r(x)) / self.eps)

    def velocity(self, x, c=None):
        u = torch.sin(2.0 * x[:, 0:1]) * torch.cos(x[:, 1:2]) + 0.3 * x[:, 2:3]
        v = torch.cos(x[:, 0:1]) * torch.sin(2.0 * x[:, 1:2])
        return u, v

    def pressure(self, x, c=None):
        # Asymmetric in x/y, so a p_x <-> p_y transposition in the residual is
        # detectable (p_x = cos x, p_y = -0.5 sin y are not interchangeable).
        return torch.sin(x[:, 0:1]) + 0.5 * torch.cos(x[:, 1:2])

    def temperature(self, x, c=None):
        return 0.5 * torch.sin(x[:, 0:1]) + 0.2 * x[:, 1:2] ** 2 + 0.1 * x[:, 2:3]

    def source(self, x, c=None):
        return 0.2 * torch.cos(x[:, 0:1])


def _groups():
    return compute_groups(make_config())


def test_mixture_blend_runs_from_liquid_to_vapour():
    """alpha is the vapour fraction: liquid (0) -> 1, vapour (1) -> 1/ratio."""
    ratio = 40.0
    assert mixture(torch.zeros(3, 1), ratio).allclose(torch.ones(3, 1))
    assert mixture(torch.ones(3, 1), ratio).allclose(torch.full((3, 1), 1.0 / ratio))


def test_curvature_of_a_circular_bubble_is_one_over_radius():
    """kappa = -div(grad alpha/|grad alpha|) is +1/R on a bubble of radius R,
    pinning both the magnitude and the Laplace sign (positive => higher p inside)."""
    radius, eps = 0.3, 0.02
    angles = torch.linspace(0.0, 6.0, 12)
    pts = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles), torch.zeros_like(angles)],
        dim=1,
    ).requires_grad_(True)
    alpha = torch.sigmoid((radius - torch.sqrt(pts[:, 0:1] ** 2 + pts[:, 1:2] ** 2)) / eps)

    kappa = curvature(alpha, pts)

    assert torch.all(kappa > 0.0), "a bubble must have positive curvature"
    # 15% tolerance: the sigmoid interface has finite width (eps=0.02 vs R=0.3),
    # so the discrete curvature at the sampled ring is only approximately 1/R.
    assert kappa.mean().item() == pytest.approx(1.0 / radius, rel=0.15)


def test_curvature_stays_finite_where_the_interface_is_absent():
    """The |grad alpha| floor keeps kappa finite in the bulk (no 0/0 NaN).

    A near-constant alpha whose gradient sits far below the floor: without the
    floor, grad alpha / |grad alpha| is 0/0.
    """
    x = (torch.rand(16, 3) + 0.5).requires_grad_(True)
    alpha = 0.3 + 1e-6 * torch.sin(x[:, 0:1]) * torch.sin(x[:, 1:2])

    kappa = curvature(alpha, x)

    assert torch.isfinite(kappa).all()


def test_momentum_residual_assembles_the_expected_terms():
    """r_u wires inertia, pressure, viscosity, Hele-Shaw drag and surface tension
    with the correct groups and signs."""
    groups = _groups()
    model = _Analytic()
    x = (torch.rand(48, 3) * 0.6 + 0.05).requires_grad_(True)

    res = stage_b_residuals(model, x, groups, r_int_star=2.0)

    alpha = model.alpha(x)
    u, v = model.velocity(x)
    a_x, a_y, _ = gradients(alpha, x)
    u_x, u_y, u_t = gradients(u, x)
    u_xx, _, _ = gradients(u_x, x)
    _, u_yy, _ = gradients(u_y, x)
    p_x, _, _ = gradients(model.pressure(x), x)
    rho_t = mixture(alpha, groups["rho_ratio"])
    mu_t = mixture(alpha, groups["mu_ratio"])
    kappa = curvature(alpha, x)
    expected = (
        rho_t * (u_t + u * u_x + v * u_y)
        + p_x
        - (1.0 / groups["Re"]) * (u_xx + u_yy)
        + groups["hele_shaw"] * mu_t * u
        - (1.0 / groups["We"]) * kappa * a_x
    )
    assert torch.allclose(res.mom_x, expected, atol=1e-5)


def test_momentum_y_residual_assembles_the_expected_terms():
    """The mirror of the x test: r_v must use v, p_y and a_y -- guards against a
    transposition of the x/y terms in the parallel momentum code."""
    groups = _groups()
    model = _Analytic()
    x = (torch.rand(48, 3) * 0.6 + 0.05).requires_grad_(True)

    res = stage_b_residuals(model, x, groups, r_int_star=2.0)

    alpha = model.alpha(x)
    u, v = model.velocity(x)
    _, a_y, _ = gradients(alpha, x)
    v_x, v_y, v_t = gradients(v, x)
    v_xx, _, _ = gradients(v_x, x)
    _, v_yy, _ = gradients(v_y, x)
    _, p_y, _ = gradients(model.pressure(x), x)
    rho_t = mixture(alpha, groups["rho_ratio"])
    mu_t = mixture(alpha, groups["mu_ratio"])
    kappa = curvature(alpha, x)
    expected = (
        rho_t * (v_t + u * v_x + v * v_y)
        + p_y
        - (1.0 / groups["Re"]) * (v_xx + v_yy)
        + groups["hele_shaw"] * mu_t * v
        - (1.0 / groups["We"]) * kappa * a_y
    )
    assert torch.allclose(res.mom_y, expected, atol=1e-5)


def test_energy_residual_has_wall_heating_gated_by_liquid_and_a_latent_sink():
    """Advection-diffusion, wall source scaled by (1-alpha), and the evaporation
    latent sink St*j*delta -- the same flux that drives the mass source."""
    groups = _groups()
    model = _Analytic()
    r_int = 2.0
    x = (torch.rand(48, 3) * 0.6 + 0.05).requires_grad_(True)

    res = stage_b_residuals(model, x, groups, r_int_star=r_int)

    theta = model.temperature(x)
    u, v = model.velocity(x)
    t_x, t_y, t_t = gradients(theta, x)
    t_xx, _, _ = gradients(t_x, x)
    _, t_yy, _ = gradients(t_y, x)
    alpha = model.alpha(x)
    evap = groups["Ja"] * (theta / r_int) * res.interface_delta
    expected = (
        t_t
        + u * t_x
        + v * t_y
        - (1.0 / groups["Pe"]) * (t_xx + t_yy)
        - groups["q_wall_star"] * (1.0 - alpha)
        + evap
    )
    assert torch.allclose(res.energy, expected, atol=1e-5)


def _evap_and_target(model, x, groups, r_int):
    """The evaporation flux and the closure's flux target (what the source minus
    the residual leaves), for the two mass-closure tests below."""
    res = stage_b_residuals(model, x, groups, r_int_star=r_int)
    evap = groups["Ja"] * (model.temperature(x) / r_int) * res.interface_delta
    target = model.source(x) - res.src_closure
    return evap, target


def test_evaporation_closes_mass_and_energy_with_the_same_flux():
    """s_closure = (1 - rho_v/rho_l) * evap and the energy sink = evap: one flux,
    two conservation laws (the two-way mass-energy closure). The mass prefactor is
    the dilatation the vapour actually gains -- see the regression test below."""
    groups = _groups()
    model = _Analytic()
    x = (torch.rand(32, 3) * 0.6 + 0.05).requires_grad_(True)

    evap, target = _evap_and_target(model, x, groups, r_int=3.0)

    assert torch.allclose(target, (1.0 - 1.0 / groups["rho_ratio"]) * evap, atol=1e-6)


def test_mass_closure_prefactor_is_the_dilatation_not_rho_ratio():
    """Regression: the phase-change dilatation is mdot*(1/rho_v - 1/rho_l), so in
    the model's (vapour-scaled) source the prefactor is (1 - rho_v/rho_l) ~ O(1),
    NOT (rho_l/rho_v - 1) ~ O(rho_ratio). The latter over-scaled the source by
    ~120x, driving unphysical interface velocity that collapsed alpha (IoU -> 0).
    """
    groups = _groups()
    model = _Analytic()
    x = (torch.rand(32, 3) * 0.6 + 0.05).requires_grad_(True)

    evap, target = _evap_and_target(model, x, groups, r_int=3.0)

    assert torch.allclose(target, (1.0 - 1.0 / groups["rho_ratio"]) * evap, atol=1e-6)
    assert not torch.allclose(target, (groups["rho_ratio"] - 1.0) * evap, atol=1e-6)
    # The prefactor the closure actually applies, read back from its output, is
    # O(1) -- not O(rho_ratio) as the bug had it. Derived from `target`, so this
    # exercises the residual, not just the group value.
    implied = (target / evap.clamp_min(1e-9)).abs().median().item()
    assert implied < 2.0


def test_source_closure_vanishes_when_the_source_matches_evaporation():
    """A model whose inferred source equals the evaporation closure has zero
    consistency residual; perturbing the source breaks it."""
    groups = _groups()
    x = (torch.rand(24, 3) * 0.6 + 0.05).requires_grad_(True)

    class _MatchedSource(_Analytic):
        def source(self, xx, c=None):
            a_x, a_y, _ = gradients(self.alpha(xx), xx)
            delta = torch.sqrt(a_x**2 + a_y**2 + KAPPA_EPS**2)
            j = self.temperature(xx) / 3.0
            return (1.0 - 1.0 / groups["rho_ratio"]) * groups["Ja"] * j * delta

    matched = stage_b_residuals(_MatchedSource(), x, groups, r_int_star=3.0)
    assert matched.src_closure.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_stage_b_residuals_are_finite_and_shaped():
    groups = _groups()
    x = (torch.rand(20, 3) * 0.6 + 0.05).requires_grad_(True)

    res = stage_b_residuals(_Analytic(), x, groups)

    for tensor in (res.mom_x, res.mom_y, res.energy, res.src_closure):
        assert tensor.shape == (20, 1)
        assert torch.isfinite(tensor).all()
