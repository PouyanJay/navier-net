"""The liquid film delta(x,t) between the bubble and the wall.

In a 150 um gap the bubble spans the channel; the liquid it displaces is left
behind as a thin film against the wall, and our depth-averaged model integrates
that film away. The film field rides on the SAME explicit front the
sharp-interface conditions already sample: one small net over (u, t), no new
sampler.

The deposition law is Aussillous & Quere's saturating form fitted to pancake
bubbles (Shukla et al. 2019), evaluated on the LOCAL normal speed. Deliberately
NOT the boundary-layer ``delta0 = C sqrt(nu t)``: the four working fluids have
nearly identical kinematic viscosities (nu = 2.6-3.2e-7 m^2/s), so that
correlation is fluid-BLIND -- it would build in exactly the blindness the film
exists to remove. The Landau-Levich scaling ``(mu/sigma)^{2/3}`` separates the
dielectrics from water by ~4x, and the fluid-dependence test below is the
non-negotiable this suite exists to pin.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import make_config
from tests.conftest import staged_run as _staged_run

SHARP = ["model=stage_b", "model.front_geometry=true", "model.sharp_interface=true"]
FILM = [*SHARP, "model.liquid_film=true"]


def _model(tmp_path, overrides):
    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths

    cfg, paths = _staged_run(tmp_path, overrides)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    return BubblePINN(cfg, geometry=_geometry_priors(cfg, data)), data, cfg


def _groups_for_fluid(fluid: str) -> dict[str, float]:
    from naviernet.physics.groups import compute_groups

    return compute_groups(make_config([f"fluid={fluid}"]))


def test_liquid_film_is_off_by_default(tmp_path):
    """Opt-in: without the flag there is no film net, no film parameters, and
    the active equation set is exactly what R4 shipped."""
    from naviernet.physics import registry

    model, _, cfg = _model(tmp_path, SHARP)
    assert cfg.model.liquid_film is False
    assert not hasattr(model, "film")

    ids = [e.id for e in registry.enabled_equations(cfg.model.fields, sharp_interface=True)]
    assert "film" not in ids


def test_liquid_film_requires_the_sharp_interface(tmp_path):
    """The film's loss term is scored on the explicit front the trainer only
    samples under sharp mode, and its later stages feed the jump condition."""
    from naviernet.models.pinn import BubblePINN

    cfg, _ = _staged_run(tmp_path, ["model=stage_b", "model.liquid_film=true"])
    with pytest.raises(ValueError, match="sharp_interface"):
        BubblePINN(cfg)


def test_registry_adds_the_film_equation_only_under_the_flag(tmp_path):
    """Declarative, like every other equation: the trainer, the API and the UI
    read one table."""
    from naviernet.physics import registry

    cfg, _ = _staged_run(tmp_path, FILM)
    without = [e.id for e in registry.enabled_equations(cfg.model.fields, sharp_interface=True)]
    with_film = [
        e.id
        for e in registry.enabled_equations(
            cfg.model.fields, sharp_interface=True, liquid_film=True
        )
    ]
    assert "film" not in without
    assert "film" in with_film


def test_flag_off_leaves_every_other_net_bit_identical(tmp_path):
    """The film net must be constructed LAST, so switching it on cannot shift
    the RNG draws that initialize every other field -- flag off is byte-identical
    not just in code path but in the weights a seeded run starts from."""
    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths

    cfg_off, paths = _staged_run(tmp_path, SHARP)
    cfg_on, _ = _staged_run(tmp_path, FILM)
    data = BubbleDataset(cfg_off, RunPaths.from_config(cfg_off), device="cpu")
    priors = _geometry_priors(cfg_off, data)

    torch.manual_seed(7)
    plain = BubblePINN(cfg_off, geometry=priors)
    torch.manual_seed(7)
    filmed = BubblePINN(cfg_on, geometry=priors)

    plain_state = plain.state_dict()
    filmed_state = filmed.state_dict()
    film_keys = {k for k in filmed_state if "film" in k}
    assert any(k.startswith("film.") for k in film_keys), (
        "the film net must live under the 'film.' namespace"
    )
    assert "_log_film_resistance" in film_keys, (
        "the trained kinetic-resistance unknown must exist under the flag"
    )
    assert set(filmed_state) - film_keys == set(plain_state)
    for key, value in plain_state.items():
        assert torch.equal(value, filmed_state[key]), (
            f"{key} shifted when the film was enabled -- the film net must not "
            f"consume RNG before the shared fields"
        )


def test_deposition_separates_the_fluids_where_sqrt_nu_t_cannot(tmp_path):
    """THE design constraint. At the same normal speed, the film thickness
    scales as (mu/sigma)^{2/3}: the dielectrics and water separate by ~4x. The
    rejected sqrt(nu t) correlation returns nearly the same value for all four
    fluids, which is exactly the blindness this feature removes."""
    from naviernet.physics.film import deposited_thickness

    speed = torch.ones(1, 1)
    fc72 = deposited_thickness(speed, _groups_for_fluid("fc72"))
    water = deposited_thickness(speed, _groups_for_fluid("water"))

    ratio = float(fc72 / water)
    assert 3.0 < ratio < 6.0, (
        f"FC-72 must deposit a ~4x thicker film than water at matched speed "
        f"(Landau-Levich (mu/sigma)^{{2/3}}); got ratio {ratio:.2f}"
    )


def test_deposition_law_is_monotone_saturating_and_advancing_only(tmp_path):
    """The Aussillous-Quere form: zero where the front recedes (a receding
    meniscus deposits nothing), monotone in the advance speed, and saturating
    strictly below the half-gap."""
    from naviernet.physics.film import AQ_Q, deposited_thickness

    _, _, cfg = _model(tmp_path, FILM)
    from naviernet.physics.groups import compute_groups

    groups = compute_groups(cfg)
    speeds = torch.linspace(-1.0, 5.0, 200).reshape(-1, 1)
    delta = deposited_thickness(speeds, groups)

    receding = speeds <= 0.0
    assert torch.all(delta[receding] == 0.0)
    advancing = delta[~receding]
    assert torch.all(advancing[1:] >= advancing[:-1]), "must be monotone in speed"
    half_gap = 0.5 * groups["H_star"]
    assert torch.all(delta < half_gap / AQ_Q + 1e-9), "must saturate below (h/2)/Q"


def _speed_profile_front(n: int = 256, seed: int = 3):
    """Front samples whose normal speed is a deterministic function of their
    axial position: receding near the root, advancing up to 1.7x the reference
    speed toward the nose -- the shape of a depositing meniscus sweep.

    Constructed rather than sampled from the synthetic bubble, whose speeds are
    ~1e-3 -- essentially the deposition law's zero floor, so a fit against them
    tests nothing but "can the net learn zero". The physical speed range is
    what exercises the law's knee and saturation.
    """
    from naviernet.models.geometry import FrontSamples

    rng = torch.Generator().manual_seed(seed)
    u = torch.rand(n, 1, generator=rng)
    t = torch.rand(n, 1, generator=rng)
    x = 0.75 + 4.0 * u  # axial position, the film net's own coordinate
    speed = 2.0 * u - 0.3
    return FrontSamples(
        points=torch.cat([x, torch.zeros(n, 1), t], dim=1),
        u=u,
        side=torch.ones(n, 1),
        on_cap=torch.zeros(n, 1),
        angle=torch.zeros(n, 1),
        kappa_par=torch.zeros(n, 1),
        normal=torch.zeros(n, 2),
        normal_speed=speed,
    )


def test_the_film_fits_the_deposition_law_on_the_front(tmp_path):
    """The T0 gate: trained against the deposition residual alone, the film net
    reproduces the Aussillous-Quere prediction across the physical speed range
    within its own fit spread (~10%)."""
    from naviernet.physics.film import deposited_thickness, deposition_residual
    from naviernet.physics.groups import compute_groups

    model, _, cfg = _model(tmp_path, FILM)
    groups = compute_groups(cfg)
    front = _speed_profile_front()

    opt = torch.optim.Adam(model.film.parameters(), lr=1e-2)
    for _ in range(1000):
        opt.zero_grad()
        loss = (deposition_residual(model, front, groups) ** 2).mean()
        loss.backward()
        opt.step()

    advancing = front.normal_speed.squeeze(1) > 0
    with torch.no_grad():
        target = deposited_thickness(front.normal_speed, groups)[advancing]
        fitted = model.film_thickness(front)[advancing]
    target_rms = float((target**2).mean().sqrt())
    rms = float(((fitted - target) ** 2).mean().sqrt())
    assert rms < 0.10 * target_rms, (
        f"film net must match the deposition law within ~10% of its RMS "
        f"({target_rms:.4g}); RMS error {rms:.4g}"
    )


def test_deposition_cannot_pull_the_front(tmp_path):
    """T0 is deposition ONLY -- the film learns from the front, never the
    reverse. The target is detached, so the film term must leave every geometry
    parameter without gradient."""
    from naviernet.physics.film import deposition_residual
    from naviernet.physics.groups import compute_groups

    model, data, cfg = _model(tmp_path, FILM)
    groups = compute_groups(cfg)
    front = model.nets["phi"].front(torch.tensor([[0.5]]), n_body=16, n_cap=4)

    loss = (deposition_residual(model, front, groups) ** 2).mean()
    loss.backward()

    geometry = model.nets["phi"]
    moved = [
        name
        for name, p in geometry.named_parameters()
        if p.grad is not None and float(p.grad.abs().max()) > 0.0
    ]
    assert not moved, f"the film term reached the geometry through {moved}"
    assert any(
        p.grad is not None and float(p.grad.abs().max()) > 0.0 for p in model.film.parameters()
    ), "the film net itself must receive gradient"


def test_deposition_anchors_advancing_samples_and_leaves_receding_ones_alone(tmp_path):
    """Only an advancing meniscus deposits -- and on the real bubble the
    advancing region is the NOSE CAP (the body measurably recedes as the
    capsule elongates), so caps are in and receding stations contribute exactly
    nothing. A receding station's film is yesterday's deposit minus what has
    evaporated: depletion's business (T1), not a fresh-deposit condition."""
    from naviernet.physics.film import deposition_residual
    from naviernet.physics.groups import compute_groups

    model, _, cfg = _model(tmp_path, FILM)
    groups = compute_groups(cfg)
    front = _speed_profile_front()

    residual = deposition_residual(model, front, groups).squeeze(1)
    advancing = front.normal_speed.squeeze(1) > 0
    assert torch.all(residual[~advancing] == 0.0)
    assert torch.any(residual[advancing] != 0.0), (
        "an untrained film should not already satisfy the deposition law"
    )

    # And on the real front construction, the advancing nose cap is anchored.
    # The precondition is asserted, not used as a guard: a geometry change that
    # left no cap sample advancing would otherwise turn this half of the test
    # into a silent no-op.
    real = model.nets["phi"].front(torch.tensor([[0.3], [0.7]]), n_body=12, n_cap=6)
    cap_advancing = (real.on_cap.squeeze(1) == 1) & (real.normal_speed.squeeze(1) > 0)
    assert cap_advancing.any(), (
        "expected at least one advancing nose-cap sample on the growing fixture"
    )
    real_residual = deposition_residual(model, real, groups).squeeze(1)
    assert torch.any(real_residual[cap_advancing] != 0.0), (
        "the advancing nose cap is the depositing meniscus and must be anchored"
    )


def test_the_film_trains_end_to_end_and_travels_in_the_checkpoint(tmp_path):
    """The whole path in one tiny real run: the flag composes, the registry adds
    the film equation, the trainer scores it, and the checkpoint records both
    the parameters and the architecture."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, FILM)
    model, _, state = train(cfg, paths)

    record = state["hist"][-1]
    assert "film" in record, f"the film term must be trained and logged, got {record}"
    assert record["film"] == pytest.approx(record["film"]), "film loss must not be NaN"

    ckpt = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    film_keys = [k for k in ckpt["model"] if k.startswith("film.")]
    assert film_keys, "the film net must travel in the checkpoint"
    assert ckpt["liquid_film"] is True

    reloaded, _, _ = load_model(cfg, paths)
    for key in film_keys:
        assert torch.equal(reloaded.state_dict()[key], ckpt["model"][key])


def test_checkpoint_refuses_a_liquid_film_mismatch(tmp_path):
    """The film adds parameters and a loss term; consuming its checkpoint
    without the flag would silently drop both."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, FILM)
    train(cfg, paths)
    plain, _ = _staged_run(tmp_path, SHARP)
    with pytest.raises(ValueError, match="liquid_film"):
        load_model(plain, paths)


# --- T1: depletion, the kinetic resistance, and dryout --------------------------


def test_molar_mass_composes_for_every_fluid():
    """The kinetic resistance needs the vapour's molar mass; every characterised
    fluid must carry it (SI, kg/mol)."""
    for fluid, m_lo, m_hi in [
        ("fc72", 0.3, 0.4),
        ("fc77", 0.3, 0.5),
        ("hfe7100", 0.2, 0.3),
        ("novec649", 0.3, 0.4),
        ("water", 0.017, 0.019),
    ]:
        cfg = make_config([f"fluid={fluid}"])
        assert m_lo < cfg.fluid.molar_mass < m_hi, fluid


def test_the_kinetic_resistance_is_fluid_dependent_and_in_the_predicted_band():
    """R_gamma as an equivalent liquid length: the Schrage resistance with the
    boiling-calibrated accommodation coefficient is worth ~0.5-1.5 um of liquid
    for the dielectrics (the plan's 0.54-1.24 um) and more for water -- and it
    is genuinely fluid-dependent, unlike the kinematic viscosity."""
    lengths_um = {}
    for fluid in ("fc72", "hfe7100", "water"):
        groups = _groups_for_fluid(fluid)
        lengths_um[fluid] = groups["R_gamma_star"] * 300.0  # L_ref = 300 um

    assert 0.5 < lengths_um["fc72"] < 1.5
    assert 0.5 < lengths_um["hfe7100"] < 1.5
    assert 1.5 < lengths_um["water"] < 3.5
    spread = max(lengths_um.values()) / min(lengths_um.values())
    assert spread > 1.5, f"R_gamma must separate the fluids; spread {spread:.2f}"


def test_film_flux_is_bounded_as_the_film_vanishes():
    """THE T1 gate. The current |grad alpha| source has q = k dT/delta, which
    DIVERGES as delta -> 0; the film flux has the kinetic resistance in series,
    so it is bounded by E_film * theta / R_gamma* whatever the film does."""
    from naviernet.physics.film import film_flux

    groups = _groups_for_fluid("fc72")
    theta = torch.tensor([[1.0]])
    delta = torch.linspace(0.0, 0.5 * groups["H_star"], 500).reshape(-1, 1)

    flux = film_flux(delta, theta, groups)
    bound = groups["film_depletion"] / groups["R_gamma_star"]
    assert torch.all(flux >= 0.0)
    assert torch.all(flux <= bound + 1e-12), "flux must be bounded by E theta / R_gamma"


def test_a_dry_station_contributes_no_evaporation():
    """Dryout: below the roughness scale there is no liquid left to evaporate,
    so the flux gates smoothly to zero instead of evaporating a film that is
    not there."""
    from naviernet.physics.film import film_flux

    groups = _groups_for_fluid("fc72")
    theta = torch.tensor([[1.0]])
    dry = torch.tensor([[0.0]])
    wet = torch.tensor([[10.0 * groups["film_dryout_star"]]])

    dry_flux = float(film_flux(dry, theta, groups))
    wet_flux = float(film_flux(wet, theta, groups))
    assert dry_flux < 0.05 * wet_flux, (
        f"a dry station must contribute ~nothing (dry {dry_flux:.3g} vs wet {wet_flux:.3g})"
    )


def test_a_deposited_film_drains_to_dryout_within_the_event():
    """Dryout is REACHABLE: integrating the depletion law from a fresh deposit
    under the reference superheat reaches the roughness scale in finite time,
    and that time matches the closed-form drain solution
    (delta + R)^2 = (delta0 + R)^2 - 2 E theta t within a few percent."""
    from naviernet.physics.film import deposited_thickness, film_flux

    groups = _groups_for_fluid("fc72")
    theta = torch.tensor([[0.5]])
    delta0 = float(deposited_thickness(torch.ones(1, 1), groups))
    dry = groups["film_dryout_star"]
    r_gamma = groups["R_gamma_star"]
    e_film = groups["film_depletion"]

    # Closed form, ignoring the dry gate (it only acts within a hair of dryout).
    t_dry = ((delta0 + r_gamma) ** 2 - (dry + r_gamma) ** 2) / (2.0 * e_film * 0.5)

    delta, t, dt = torch.tensor([[delta0]]), 0.0, 1e-3
    while float(delta) > dry and t < 100.0:
        delta = delta - dt * film_flux(delta, theta, groups)
        t += dt
    assert t < 100.0, "the film never drained"
    # Measured Euler error at dt=1e-3 is ~0.3%; 2% leaves headroom for a future
    # dt change without ever letting a sign or prefactor slip through.
    assert t == pytest.approx(t_dry, rel=0.02)
    # And the event window is ~3.3 t*, so a fresh deposit CAN dry out in-event.
    assert t_dry < 3.3


def test_liquid_film_requires_the_temperature_field(tmp_path):
    """Depletion is driven by the local superheat; a model without T has no
    superheat to read."""
    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths

    cfg, _ = _staged_run(tmp_path, [*FILM, "model.fields=[phi,u,v,s,p]"])
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    with pytest.raises(ValueError, match="'T'"):
        BubblePINN(cfg, geometry=_geometry_priors(cfg, data))


def test_depletion_residual_wiring_matches_finite_differences(tmp_path):
    """The residual is d(delta)/dt at FIXED x plus the gated flux; check the
    autograd time-derivative against a central difference of the film net."""
    from naviernet.physics.film import depletion_residual, film_flux
    from naviernet.physics.groups import compute_groups

    model, data, cfg = _model(tmp_path, FILM)
    groups = compute_groups(cfg)
    times = torch.tensor([[0.4], [0.9]])

    residual = depletion_residual(model, times, groups, stations=8)
    assert residual.shape == (16, 1)

    # Rebuild the same quadrature by hand and compare one sample.
    x0, y0 = model.film_root()
    s = model.apex(times)[:, 0:1].detach()
    frac = (torch.arange(8, dtype=torch.float32) + 0.5) / 8
    x = (x0 + (s - x0) * frac.reshape(1, -1)).reshape(-1, 1)
    t = times.repeat_interleave(8, dim=0)
    eps = 1e-4
    with torch.no_grad():
        d_plus = model.film(x, t + eps)
        d_minus = model.film(x, t - eps)
        ddt = (d_plus - d_minus) / (2 * eps)
        theta = model.temperature(torch.cat([x, torch.full_like(x, y0), t], dim=1))
        expected = ddt + film_flux(model.film(x, t), theta, groups)
    assert torch.allclose(residual, expected, atol=1e-4)


def test_depletion_trains_only_the_film(tmp_path):
    """T1 is still UNCOUPLED: the film reads the temperature and the front,
    never writes them. The residual's graph must touch the film's own unknowns
    alone -- the net, and the trained kinetic resistance the flux carries."""
    from naviernet.physics.film import depletion_residual
    from naviernet.physics.groups import compute_groups

    model, data, cfg = _model(tmp_path, FILM)
    groups = compute_groups(cfg)

    loss = (depletion_residual(model, torch.tensor([[0.5]]), groups, stations=8) ** 2).mean()
    loss.backward()

    outsiders = [
        name
        for name, p in model.named_parameters()
        if "film" not in name and p.grad is not None and float(p.grad.abs().max()) > 0
    ]
    assert not outsiders, f"depletion reached beyond the film's unknowns: {outsiders}"
    assert any(
        p.grad is not None and float(p.grad.abs().max()) > 0 for p in model.film.parameters()
    ), "the film net itself must receive gradient"


def test_registry_adds_the_depletion_equation_under_the_flag(tmp_path):
    from naviernet.physics import registry

    cfg, _ = _staged_run(tmp_path, FILM)
    ids = [
        e.id
        for e in registry.enabled_equations(
            cfg.model.fields, sharp_interface=True, liquid_film=True
        )
    ]
    assert "film_depletion" in ids
    without = [e.id for e in registry.enabled_equations(cfg.model.fields, sharp_interface=True)]
    assert "film_depletion" not in without


def test_depletion_trains_end_to_end(tmp_path):
    """The term is scored and logged in a real tiny run."""
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, FILM)
    _, _, state = train(cfg, paths)
    record = state["hist"][-1]
    assert "film_depletion" in record
    assert record["film_depletion"] == pytest.approx(record["film_depletion"])


# --- T2: the film's evaporation becomes the mass source -------------------------


def test_the_film_source_replaces_the_grad_alpha_closure(tmp_path):
    """Under the flag the mass source is closed against the film's evaporation,
    not the |grad alpha| distribution: one closure per field, or the two would
    fight over `s` with different distributions."""
    from naviernet.physics import registry

    cfg, _ = _staged_run(tmp_path, FILM)
    with_film = [
        e.id
        for e in registry.enabled_equations(
            cfg.model.fields, sharp_interface=True, liquid_film=True
        )
    ]
    assert "film_source" in with_film
    assert "evap" not in with_film, "the old closure must be REPLACED, not doubled"

    without = [e.id for e in registry.enabled_equations(cfg.model.fields, sharp_interface=True)]
    assert "evap" in without
    assert "film_source" not in without


def test_film_source_prefactor_conserves_mass():
    """Algebraic gate on the depth-averaged prefactor: over a footprint of area
    A, the volume created per time must be (both walls) x (rho_l/rho_v - 1) x
    (liquid consumed per area) x A. Integrating s over the gap multiplies by
    H*, so s must carry 2 (rho_ratio - 1) / H* -- a factor-of-two or an H*
    slip here IS a mass-conservation bug."""
    from naviernet.physics.film import film_flux, film_source_density

    groups = _groups_for_fluid("fc72")
    delta = torch.full((5, 1), 0.01)
    theta = torch.full((5, 1), 0.5)

    flux = film_flux(delta, theta, groups)
    density = film_source_density(delta, theta, groups)
    expected = 2.0 * (groups["rho_ratio"] - 1.0) / groups["H_star"] * flux
    assert torch.allclose(density, expected)


def test_the_film_distribution_differs_from_the_grad_alpha_one(tmp_path):
    """The T2 gate: at EQUAL totals, the two closures put the mass in different
    places. The discriminator is the bubble's INTERIOR (alpha ~ 1, grad ~ 0):
    the |grad alpha| closure is structurally ~zero there, while the film
    evaporates across the whole footprint the bubble covers -- which is exactly
    the ">70% of heat transfer is the film" physics this task encodes.

    alpha_eps is sharpened so the tiny fixture bubble HAS an interior; at the
    default 0.05 the blur is comparable to the whole bubble and every vapour
    point is also 'interface'."""
    from naviernet.physics.film import film_source_target
    from naviernet.physics.groups import compute_groups
    from naviernet.physics.residuals import gradients

    model, data, cfg = _model(tmp_path, [*FILM, "model.alpha_eps=0.005"])
    groups = compute_groups(cfg)

    n = 4000
    rng = torch.Generator().manual_seed(11)
    d = data.domain
    x = torch.rand(n, 1, generator=rng) * (d.x_max - d.x_min) + d.x_min
    y = torch.rand(n, 1, generator=rng) * (d.y_max - d.y_min) + d.y_min
    t = torch.rand(n, 1, generator=rng) * (d.t_max - d.t_min) + d.t_min
    pts = torch.cat([x, y, t], dim=1).requires_grad_(True)

    alpha = model.alpha(pts)
    a_x, a_y, _ = gradients(alpha, pts)
    grad_mag = torch.sqrt(a_x**2 + a_y**2).detach().squeeze(1)
    theta = model.temperature(pts).detach().squeeze(1)
    vapour = alpha.detach().squeeze(1)
    old = theta * grad_mag  # the old closure's spatial shape

    new = film_source_target(model, pts, groups).squeeze(1)
    assert float(new.sum().detach()) > 0.0, "the film source must put mass somewhere"

    # Normalize to equal totals, then compare the interior's share of the mass.
    old = old / old.sum().clamp(min=1e-12)
    new = new / new.sum().clamp(min=1e-12)
    interior = ((vapour > 0.95) & (grad_mag < 0.05 * grad_mag.max())).float()
    assert interior.sum() > 20, "fixture must expose a genuine interior"
    old_interior = float((old * interior).sum())
    new_interior = float((new * interior).sum())
    assert old_interior < 0.05, "the |grad alpha| closure is ~zero in the interior"
    assert new_interior > 5.0 * max(old_interior, 1e-3), (
        f"the film must put real mass over the interior footprint "
        f"(old {old_interior:.4f} vs new {new_interior:.4f})"
    )


def test_film_source_target_moves_only_the_resistance_and_is_gated_by_vapour(tmp_path):
    """One-way in the FIELDS, the same design as the shipped evap closure: the
    target cannot move the film net, theta, or the interface. Its one live
    parameter is the trained kinetic resistance -- this closure, where the
    film's implied growth meets the observed growth, is where the
    accommodation unknown is identifiable. And outside the bubble there is no
    film surface, so the target vanishes with alpha."""
    from naviernet.physics.film import film_source_target
    from naviernet.physics.groups import compute_groups

    model, data, cfg = _model(tmp_path, FILM)
    groups = compute_groups(cfg)

    pts = torch.tensor([[1.0, 0.5, 0.5], [10.0, 0.95, 0.5]])  # inside; far outside
    target = film_source_target(model, pts, groups)
    target.sum().backward()
    moved = [
        name
        for name, p in model.named_parameters()
        if p.grad is not None and float(p.grad.abs().max()) > 0
    ]
    assert moved == ["_log_film_resistance"], (
        f"only the resistance unknown may be live in the target, got {moved}"
    )
    with torch.no_grad():
        assert float(target[1]) < 0.05 * float(target[0].clamp(min=1e-12)), (
            "outside the bubble the target must vanish with alpha"
        )


def test_src_penalty_moves_to_the_liquid_under_the_film(tmp_path):
    """The shipped src penalty suppresses `s` outside the INTERFACE BAND --
    which is exactly where the film's source lives (the interior footprint).
    Under the flag the penalty's forbidden region becomes the liquid outside
    the bubble, or the two terms would fight over every interior point."""
    from naviernet.physics import registry
    from naviernet.physics.groups import compute_groups

    model, data, cfg = _model(tmp_path, [*FILM, "model.alpha_eps=0.005"])
    groups = compute_groups(cfg)

    # Probe points from the model's own capsule: mid-spine at the root height
    # is interior; far downstream of the nose is liquid.
    x0, y0 = model.film_root()
    nose = float(model.apex(torch.tensor([[0.5]]))[0, 0])
    inside = torch.tensor([[0.5 * (x0 + nose), y0, 0.5]])
    outside = torch.tensor([[nose + 2.0, y0, 0.5]])
    pts = torch.cat([inside, outside]).requires_grad_(True)
    with torch.no_grad():
        alpha = model.alpha(pts)
    assert float(alpha[0]) > 0.95 and float(alpha[1]) < 0.05, "probe points must bracket"

    ctx = registry.LossContext(model, pts, groups=groups)
    penalty = registry._src_sq(ctx).squeeze(1)
    with torch.no_grad():
        source_sq = (model.source(pts) ** 2).squeeze(1)
    weight = penalty / source_sq.clamp(min=1e-18)
    assert float(weight[0]) < 0.01, "the interior must be free for the film's mass"
    assert float(weight[1]) > 0.81, "the outside liquid stays penalised"


def test_film_source_trains_end_to_end_and_replaces_evap_in_the_log(tmp_path):
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, FILM)
    _, _, state = train(cfg, paths)
    record = state["hist"][-1]
    assert "film_source" in record
    assert "evap" not in record, "the old closure must not be scored under the flag"
    assert record["film_source"] == pytest.approx(record["film_source"])


# --- T3: the film surface's capillary pressure enters the jump condition --------


class _AnalyticFilm(torch.nn.Module):
    """A film with a Gaussian thinning patch at a known station -- the
    controlled input for testing the pressure's station selection. The system
    under test is film_surface_pressure; only the film field is synthesized,
    exactly as _speed_profile_front synthesizes the front."""

    def __init__(self, centre: float, width: float = 0.25):
        super().__init__()
        self.centre, self.width = centre, width

    def forward(self, x, t, c=None):
        dip = 0.006 * torch.exp(-((x - self.centre) ** 2) / (2.0 * self.width**2))
        return 0.012 - dip


def test_film_pressure_peaks_where_the_film_thins(tmp_path):
    """THE T3 mechanism, stated as a measurement: at a local thinning patch
    (incipient dryout) the film's surface curvature raises its pressure toward
    the vapour's, and the perturbation the body meniscus faces peaks AT that
    station -- the jump condition can now select a waist where the film
    thins, which is Richards & Pegler's pinch-off route."""
    from naviernet.physics.film import film_surface_pressure
    from naviernet.physics.groups import compute_groups

    model, data, cfg = _model(tmp_path, FILM)
    groups = compute_groups(cfg)
    times = torch.tensor([[0.5]])
    front = model.nets["phi"].front(times, n_body=48, n_cap=6)

    x0, _ = model.film_root()
    nose = float(model.apex(times)[0, 0].detach())
    centre = x0 + 0.45 * (nose - x0)
    model.film = _AnalyticFilm(centre)

    p = film_surface_pressure(model, front, groups).squeeze(1)
    body = front.on_cap.squeeze(1) == 0
    xb = front.points[body, 0].detach()
    peak_x = float(xb[p[body].argmax()])
    assert abs(peak_x - centre) < 0.15 * (nose - x0), (
        f"the pressure must peak at the thinning patch ({centre:.2f}), got {peak_x:.2f}"
    )
    assert float(p[body].max()) > 0.0, "a thinning patch must RAISE the film pressure"


def test_film_pressure_enters_the_jump_on_the_body_only(tmp_path):
    """The body meniscus faces the film; the caps face bulk liquid. The
    forcing must be detached (a computed forcing, like the solved shape
    coefficients), zero on the caps, and alive on the body."""
    from naviernet.physics.film import film_surface_pressure
    from naviernet.physics.groups import compute_groups
    from naviernet.physics.residuals import laplace_jump_residual

    model, data, cfg = _model(tmp_path, FILM)
    groups = compute_groups(cfg)
    front = model.nets["phi"].front(torch.tensor([[0.4], [0.9]]), n_body=16, n_cap=6)

    residual = laplace_jump_residual(model, front, groups)
    assert residual.requires_grad, "the jump must still train"

    p = film_surface_pressure(model, front, groups)
    on_cap = front.on_cap.squeeze(1) == 1
    assert not p.requires_grad, "the film pressure is a detached forcing"
    assert torch.all(p[on_cap] == 0.0), "caps face bulk liquid, not the film"
    assert torch.any(p[~on_cap] != 0.0), (
        "a trained film's curvature must perturb the body's pressure"
    )


def test_film_pressure_is_mean_free_and_bounded_by_the_relief_scale(tmp_path):
    """Two structural properties: the constant part of the film-to-bulk
    discrepancy belongs to the scalar film_offset (so this forcing is
    mean-free per time and cannot shift the whole jump), and pressures beyond
    the film's own capillary scale deform the film instead of transmitting
    (the soft cap)."""
    from naviernet.physics.film import film_surface_pressure
    from naviernet.physics.groups import compute_groups

    model, data, cfg = _model(tmp_path, FILM)
    groups = compute_groups(cfg)
    times = torch.tensor([[0.3], [0.8]])
    front = model.nets["phi"].front(times, n_body=32, n_cap=8)

    p = film_surface_pressure(model, front, groups).squeeze(1)
    assert torch.all(torch.isfinite(p))
    relief = (2.0 / groups["H_star"]) / groups["We"]
    assert torch.all(p.abs() <= relief + 1e-6)

    body = front.on_cap.squeeze(1) == 0
    t = front.points[:, 2].detach()
    for tv in torch.unique(t):
        row = body & (t == tv)
        assert float(p[row].mean().abs()) < 0.05 * relief, (
            "the per-time mean belongs to film_offset, not this forcing"
        )


# --- T4: the film's diagnostics in metrics.json ---------------------------------


def test_film_report_measures_profile_dryness_partition_and_peak(tmp_path):
    """The T4 gate: the three quantities the film exists to provide are
    MEASURABLE -- the thickness profile (the fluid-dependent prediction), the
    dry fraction (the pinch precursor), and the evaporation partition (the
    literature's >70% through the film) -- plus the T3 mechanism's own peak
    station, per frame."""
    from naviernet.physics.diagnostics import film_report
    from naviernet.physics.groups import compute_groups

    model, data, cfg = _model(tmp_path, FILM)
    groups = compute_groups(cfg)
    report = film_report(model, data, groups)

    assert len(report["per_frame"]) == len(data.frame_numbers)
    for frame in report["per_frame"]:
        assert all(v >= 0.0 for v in frame["delta_um"]), "a film cannot be negative"
        assert 0.0 <= frame["dry_fraction"] <= 1.0
        assert 0.0 <= frame["film_evaporation_fraction"] <= 1.0
        assert 0.0 <= frame["film_pressure_peak_fraction"] <= 1.0
    assert report["r_gamma_um"] == pytest.approx(1.14, abs=0.15), (
        "FC-72's kinetic resistance must sit in the plan's 0.54-1.24 um band"
    )
    assert report["resistance_scale"] > 0.0


def test_physics_report_carries_the_film_block_only_under_the_flag(tmp_path):
    from naviernet.physics.diagnostics import physics_report

    filmed, data, _ = _model(tmp_path, FILM)
    assert "film" in physics_report(filmed, data)

    plain, data, _ = _model(tmp_path, SHARP)
    assert "film" not in physics_report(plain, data)


def test_metrics_json_carries_the_film_block(tmp_path):
    """End to end: train a tiny run, evaluate it, and read the artifact the
    bench and the UI will read."""
    import json

    from naviernet.evaluation import evaluate
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, FILM)
    model, data, _ = train(cfg, paths)
    evaluate(cfg, model, data, paths)

    metrics = json.loads(paths.metrics_json.read_text())
    film = metrics["physics"]["film"]
    assert "film_evaporation_fraction" in film
    assert "dry_fraction" in film
    assert "film_pressure_peak_fraction" in film
    assert len(film["per_frame"][0]["delta_um"]) == len(film["stations"])
