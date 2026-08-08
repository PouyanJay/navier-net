"""The liquid film delta(s,t) between the bubble and the wall (T0: deposition).

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
    film_keys = {k for k in filmed_state if k.startswith("film.")}
    assert film_keys, "the film net must live under the 'film.' namespace"
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
    assert t == pytest.approx(t_dry, rel=0.05)
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
    never writes them. The residual's graph must touch the film net alone."""
    from naviernet.physics.film import depletion_residual
    from naviernet.physics.groups import compute_groups

    model, data, cfg = _model(tmp_path, FILM)
    groups = compute_groups(cfg)

    loss = (depletion_residual(model, torch.tensor([[0.5]]), groups, stations=8) ** 2).mean()
    loss.backward()

    outsiders = [
        name
        for name, p in model.named_parameters()
        if not name.startswith("film.") and p.grad is not None and float(p.grad.abs().max()) > 0
    ]
    assert not outsiders, f"depletion reached beyond the film net: {outsiders}"
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
