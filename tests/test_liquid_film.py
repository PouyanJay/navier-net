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
