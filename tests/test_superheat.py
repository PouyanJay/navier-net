"""Depletable superheat: letting the liquid cool below the inlet.

The Stage-B temperature is parameterised `theta = theta_in + (1 - theta_in)
sigmoid(raw)`, i.e. bounded BELOW by the inferred inlet superheat. But liquid
that has just boiled a bubble is COLDER than the inlet -- the latent heat came
out of it -- and that depletion is the feedback which shuts evaporation off as
the bubble blankets the wall.

Measured on the merged R4+film run, the bound is not a technicality: the field
collapsed onto its floor and stayed there. theta read 0.351072-0.351076 across
the WHOLE domain at EVERY frame (std 4e-07) -- a constant, exactly theta_in. So
evaporation reduced to `Ja (theta_in/r_int) delta`, a fixed multiple of the
interfacial area, which can only grow as the bubble elongates. The model
over-produced vapour by 3.9x by frame 11 with nothing able to stop it.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import staged_run as _staged_run

STAGE_B = ["model=stage_b"]
DEPLETABLE = [*STAGE_B, "model.depletable_superheat=true"]


def _model(tmp_path, overrides):
    from naviernet.models.pinn import BubblePINN

    cfg, _ = _staged_run(tmp_path, overrides)
    return BubblePINN(cfg), cfg


def _saturate(model, value: float):
    """Drive the temperature net's output layer hard, to reach the bound."""
    with torch.no_grad():
        model.nets["T"].mlp[-1].weight.zero_()
        model.nets["T"].mlp[-1].bias.fill_(value)


def test_the_default_bound_is_unchanged(tmp_path):
    """Opt-in: without the flag the temperature is exactly what it was."""
    model, cfg = _model(tmp_path, STAGE_B)
    assert cfg.model.depletable_superheat is False

    x = torch.rand(64, 3)
    _saturate(model, -30.0)
    assert torch.allclose(model.temperature(x), model.theta_in.expand(64, 1), atol=1e-5)


def test_the_superheat_can_be_depleted_toward_saturation(tmp_path):
    """The capability the flag buys: theta can fall to saturation, where the
    evaporation drive (proportional to theta) switches itself off. Without it,
    the coldest liquid the model can express is still the inlet."""
    model, _ = _model(tmp_path, DEPLETABLE)
    x = torch.rand(64, 3)

    _saturate(model, -30.0)
    coldest = model.temperature(x)
    assert float(coldest.max()) < float(model.theta_in), (
        "the liquid must be able to cool below the inlet it arrived at"
    )
    assert float(coldest.min()) >= 0.0, "and never below saturation"

    _saturate(model, +30.0)
    assert float(model.temperature(x).min()) > 0.99, "the wall is still the ceiling"


def test_the_inlet_superheat_becomes_a_boundary_condition(tmp_path):
    """With the bound gone, `theta_in` would be a dead parameter -- so it does
    the job it names instead: it anchors the temperature at the inlet."""
    from naviernet.physics.residuals import boundary_losses

    model, _ = _model(tmp_path, DEPLETABLE)
    inlet, walls = torch.rand(16, 3), torch.rand(16, 3)

    _saturate(model, -30.0)  # temperature far from theta_in everywhere
    without = boundary_losses(model, inlet, walls, 0.5)
    with_inlet = boundary_losses(model, inlet, walls, 0.5, theta_in=model.theta_in)
    assert float(with_inlet) > float(without), (
        "the inlet temperature condition must contribute to the boundary loss"
    )


def test_evaporation_shuts_off_as_the_superheat_depletes(tmp_path):
    """The mechanism, end to end: the Hardt-Wondra flux is proportional to the
    superheat, so a depleted liquid stops feeding the bubble. This is the
    feedback the bound made unreachable."""
    from naviernet.physics.groups import compute_groups
    from naviernet.physics.residuals import energy_residuals

    model, cfg = _model(tmp_path, DEPLETABLE)
    groups = compute_groups(cfg)
    x = torch.rand(64, 3, requires_grad=True)

    _saturate(model, +6.0)
    hot = energy_residuals(model, x, groups, model.r_int_star)
    _saturate(model, -6.0)
    cold = energy_residuals(model, x, groups, model.r_int_star)

    # The evaporation mass closure is `source - (1 - 1/rho) * evap`; with the
    # source unchanged, a smaller closure means a smaller evaporation flux.
    hot_flux = float((hot.src_closure - model.source(x)).abs().mean())
    cold_flux = float((cold.src_closure - model.source(x)).abs().mean())
    assert cold_flux < 0.25 * hot_flux, (
        f"depleting the superheat must throttle evaporation: {hot_flux:.4g} -> {cold_flux:.4g}"
    )


def test_depletable_superheat_requires_the_temperature_field(tmp_path):
    from naviernet.models.pinn import BubblePINN

    cfg, _ = _staged_run(tmp_path, ["model.depletable_superheat=true"])
    with pytest.raises(ValueError, match="'T'"):
        BubblePINN(cfg)


def test_the_flag_is_recorded_and_compat_checked(tmp_path):
    """It changes what the temperature weights mean without changing their
    shape, like every other architectural flag."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, DEPLETABLE)
    train(cfg, paths)
    ckpt = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    assert ckpt["depletable_superheat"] is True

    plain, _ = _staged_run(tmp_path, STAGE_B)
    with pytest.raises(ValueError, match="depletable_superheat"):
        load_model(plain, paths)


# --- the evaporation closure's gradient path ---------------------------------


def _closure_grads(model, cfg, x):
    """Which nets the evaporation mass closure can actually move."""
    from naviernet.physics.groups import compute_groups
    from naviernet.physics.residuals import energy_residuals

    res = energy_residuals(
        model,
        x,
        compute_groups(cfg),
        model.r_int_star,
        two_way_closure=bool(cfg.training.evap_closure_two_way),
    )
    model.zero_grad(set_to_none=True)
    (res.src_closure**2).mean().backward()
    return {
        name: float(sum((p.grad**2).sum() for p in net.parameters() if p.grad is not None))
        for name, net in model.nets.items()
    }


def test_by_default_the_closure_can_only_move_the_source(tmp_path):
    """The shipped closure detaches its whole target, so it pulls the source down
    to the drive and can never push the drive up to the source. Measured, that is
    why no weight on it ever raised the temperature: at weight 10 it closed an
    82x gap by shrinking the source 6x while theta did not move."""
    model, cfg = _model(tmp_path, STAGE_B)
    grads = _closure_grads(model, cfg, torch.rand(64, 3, requires_grad=True))

    assert grads["s"] > 0.0, "the source is what it trains"
    assert grads["T"] == 0.0, "and it cannot reach the temperature at all"


def test_two_way_the_closure_can_raise_the_temperature(tmp_path):
    """The blocker removed: the closure may now push the drive UP to meet the
    source it observes. That matters because the source is the only evaporation
    quantity the data constrains -- through the bubble's growth -- and the energy
    equation is too weakly forced to determine theta on its own."""
    model, cfg = _model(tmp_path, [*STAGE_B, "training.evap_closure_two_way=true"])
    grads = _closure_grads(model, cfg, torch.rand(64, 3, requires_grad=True))

    assert grads["T"] > 0.0, "the closure must be able to move the temperature"
    assert grads["s"] > 0.0, "and still the source"


def test_two_way_still_cannot_cheat_by_flattening_the_interface(tmp_path):
    """The protection the detach was added for, kept. The drive is
    `Ja (theta/r_int) delta`, and `delta` is the interfacial area -- left live,
    the cheapest way to satisfy the closure would be to flatten the interface
    rather than get the physics right. Only theta is freed; delta stays detached.
    """
    model, cfg = _model(tmp_path, [*STAGE_B, "training.evap_closure_two_way=true"])
    grads = _closure_grads(model, cfg, torch.rand(64, 3, requires_grad=True))

    assert grads["phi"] == 0.0, (
        "the closure must not be able to reshape the interface to satisfy itself"
    )


def test_the_energy_sink_itself_is_unchanged(tmp_path):
    """Only the CLOSURE's target is detached differently. The latent-heat sink in
    the energy equation is the real physical term and stays fully live."""
    from naviernet.physics.groups import compute_groups
    from naviernet.physics.residuals import energy_residuals

    x = torch.rand(64, 3, requires_grad=True)
    values = []
    for overrides in (STAGE_B, [*STAGE_B, "training.evap_closure_two_way=true"]):
        torch.manual_seed(0)
        model, cfg = _model(tmp_path, overrides)
        res = energy_residuals(model, x, compute_groups(cfg), model.r_int_star)
        values.append(res.energy.detach().clone())
    assert torch.allclose(values[0], values[1]), "the energy residual must not change"
