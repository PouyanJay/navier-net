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
