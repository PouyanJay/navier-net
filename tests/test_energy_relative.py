"""Training the liquid energy equation on its error relative to its own terms
(``training.energy_relative``), so the open liquid counts as the interface does."""

from __future__ import annotations

import pytest
import torch

from naviernet import training
from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BubblePINN
from naviernet.physics import registry
from naviernet.physics.groups import compute_groups
from naviernet.physics.residuals import energy_liquid_balance, energy_liquid_residual

from .conftest import staged_capsule_run

THERMAL = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.thermal_model=sharp",
    "model.momentum_closure=brinkman",
    "experiment.T_inlet_C=56.6",
    "experiment.T_init_C=58.0",
    "training.causal_weighting=true",
]


def _context(tmp_path, extra=()):
    cfg, paths = staged_capsule_run(tmp_path, [*THERMAL, *extra])
    data = BubbleDataset(cfg, paths)
    torch.manual_seed(0)
    model = BubblePINN(cfg, geometry=training._geometry_priors(cfg, data))
    x = torch.rand(256, 3) * torch.tensor([float(data.x.max()), float(data.y.max()), 1.0])
    x.requires_grad_(True)
    return cfg, model, registry.LossContext(model, x, groups=compute_groups(cfg))


def test_the_balance_keeps_the_residual_and_adds_the_terms_size(tmp_path):
    cfg, model, ctx = _context(tmp_path)
    groups = compute_groups(cfg)

    residual, scale = energy_liquid_balance(model, ctx.x_coll, groups)

    assert torch.allclose(residual, energy_liquid_residual(model, ctx.x_coll, groups))
    assert not scale.requires_grad
    assert bool((scale >= residual.detach().abs() - 1e-6).all()), (
        "no residual exceeds its terms"
    )


def test_off_the_objective_is_the_squared_residual(tmp_path):
    _, _, ctx = _context(tmp_path)
    energy = next(e for e in registry.REGISTRY if e.id == "energy_liquid")

    assert torch.equal(registry.training_pointwise(energy, ctx), ctx.energy_liquid_res**2)


def test_on_the_objective_is_the_residual_over_its_floored_terms(tmp_path):
    _, _, ctx = _context(
        tmp_path, ["training.energy_relative=true", "training.energy_relative_floor=0.05"]
    )
    energy = next(e for e in registry.REGISTRY if e.id == "energy_liquid")

    expected = (ctx.energy_liquid_res / (ctx.energy_liquid_scale + 0.05)) ** 2
    assert torch.allclose(registry.training_pointwise(energy, ctx), expected)
    assert torch.equal(energy.pointwise(ctx), ctx.energy_liquid_res**2), (
        "reported stays absolute"
    )


def test_a_relative_energy_run_trains(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            *THERMAL,
            "training.energy_relative=true",
            "training.stage_b_warmup_steps=1",
            "training.steps=3",
            "training.log_every=1",
            "training.collocation_time_bins=2",
        ],
    )

    _, _, state = training.train(cfg, paths)

    assert torch.isfinite(torch.tensor(state["hist"][-1]["energy_liquid"]))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("training.causal_weighting=false", "causal_weighting"),
        ("training.energy_relative_floor=0.0", "energy_relative_floor"),
    ],
)
def test_an_unsupported_relative_energy_setup_is_refused(tmp_path, override, message):
    cfg, paths = staged_capsule_run(
        tmp_path, [*THERMAL, "training.energy_relative=true", override]
    )

    with pytest.raises(ValueError, match=message):
        training.train(cfg, paths)
