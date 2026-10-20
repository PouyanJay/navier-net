"""Sharing the fields among a training step's equations
(``BubblePINN.caching_fields``) changes no value: a run with it ends where a run
without it does, to rounding."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from naviernet import training
from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BubblePINN

from .conftest import staged_capsule_run

FULL = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.thermal_model=sharp",
    "model.pressure_representation=two_phase",
    "model.momentum_closure=brinkman",
    "model.hard_kinematics=true",
    "experiment.T_inlet_C=56.6",
    "experiment.T_init_C=58.0",
    "training.stage_b_warmup_steps=2",
    "training.steps=6",
]


def _trained(tmp_path, name: str) -> dict[str, torch.Tensor]:
    cfg, paths = staged_capsule_run(tmp_path, [*FULL, f"run_name={name}"])
    model, _, _ = training.train(cfg, paths)
    return model.state_dict()


def test_a_run_sharing_its_fields_ends_where_one_recomputing_them_does(tmp_path, monkeypatch):
    shared = _trained(tmp_path, "shared")
    monkeypatch.setattr(BubblePINN, "caching_fields", lambda self: nullcontext())
    recomputed = _trained(tmp_path, "recomputed")

    worst = max(float((shared[k] - recomputed[k]).abs().max()) for k in shared)
    assert worst < 1e-5, f"largest parameter difference {worst:.2e}"


def test_a_batch_is_served_its_own_fields_only(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, FULL)
    data = BubbleDataset(cfg, paths)
    model = BubblePINN(cfg, geometry=training._geometry_priors(cfg, data))
    a = torch.rand(16, 3).requires_grad_(True)
    b = a.detach().clone().requires_grad_(True)

    with model.caching_fields():
        first, again, other = model.velocity(a), model.velocity(a), model.velocity(b)

    assert again[0] is first[0], "the same batch is computed once"
    assert other[0] is not first[0], "an equal but distinct batch gets its own graph"
    assert model._field_cache is None, "nothing outlives the block"
