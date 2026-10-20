"""Each equation's error relative to its terms (naviernet.relative_error)."""

from __future__ import annotations

import math

import torch

from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BubblePINN
from naviernet.physics.groups import compute_groups
from naviernet.relative_error import _balance, relative_errors
from naviernet.training import _geometry_priors

from .conftest import staged_capsule_run


def test_the_balance_is_the_sum_against_the_summed_magnitudes():
    terms = [torch.tensor([[3.0]]), torch.tensor([[-2.5]])]

    residual, scale = _balance(terms, 1.0)

    assert float(residual) == 0.5 and float(scale) == 5.5


def test_every_equation_the_model_has_is_reported_by_region(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            "model=stage_b",
            "model.front_geometry=true",
            "model.sharp_interface=true",
            "model.thermal_model=sharp",
            "model.momentum_closure=brinkman",
            "experiment.T_inlet_C=56.6",
            "experiment.T_init_C=58.0",
        ],
    )
    data = BubbleDataset(cfg, paths)
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))

    result = relative_errors(model, compute_groups(cfg), data, float(data.t[2]), 64, 21)

    assert {"temperature", "momentum", "continuity", "edge transport"} <= set(result)
    assert "vapour" in result["continuity"]
    for name in ("temperature", "momentum", "continuity", "edge transport"):
        assert {"near edge", "all liquid"} <= set(result[name])
        assert all(math.isfinite(v) and v >= 0 for v in result[name].values())
