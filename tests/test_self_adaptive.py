"""Self-adaptive weights over space and time (``training.self_adaptive``): a
map that rises where a term's residual stays high, trained to ascend the loss
the fields descend."""

from __future__ import annotations

import math

import pytest
import torch

from naviernet import training
from naviernet.data.dataset import BubbleDataset
from naviernet.models.attention import SelfAdaptiveWeights
from naviernet.models.pinn import BubblePINN

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


def test_the_map_starts_uniform():
    attention = SelfAdaptiveWeights(("a", "b"), ratio=10.0)

    weights = attention(torch.rand(50, 3))

    assert set(weights) == {"a", "b"}
    for w in weights.values():
        assert w.shape == (50, 1)
        assert torch.allclose(w, torch.ones_like(w))


def test_the_weights_average_one_and_stay_within_the_ratio():
    attention = SelfAdaptiveWeights(("a",), ratio=10.0)
    with torch.no_grad():
        for p in attention.parameters():
            p.normal_(std=5.0)

    w = attention(torch.rand(400, 3))["a"]

    assert float(w.mean()) == pytest.approx(1.0, abs=1e-5)
    assert float(w.max() / w.min()) <= 10.0 + 1e-4


def test_a_step_raises_the_weight_where_the_residual_is_high():
    torch.manual_seed(0)
    attention = SelfAdaptiveWeights(("a",), ratio=10.0)
    opt = torch.optim.Adam(attention.parameters(), lr=1e-2)
    x = torch.rand(256, 3)
    residual_sq = (x[:, 0:1] > 0.5).float()  # high on the right half only
    right = x[:, 0] > 0.5

    for _ in range(50):
        opt.zero_grad()
        (attention(x)["a"] * residual_sq).mean().backward()
        opt.step()  # descends the loss as the fields would: the reversal ascends it

    w = attention(x)["a"].detach().ravel()
    assert float(w[right].mean()) > float(w[~right].mean())


def test_off_builds_no_map_and_leaves_every_net_bit_identical(tmp_path):
    cfg_off, paths = staged_capsule_run(tmp_path / "off", THERMAL)
    cfg_on, _ = staged_capsule_run(tmp_path / "on", [*THERMAL, "training.self_adaptive=energy"])
    priors = training._geometry_priors(cfg_off, BubbleDataset(cfg_off, paths))
    torch.manual_seed(3)
    plain = BubblePINN(cfg_off, geometry=priors)
    torch.manual_seed(3)
    adaptive = BubblePINN(cfg_on, geometry=priors)

    assert plain.attention is None
    assert adaptive.attention.keys == ("energy_liquid",)
    adaptive_state = adaptive.state_dict()
    for key, value in plain.state_dict().items():
        assert torch.equal(value, adaptive_state[key]), key


def test_all_weights_every_collocation_equation(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*THERMAL, "training.self_adaptive=all"])

    model = BubblePINN(cfg, geometry=training._geometry_priors(cfg, BubbleDataset(cfg, paths)))

    assert "energy_liquid" in model.attention.keys
    assert len(model.attention.keys) > 1


def test_a_self_adaptive_run_trains(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            *THERMAL,
            "training.self_adaptive=energy",
            "training.stage_b_warmup_steps=1",
            "training.steps=3",
            "training.log_every=1",
            "training.collocation_time_bins=2",
        ],
    )

    _, _, state = training.train(cfg, paths)

    assert math.isfinite(state["hist"][-1]["energy_liquid"])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("training.causal_weighting=false", "causal_weighting"),
        ("training.self_adaptive=both", "self_adaptive"),
        ("training.self_adaptive_ratio=1.0", "self_adaptive_ratio"),
    ],
)
def test_an_unsupported_self_adaptive_setup_is_refused(tmp_path, override, message):
    cfg, paths = staged_capsule_run(
        tmp_path, [*THERMAL, "training.self_adaptive=energy", override]
    )

    with pytest.raises(ValueError, match=message):
        training.train(cfg, paths)
