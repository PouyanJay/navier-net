"""How the front's samples count in the Young-Laplace loss
(``model.laplace_root`` and ``model.laplace_weighting``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from naviernet import training
from naviernet.physics.registry import _laplace_weights, in_view_mean, per_instant_mean
from naviernet.physics.residuals import front_arc_lengths, root_cap_mask

from .conftest import staged_capsule_run

SHARP = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.momentum_closure=brinkman",
    "model.pressure_representation=two_phase",
]


def _stadium(t: float = 0.0, stretch: float = 1.0) -> SimpleNamespace:
    """A 4 by 2 outline in the fronts' layout: upper body, lower body, root cap,
    nose cap. The body's samples stop one step short of the seams."""
    upper = [[x, 1.0] for x in (1.0, 2.0, 3.0)]
    lower = [[x, -1.0] for x in (1.0, 2.0, 3.0)]
    root = [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]]
    nose = [[4.0, -1.0], [4.0, 0.0], [4.0, 1.0]]
    xy = torch.tensor(upper + lower + root + nose) * stretch
    n = len(xy)
    return SimpleNamespace(
        points=torch.cat([xy, torch.full((n, 1), t)], dim=1),
        u=torch.tensor([[0.25]] * 6 + [[0.0]] * 3 + [[1.0]] * 3),
        side=torch.tensor([[1.0]] * 3 + [[-1.0]] * 3 + [[-1.0], [0.0], [1.0]] * 2),
        on_cap=torch.tensor([[0.0]] * 6 + [[1.0]] * 6),
    )


def _context(front, **options) -> SimpleNamespace:
    return SimpleNamespace(
        front=front,
        model=SimpleNamespace(cfg=SimpleNamespace(model=SimpleNamespace(**options))),
    )


def test_each_sample_stands_for_its_length_of_the_outline():
    lengths = front_arc_lengths(_stadium())

    expected = [1.5, 1.0, 1.5] * 2 + [0.5, 1.0, 0.5] * 2
    assert lengths.squeeze(1).tolist() == pytest.approx(expected)


def test_the_lengths_average_one_at_every_instant_however_big_the_outline():
    small, big = _stadium(t=0.0), _stadium(t=1.0, stretch=3.0)
    front = SimpleNamespace(
        **{
            k: torch.cat([getattr(small, k), getattr(big, k)])
            for k in ("points", "u", "side", "on_cap")
        }
    )

    lengths = front_arc_lengths(front).squeeze(1)

    assert float(lengths[:12].mean()) == pytest.approx(1.0)
    assert float(lengths[12:].mean()) == pytest.approx(1.0)
    assert lengths[:12].tolist() == pytest.approx(lengths[12:].tolist())


def test_the_root_cap_is_only_the_caps_low_end():
    assert (
        root_cap_mask(_stadium()).squeeze(1).tolist() == [False] * 6 + [True] * 3 + [False] * 3
    )


def test_the_defaults_count_every_sample_once():
    weights = _laplace_weights(
        _context(_stadium(), laplace_root="include", laplace_weighting="samples")
    )

    assert bool((weights == 1.0).all())


def test_an_excluded_root_carries_no_weight_and_the_rest_keep_theirs():
    front = _stadium()

    weights = _laplace_weights(
        _context(front, laplace_root="exclude", laplace_weighting="length")
    )

    assert bool((weights[6:9] == 0.0).all())
    assert weights[:6].squeeze(1).tolist() == pytest.approx([1.5, 1.0, 1.5] * 2)


def test_an_impossible_root_demand_costs_nothing_once_excluded():
    front = _stadium()
    squared = torch.ones(12, 1)
    squared[6:9] = 1.0e4
    weights = _laplace_weights(
        _context(front, laplace_root="exclude", laplace_weighting="samples")
    )

    assert float(in_view_mean(squared, front.points[:, 0:1], None, weights)) == pytest.approx(
        1.0
    )
    assert float(in_view_mean(squared, front.points[:, 0:1], None)) > 1.0e3


def test_the_mean_per_instant_honours_the_weights():
    values = torch.tensor([[1.0], [3.0], [10.0]])
    times = torch.tensor([0.0, 0.0, 1.0])
    weights = torch.tensor([[3.0], [1.0], [2.0]])

    assert per_instant_mean(values, times, weights).squeeze(1).tolist() == pytest.approx(
        [1.5, 10.0]
    )


def test_the_view_limit_still_applies_with_weights():
    squared = torch.tensor([[1.0], [3.0], [100.0]])
    x = torch.tensor([[0.5], [1.0], [1.2]])
    weights = torch.tensor([[1.0], [3.0], [5.0]])

    assert float(in_view_mean(squared, x, 1.0, weights)) == pytest.approx(2.5)


def test_a_run_with_the_root_left_out_and_length_weights_trains(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            *SHARP,
            "model.laplace_root=exclude",
            "model.laplace_weighting=length",
            "training.stage_b_warmup_steps=1",
            "training.steps=3",
            "training.log_every=1",
        ],
    )

    _, _, state = training.train(cfg, paths)

    assert torch.isfinite(torch.tensor(state["hist"][-1]["laplace"]))


@pytest.mark.parametrize(
    ("override", "name"),
    [
        ("model.laplace_root=maybe", "laplace_root"),
        ("model.laplace_weighting=area", "laplace_weighting"),
    ],
)
def test_an_unknown_choice_is_refused(tmp_path, override, name):
    cfg, paths = staged_capsule_run(tmp_path, [*SHARP, override])

    with pytest.raises(ValueError, match=name):
        training.train(cfg, paths)
