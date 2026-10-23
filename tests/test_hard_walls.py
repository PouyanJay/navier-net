"""No flow through the side walls by construction (``model.hard_walls``): the
velocity across the channel vanishes at both walls, the flow along it is the
network's."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from naviernet import training
from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BubblePINN

from .conftest import staged_run, write_capsule

WALLED = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.walls_at=physical",
]


def _write_with_walls(path) -> None:
    """The analytic capsule with its physical walls recorded just outside the
    imaged span, as preprocessing records them."""
    write_capsule(path)
    archive = dict(np.load(path))
    meta = json.loads(str(archive["meta"]))
    ys = archive["y_star"]
    lo, hi = float(ys.min()) - 0.01, float(ys.max()) + 0.01
    meta.update(y_wall_lo_star=lo, y_wall_hi_star=hi, y_center_star=0.5 * (lo + hi))
    archive["meta"] = json.dumps(meta)
    np.savez_compressed(path, **archive)


def staged_capsule_run(tmp_path, overrides=None):
    return staged_run(tmp_path, overrides, write=_write_with_walls)


def _pair(tmp_path, extra=()):
    """The same seeded model with the walls as a penalty and built in."""
    cfg, paths = staged_capsule_run(tmp_path, [*WALLED, *extra])
    data = BubbleDataset(cfg, paths)
    torch.manual_seed(0)
    soft = BubblePINN(cfg, geometry=training._geometry_priors(cfg, data))
    cfg_hard, _ = staged_capsule_run(tmp_path, [*WALLED, *extra, "model.hard_walls=true"])
    hard = BubblePINN(cfg_hard, geometry=training._geometry_priors(cfg_hard, data))
    hard.load_state_dict(soft.state_dict())
    return data, soft, hard


def _line(data, y: float, t: float, n: int = 64) -> torch.Tensor:
    xs = torch.linspace(float(data.x.min()), float(data.x.max()), n)
    return torch.stack([xs, torch.full_like(xs, y), torch.full_like(xs, t)], dim=1)


def test_nothing_crosses_either_wall(tmp_path):
    data, soft, hard = _pair(tmp_path)
    for t in (float(data.t[0]), float(data.t[-1])):
        for wall in (float(data.y_lo), float(data.y_hi)):
            _, v = hard.velocity(_line(data, wall, t))

            assert float(v.abs().max()) < 1e-6
    _, v_soft = soft.velocity(_line(data, float(data.y_lo), float(data.t[0])))
    assert float(v_soft.abs().max()) > 1e-4, "the penalty alone does not hold it"


def test_the_flow_along_the_channel_and_the_centreline_are_the_networks(tmp_path):
    data, soft, hard = _pair(tmp_path)
    centre = 0.5 * (float(data.y_lo) + float(data.y_hi))
    x = _line(data, centre, float(data.t[3]))

    (us, vs), (uh, vh) = soft.velocity(x), hard.velocity(x)

    assert torch.allclose(uh, us)
    assert torch.allclose(vh, vs, atol=1e-6), "the factor is one on the centreline"


def test_v_stays_odd_under_mirror_symmetry(tmp_path):
    data, _, hard = _pair(tmp_path, ["model.mirror_symmetry=true"])
    axis = hard.mirror_axis
    x = torch.rand(40, 3) * torch.tensor([float(data.x.max()), 0.3, float(data.t[-1])])
    x[:, 1] += axis - 0.15
    mirrored = x.clone()
    mirrored[:, 1] = 2 * axis - x[:, 1]

    _, v = hard.velocity(x)
    _, v_mirror = hard.velocity(mirrored)

    assert torch.allclose(v, -v_mirror, atol=1e-6)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("model.walls_at=crop", "walls_at=physical"),
        ("model.hard_kinematics=true", "hard_kinematics"),
    ],
)
def test_an_unsupported_hard_walls_setup_is_refused(tmp_path, override, message):
    cfg, paths = staged_capsule_run(tmp_path, [*WALLED, "model.hard_walls=true", override])
    data = BubbleDataset(cfg, paths)

    with pytest.raises(ValueError, match=message):
        BubblePINN(cfg, geometry=training._geometry_priors(cfg, data))


def test_a_hard_walls_run_trains(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            *WALLED,
            "model.hard_walls=true",
            "training.stage_b_warmup_steps=1",
            "training.steps=3",
            "training.log_every=1",
        ],
    )

    _, _, state = training.train(cfg, paths)

    assert math.isfinite(state["hist"][-1]["div"])
