"""The exact kinematic condition (``model.hard_kinematics``): on the whole
interface the liquid's normal velocity is the interface's own normal speed, by
construction; the tangential velocity and the far field are the network's."""

from __future__ import annotations

import pytest
import torch

from naviernet import training
from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BubblePINN

from .conftest import staged_capsule_run

FRONT = ["model=stage_b", "model.front_geometry=true", "model.sharp_interface=true"]


def _pair(tmp_path, extra=()):
    """The same seeded model with the condition off and on."""
    cfg, paths = staged_capsule_run(tmp_path, [*FRONT, *extra])
    data = BubbleDataset(cfg, paths)
    torch.manual_seed(0)
    soft = BubblePINN(cfg, geometry=training._geometry_priors(cfg, data))
    cfg_hard, _ = staged_capsule_run(tmp_path, [*FRONT, *extra, "model.hard_kinematics=true"])
    hard = BubblePINN(cfg_hard, geometry=training._geometry_priors(cfg_hard, data))
    hard.load_state_dict(soft.state_dict())
    return data, soft, hard


def _front(model, data, k: int = 4):
    t = torch.tensor([[float(data.t[k])]], requires_grad=True)
    return model.front(t, 32, 8)


def test_the_normal_velocity_is_the_interface_speed_everywhere_on_it(tmp_path):
    # Exact at every sample but the four where the caps join the body: the
    # outline turns there by about 0.01 rad, so its normal is one-sided.
    data, _, hard = _pair(tmp_path)
    for k in (1, 4, len(data.t) - 1):
        front = _front(hard, data, k)
        u, v = hard.velocity(front.points)
        gap = (u * front.normal[:, 0:1] + v * front.normal[:, 1:2] - front.normal_speed).abs()

        assert int((gap > 1e-5).sum()) <= 4, f"frame row {k}"
        assert float(gap.max()) < 0.05


def test_sliding_along_the_interface_and_the_far_field_are_left_free(tmp_path):
    data, soft, hard = _pair(tmp_path)
    front = _front(hard, data)
    tangent = torch.cat([-front.normal[:, 1:2], front.normal[:, 0:1]], dim=1)
    (us, vs), (uh, vh) = soft.velocity(front.points), hard.velocity(front.points)

    slide = ((uh - us) * tangent[:, 0:1] + (vh - vs) * tangent[:, 1:2]).abs()
    assert int((slide > 1e-5).sum()) <= 4, "only the cap joins, where the normal is one-sided"

    far = torch.tensor([[float(data.x.max()) - 0.01, float(data.y.mean()), float(data.t[4])]])
    assert torch.allclose(hard.velocity(far)[0], soft.velocity(far)[0], atol=1e-6)


def test_the_correction_keeps_the_mirror_symmetry(tmp_path):
    data, _, hard = _pair(tmp_path, ["model.mirror_symmetry=true"])
    torch.manual_seed(1)
    x = torch.rand(128, 3) * torch.tensor([float(data.x.max()), 0.0, float(data.t.max())])
    axis = hard.mirror_axis
    x[:, 1] = axis + (torch.rand(128) - 0.5) * 0.6
    mirrored = torch.cat([x[:, :1], 2 * axis - x[:, 1:2], x[:, 2:]], dim=1)

    (u, v), (um, vm) = hard.velocity(x), hard.velocity(mirrored)

    assert torch.allclose(u, um, atol=1e-5) and torch.allclose(v, -vm, atol=1e-5)


def test_the_condition_needs_the_explicit_geometry(tmp_path):
    cfg, _ = staged_capsule_run(tmp_path, ["model=stage_b", "model.hard_kinematics=true"])

    with pytest.raises(ValueError, match="hard_kinematics"):
        BubblePINN(cfg)


def test_a_run_with_the_exact_condition_trains_and_scores_no_kinematic_error(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [
            *FRONT,
            "model.momentum_closure=brinkman",
            "model.hard_kinematics=true",
            "training.stage_b_warmup_steps=1",
            "training.steps=3",
            "training.log_every=1",
        ],
    )

    _, _, state = training.train(cfg, paths)

    last = state["hist"][-1]
    assert last["kinematic"] < 1e-6
    assert torch.isfinite(torch.tensor(last["brinkman"]))
