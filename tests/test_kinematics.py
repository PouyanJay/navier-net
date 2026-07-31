"""Kinematic growth constraints: physics-only late-window functionals.

The R2 terms — monotone bubble volume, volume-vs-source balance, and the
evaporation floor — are scalar functionals on a fixed quadrature grid over the
late time window, normalized by the dataset's supervised-tail growth rate.
These tests drive the real layers: synthetic tensors -> quadrature/references ->
terms -> trainer -> history record.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tests.conftest import make_config
from tests.test_hard_pin import (
    GROWING_H,
    GROWING_W,
    VAPOR_ROWS,
    _staged_run,
)

# Stage-A tiny runs have no T field, so the evap-floor term must be disabled.
TINY_KIN = [
    "training.kinematics=true",
    "training.kin_grid=6",
    "training.kin_times=3",
    "training.kin_weight_evap=0",
]


def test_kinematics_terms_train_and_are_logged(tmp_path):
    """The walking skeleton: a tiny run with the flag on trains, and the kin
    term values appear in the logged history record."""
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, [*TINY_KIN, "training.log_every=1"])
    _, _, state = train(cfg, paths)

    record = state["hist"][-1]
    assert "kin_mono" in record, f"kin terms missing from the history record: {record.keys()}"
    assert math.isfinite(record["kin_mono"])
    assert "kin_vbal" in record and math.isfinite(record["kin_vbal"])


def test_kinematics_off_leaves_the_record_clean(tmp_path):
    """Flag off (the default): no kin terms anywhere in the history."""
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, ["training.log_every=1"])
    assert cfg.training.kinematics is False, "kinematics must default off"
    _, _, state = train(cfg, paths)

    assert not any(k.startswith("kin_") for k in state["hist"][-1])


# --- Growth-rate reference (dataset layer) -----------------------------------


def test_supervised_growth_rate_matches_the_closed_form(tmp_path):
    """The synthetic event grows one column (2 px) per frame: the rate over the
    last two frames is known exactly."""
    from naviernet.data.dataset import BubbleDataset

    cfg, paths = _staged_run(tmp_path)
    data = BubbleDataset(cfg, paths)

    area = 1.1 * 0.5  # domain area of the synthetic grid
    dv = (2 / (GROWING_H * GROWING_W)) * area  # one 2-px column per frame
    assert data.supervised_growth_rate == pytest.approx(dv / 0.1, rel=1e-5)


def test_growth_rate_never_reads_held_out_frames(tmp_path):
    """Leakage guard: with a tail split, corrupting the held-out frame must not
    change the reference rate."""

    from naviernet.data.dataset import BubbleDataset

    split = ["training.val_fraction=0.25", "training.val_strategy=tail"]
    cfg, paths = _staged_run(tmp_path, split)
    clean = BubbleDataset(cfg, paths).supervised_growth_rate

    archive = dict(np.load(paths.tensors))
    meta = archive.pop("meta")
    archive["alpha"][-1] = 1.0  # held-out frame becomes all-vapour: a huge fake rate
    np.savez_compressed(paths.tensors, **archive, meta=meta)
    corrupted = BubbleDataset(cfg, paths)

    assert corrupted.supervised_growth_rate == clean


def test_growth_rate_rejects_a_static_bubble(tmp_path):
    from naviernet.data.dataset import BubbleDataset

    cfg, paths = _staged_run(tmp_path)
    archive = dict(np.load(paths.tensors))
    meta = archive.pop("meta")
    static = np.zeros_like(archive["alpha"])
    static[:, VAPOR_ROWS, 3:7] = 1.0
    archive["alpha"] = static
    np.savez_compressed(paths.tensors, **archive, meta=meta)

    with pytest.raises(ValueError, match="growing"):
        _ = BubbleDataset(cfg, paths).supervised_growth_rate


# --- Term mathematics (pure, analytic stub fields) ----------------------------


class _AnalyticModel:
    """A front at x = x0 + g*t with a prescribed uniform source and superheat --
    every kinematic quantity is known in closed form."""

    eps = 0.05
    fields = ["phi", "u", "v", "s", "T"]
    r_int_star = torch.tensor(1.0)

    def __init__(self, growth: float, source: float, theta: float = 0.0, x0: float = 0.4):
        self.growth, self.source_value, self.theta, self.x0 = growth, source, theta, x0

    def alpha(self, x, c=None):
        front = self.x0 + self.growth * x[:, 2:3]
        return torch.sigmoid((front - x[:, 0:1]) / self.eps)

    def source(self, x, c=None):
        return torch.full_like(x[:, 0:1], self.source_value)

    def temperature(self, x, c=None):
        return torch.full_like(x[:, 0:1], self.theta)


def _plan_and_cfg(**overrides):
    from naviernet.physics.kinematics import build_plan

    cfg = make_config(
        [
            "training.kinematics=true",
            # Grid spacing well under the interface half-thickness (h ~ 0.4*eps),
            # so the discrete quadrature matches the closed forms to ~1%.
            "training.kin_grid=48",
            "training.kin_times=4",
            *[f"training.{k}={v}" for k, v in overrides.items()],
        ]
    )

    class _Domain:
        x_min, x_max, y_min, y_max, t_min, t_max = 0.0, 1.0, 0.0, 0.5, 0.0, 1.0
        area = 0.5

    return build_plan(_Domain, r_ref=0.1, tcfg=cfg.training, device="cpu"), cfg.training


GROUPS = {"Ja": 0.4, "rho_ratio": 120.0}


def test_mono_is_zero_when_growth_clears_the_margin():
    """A front moving at 0.3 gives dQ/dt = 0.3*y_span = 0.15 = 1.5*r_ref, far
    above the 0.3 margin -- no penalty."""
    from naviernet.physics.kinematics import kinematic_losses

    plan, tcfg = _plan_and_cfg()
    _, record = kinematic_losses(
        _AnalyticModel(growth=0.3, source=0.0), plan, tcfg, GROUPS, 1.0
    )

    assert record["kin_mono"] == pytest.approx(0.0, abs=1e-8)


def test_mono_penalizes_a_shrinking_bubble():
    from naviernet.physics.kinematics import kinematic_losses

    plan, tcfg = _plan_and_cfg()
    # x0=0.7 keeps the retreating front well inside the domain over the whole
    # quadrature window (no boundary truncation of the sigmoid tail).
    _, record = kinematic_losses(
        _AnalyticModel(growth=-0.3, source=0.0, x0=0.7), plan, tcfg, GROUPS, 1.0
    )

    # dQ/dt = -0.15 -> normalized -1.5; violation (0.3 + 1.5)^2 = 3.24
    assert record["kin_mono"] == pytest.approx(3.24, rel=0.05)


def test_balance_vanishes_for_a_static_consistent_field():
    """No growth and no source: dQ/dt = 0 = integral of alpha*s -- exact zero."""
    from naviernet.physics.kinematics import kinematic_losses

    plan, tcfg = _plan_and_cfg(kin_margin_frac=0.0)
    _, record = kinematic_losses(
        _AnalyticModel(growth=0.0, source=0.0), plan, tcfg, GROUPS, 1.0
    )

    assert record["kin_vbal"] == pytest.approx(0.0, abs=1e-8)


def test_balance_penalizes_sourceless_growth():
    from naviernet.physics.kinematics import kinematic_losses

    plan, tcfg = _plan_and_cfg()
    _, record = kinematic_losses(
        _AnalyticModel(growth=0.3, source=0.0), plan, tcfg, GROUPS, 1.0
    )

    # dQ/dt/r_ref = 1.5 with S = 0 -> vbal = 1.5^2
    assert record["kin_vbal"] == pytest.approx(2.25, rel=0.05)


def test_evap_floor_penalizes_a_dead_source_against_a_live_drive():
    """theta > 0 keeps the (detached) drive alive; s = 0 violates the floor."""
    from naviernet.physics.kinematics import kinematic_losses

    plan, tcfg = _plan_and_cfg()
    _, record = kinematic_losses(
        _AnalyticModel(growth=0.0, source=0.0, theta=0.5), plan, tcfg, GROUPS, 1.0
    )

    assert record["kin_evap"] > 0.0


def test_evap_floor_is_satisfied_by_a_carrying_source():
    from naviernet.physics.kinematics import kinematic_losses

    plan, tcfg = _plan_and_cfg()
    _, record = kinematic_losses(
        _AnalyticModel(growth=0.0, source=5.0, theta=0.1), plan, tcfg, GROUPS, 1.0
    )

    assert record["kin_evap"] == pytest.approx(0.0, abs=1e-8)


def test_evap_drive_is_detached_from_the_temperature_net(tmp_path):
    """The floor must push the source UP, never drag the (healthy) temperature
    down: backward through the kin total leaves the T net without gradients."""
    from naviernet.models.pinn import BubblePINN
    from naviernet.physics.kinematics import kinematic_losses

    cfg, _ = _staged_run(
        tmp_path, ["model=stage_b", "training.kinematics=true", "training.kin_grid=6"]
    )
    torch.manual_seed(0)
    model = BubblePINN(cfg)
    plan, tcfg = _plan_and_cfg()

    total, _ = kinematic_losses(model, plan, tcfg, GROUPS, 1.0)
    total.backward()

    t_grads = [p.grad for p in model.nets["T"].parameters() if p.grad is not None]
    assert not any(g.abs().sum() > 0 for g in t_grads), "kevap leaked gradient into T"
    s_grads = [p.grad for p in model.nets["s"].parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in s_grads), "kevap/vbal never reached the source net"


# --- Config validation and composition ---------------------------------------


def test_kinematics_evap_requires_the_temperature_field(tmp_path):
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, ["training.kinematics=true"])  # stage-A fields, evap on

    with pytest.raises(ValueError, match="kin_weight_evap"):
        train(cfg, paths)


@pytest.mark.parametrize(
    "bad",
    [
        "training.kin_grid=1",
        "training.kin_times=0",
        "training.kin_time_frac=0",
        "training.kin_time_frac=1.5",
        "training.kin_margin_frac=-0.1",
    ],
)
def test_kinematics_rejects_degenerate_config(tmp_path, bad):
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, [*TINY_KIN, bad])

    with pytest.raises(ValueError):
        train(cfg, paths)


def test_kinematics_trains_stage_b_with_all_three_terms(tmp_path):
    from naviernet.training import train

    cfg, paths = _staged_run(
        tmp_path,
        [
            "model=stage_b",
            "training.kinematics=true",
            "training.kin_grid=6",
            "training.kin_times=3",
            "training.log_every=1",
        ],
    )
    _, _, state = train(cfg, paths)

    record = state["hist"][-1]
    assert all(k in record for k in ("kin_mono", "kin_vbal", "kin_evap"))


def test_kinematics_composes_with_causal_and_rba(tmp_path):
    from naviernet.training import train

    for name, extra in (
        ("causal", "training.causal_weighting=true"),
        ("rba", "training.weighting=rba"),
    ):
        cfg, paths = _staged_run(tmp_path / name, [*TINY_KIN, extra])
        _, _, state = train(cfg, paths)
        assert "kin_mono" in state["hist"][-1], f"kin terms missing under {name}"


def test_kinematics_on_diverges_from_off(tmp_path):
    """The terms must reach the optimized graph: same seed, different weights."""
    from naviernet.training import train

    cfg_off, paths_off = _staged_run(tmp_path / "off")
    train(cfg_off, paths_off)
    cfg_on, paths_on = _staged_run(tmp_path / "on", TINY_KIN)
    train(cfg_on, paths_on)

    off = torch.load(paths_off.checkpoint, weights_only=False)["model"]
    on = torch.load(paths_on.checkpoint, weights_only=False)["model"]
    assert any(not torch.equal(off[k], on[k]) for k in off)


# --- Front-position metric (bench) -------------------------------------------


def test_front_position_picks_the_edge_farther_from_the_anchor():
    """The front is the mask's moving edge -- the x-extent edge FARTHER from the
    root anchor (root_position's mirror); NaN for an empty mask."""
    from naviernet.evaluation import front_position

    xs = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    mask = np.zeros((3, 11), dtype=bool)
    mask[1, 2:8] = True  # bubble spanning x* 0.2 .. 0.7

    assert front_position(mask, xs, x_anchor=0.15) == pytest.approx(0.7)
    assert front_position(mask, xs, x_anchor=0.75) == pytest.approx(0.2)
    assert math.isnan(front_position(np.zeros((3, 11), dtype=bool), xs, x_anchor=0.5))


# --- Joint (multi-dataset) runs ----------------------------------------------


def test_joint_kinematics_trains_and_logs(tmp_path):
    """Each dataset gets its own quadrature and growth-rate reference; the
    logged kin terms are the across-dataset means."""
    from naviernet.training import train
    from tests.test_hard_pin import _staged_joint_run

    cfg, paths = _staged_joint_run(tmp_path, [*TINY_KIN, "training.log_every=1"])
    _, _, state = train(cfg, paths)

    record = state["hist"][-1]
    assert "kin_mono" in record and math.isfinite(record["kin_mono"])
    assert "kin_vbal" in record


def test_joint_kinematics_composes_with_pin_and_causal(tmp_path):
    """The full bench stack: joint + hard pin + causal + kinematics trains."""
    from naviernet.training import train
    from tests.test_hard_pin import _staged_joint_run

    cfg, paths = _staged_joint_run(
        tmp_path, [*TINY_KIN, "model.hard_pin=true", "training.causal_weighting=true"]
    )
    train(cfg, paths)

    assert paths.checkpoint.exists()
