"""The gated causal schedule and the per-time-chunk residual record."""

from __future__ import annotations

import pytest
import torch

from naviernet.training import CausalSchedule, _causal_eps, _chunk_diagnostics, train

from .conftest import make_config, staged_run

SCHEDULE = [0.01, 1.0, 10.0]


def test_gated_eps_holds_until_every_chunk_weight_clears_the_gate():
    schedule = CausalSchedule(SCHEDULE, first_step=1, last_step=100, gate=0.99)

    schedule.observe(torch.tensor([1.0, 0.5, 0.2]))
    assert schedule.eps(50) == pytest.approx(0.01), "a chunk below the gate holds eps"

    schedule.observe(torch.tensor([1.0, 0.995, 0.991]))
    assert schedule.eps(50) == pytest.approx(1.0), "all chunks above the gate advance eps"

    schedule.observe(torch.tensor([1.0, 1.0, 1.0]))
    schedule.observe(torch.tensor([1.0, 1.0, 1.0]))
    assert schedule.eps(50) == pytest.approx(10.0), "eps stops at the schedule's last value"
    assert schedule.min_weight == pytest.approx(1.0)


def test_an_unarmed_gate_records_the_weights_but_holds_eps():
    """While a collocation term is held at zero the weights describe a loss the
    physics is absent from, so a clear gate must not advance the schedule."""
    schedule = CausalSchedule(SCHEDULE, first_step=1, last_step=100, gate=0.99)

    schedule.observe(torch.tensor([1.0, 1.0, 0.995]), armed=False)

    assert schedule.eps(50) == pytest.approx(0.01)
    assert schedule.min_weight == pytest.approx(0.995), "the record still shows the weights"
    schedule.observe(torch.tensor([1.0, 1.0, 0.995]), armed=True)
    assert schedule.eps(50) == pytest.approx(1.0), "the same weights advance it once armed"


def test_the_gate_is_armed_only_once_every_collocation_term_is_in_the_loss():
    from naviernet.training import _gate_armed, _loss_schedule

    tcfg = make_config(["training.stage_b_warmup_steps=10"]).training
    keys = frozenset({"vof", "div", "src", "darcy"})

    assert not _gate_armed(_loss_schedule(5, tcfg, ("darcy",)), keys), "warm-up: Darcy at 0"
    assert _gate_armed(_loss_schedule(11, tcfg, ("darcy",)), keys), "after the warm-up"
    assert _gate_armed(_loss_schedule(5, tcfg, ()), keys), "no Stage-B term to wait for"


def test_ungated_schedule_is_the_fixed_per_step_annealing():
    schedule = CausalSchedule(SCHEDULE, first_step=1, last_step=300, gate=None)
    schedule.observe(torch.tensor([1.0, 1.0]))  # a clear gate must not move a fixed schedule

    for step in (1, 150, 300):
        assert schedule.eps(step) == pytest.approx(_causal_eps(step, 1, 300, SCHEDULE))


def test_chunk_diagnostics_report_every_term_per_chunk_in_time_order():
    from naviernet.models.pinn import BubblePINN
    from naviernet.physics import registry
    from naviernet.physics.groups import compute_groups

    cfg = make_config(["model.hidden=8", "model.layers=2", "model.fourier_feats=4"])
    torch.manual_seed(0)
    model = BubblePINN(cfg)
    coll_equations = registry.collocation_equations(
        registry.enabled_equations(cfg.model.fields)
    )
    x_coll = torch.rand(32, 3, requires_grad=True)

    report = _chunk_diagnostics(model, x_coll, compute_groups(cfg), coll_equations, 4)

    assert report["t"] == sorted(report["t"]), "chunks run earliest to latest"
    for eq in coll_equations:
        assert len(report[eq.weight_key]) == 4, f"{eq.weight_key}: one mean per chunk"
        assert all(torch.isfinite(torch.tensor(report[eq.weight_key])))


GATED_RUN = [
    "training.causal_weighting=true",
    "training.causal_time_chunks=4",
    f"training.causal_eps_schedule=[{','.join(str(v) for v in SCHEDULE)}]",
    "training.causal_gate_min_weight=0.99",
    "training.log_chunk_residuals=true",
    "training.log_every=1",
]


def test_gated_run_records_its_causal_state_and_chunk_residuals(tmp_path):
    cfg, paths = staged_run(tmp_path, GATED_RUN)

    _, _, state = train(cfg, paths)

    assert {"causal_eps", "causal_min_weight"} <= state["hist"][-1].keys()
    assert state["hist"][-1]["causal_eps"] in SCHEDULE
    assert 0.0 <= state["hist"][-1]["causal_min_weight"] <= 1.0
    assert len(state["chunk_hist"]) == len(state["hist"])
    assert set(state["chunk_hist"][-1]) >= {"step", "t", "vof", "div", "src"}
    assert "causal_index" in state


def test_resumed_run_continues_from_the_saved_gate_index(tmp_path):
    cfg, paths = staged_run(tmp_path, GATED_RUN)
    _, _, state = train(cfg, paths)
    state["causal_index"] = len(SCHEDULE) - 1  # as if the gate had opened all the way
    checkpoint = torch.load(paths.checkpoint, weights_only=False)
    checkpoint["state"] = state
    torch.save(checkpoint, paths.checkpoint)

    _, _, resumed = train(cfg, paths, steps=1)

    assert resumed["hist"][-1]["causal_eps"] == pytest.approx(SCHEDULE[-1])
    assert resumed["causal_index"] == len(SCHEDULE) - 1


def test_history_is_unchanged_when_the_diagnostics_are_off(tmp_path):
    cfg, paths = staged_run(tmp_path, ["training.log_every=1"])

    _, _, state = train(cfg, paths)

    assert "chunk_hist" not in state
    assert not {"causal_eps", "causal_min_weight"} & state["hist"][-1].keys()


def test_one_pass_chunk_losses_equal_the_per_chunk_evaluation():
    """The chunk losses come from one evaluation of the time-ordered batch; each
    chunk's value must equal what evaluating that chunk on its own gives."""
    from naviernet.models.pinn import BubblePINN
    from naviernet.physics import registry
    from naviernet.physics.groups import compute_groups
    from naviernet.training import _chunk_losses, _time_chunks

    cfg = make_config(["model.hidden=8", "model.layers=2", "model.fourier_feats=4"])
    torch.manual_seed(0)
    model = BubblePINN(cfg)
    groups = compute_groups(cfg)
    equations = registry.collocation_equations(registry.enabled_equations(cfg.model.fields))
    weights = {eq.weight_key: 1.0 + i for i, eq in enumerate(equations)}
    x_coll = torch.rand(37, 3, requires_grad=True)  # 37: the chunks are unequal

    ctx = registry.LossContext(model, x_coll, groups=groups)
    one_pass = _chunk_losses(ctx, equations, weights, {}, 4)

    per_chunk = torch.stack(
        [
            sum(
                weights[eq.weight_key]
                * eq.term(registry.LossContext(model, chunk, groups=groups))
                for eq in equations
            )
            for chunk in _time_chunks(x_coll, 4)
        ]
    )
    assert one_pass.shape == (4,)
    assert torch.allclose(one_pass, per_chunk, rtol=1e-5, atol=1e-7)
    assert one_pass.requires_grad


def test_periodic_checkpoints_record_the_step_they_were_written_at(tmp_path, monkeypatch):
    """With ``checkpoint_every`` the run saves on the way, each save carrying its
    step count, so a run cut short resumes from its last save."""
    from naviernet import training

    cfg, paths = staged_run(
        tmp_path, ["training.steps=3", "training.checkpoint_every=1", "training.log_every=1"]
    )
    written: list[int] = []
    original = training._save_checkpoint

    def record(model, opt, state, cfg, paths):
        written.append(int(state["done"]))
        original(model, opt, state, cfg, paths)

    monkeypatch.setattr(training, "_save_checkpoint", record)

    training.train(cfg, paths)

    assert written == [1, 2, 3, 3], "every step on the way, then the final save"
    assert (
        torch.load(paths.checkpoint, map_location="cpu", weights_only=False)["state"]["done"]
        == 3
    )


def test_gradient_clipping_bounds_the_total_gradient_norm(tmp_path, monkeypatch):
    """With ``grad_clip`` the parameters' gradient norm never exceeds the bound
    at the optimiser step; without it the same step sees the raw norm."""
    from naviernet import training

    norms: list[float] = []
    original_step = torch.optim.Adam.step

    def measuring_step(self, *args, **kwargs):
        grads = [
            p.grad for group in self.param_groups for p in group["params"] if p.grad is not None
        ]
        norms.append(float(torch.norm(torch.stack([g.norm() for g in grads]))))
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", measuring_step)
    cfg, paths = staged_run(tmp_path, ["training.steps=3", "training.grad_clip=0.01"])

    training.train(cfg, paths)

    assert norms and max(norms) <= 0.01 * (1 + 1e-4)
