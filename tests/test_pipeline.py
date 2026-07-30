"""Trainer mechanics and end-to-end behaviour against the real dataset."""

from __future__ import annotations

import pytest
import torch

from naviernet.data.preprocess import MIN_USABLE_FRAMES, usable_frame_numbers
from naviernet.pipeline import Pipeline
from naviernet.training import REBALANCED_TERMS, _rebalance
from naviernet.utils.paths import RunPaths

from .conftest import make_config


def test_rebalance_equalises_gradient_contributions():
    """A term with a small gradient gets a larger weight, and vice versa."""
    weights = {"data": 10.0, "vof": 1.0, "div": 1.0, "src": 0.1, "bc": 1.0}
    before = dict(weights)
    # `data` contributes 10 * 1 = 10; vof's gradient is tiny, div's is huge.
    _rebalance(weights, {"data": 1.0, "vof": 0.01, "div": 100.0, "bc": 1.0})

    assert weights["vof"] > before["vof"]
    assert weights["div"] < before["div"]
    assert weights["src"] == before["src"], "src is a fixed penalty, not rebalanced"
    assert weights["data"] == before["data"], "data is the reference scale"


def test_rebalance_stays_within_bounds():
    weights = dict.fromkeys(("data", "vof", "div", "src", "bc"), 1.0)
    for _ in range(50):
        _rebalance(weights, {"data": 1.0, "vof": 1e-30, "div": 1e30, "bc": 1.0})

    for term in REBALANCED_TERMS:
        assert 1e-2 <= weights[term] <= 1e3


def test_stage_a_training_is_byte_for_byte_unchanged(tmp_path):
    """Guards the equation-registry refactor.

    The Stage-A loss trajectory and rebalanced weights must match the golden
    captured from the pre-registry trainer. If deriving the active terms from
    the registry ever changes a number, this fails loudly.
    """
    import json
    from pathlib import Path

    from naviernet.training import train

    golden = json.loads(
        (Path(__file__).parent / "fixtures" / "stage_a_golden.json").read_text()
    )
    cfg = make_config([f"paths.root={tmp_path}", *golden["overrides"]])
    paths = RunPaths.from_config(cfg)
    paths.ensure()
    paths.tensors.parent.mkdir(parents=True, exist_ok=True)
    _write_tensors(paths.tensors, list(range(1, 9)), n_event=8)

    _, _, state = train(cfg, paths)

    assert len(state["hist"]) == len(golden["hist"])
    for got, want in zip(state["hist"], golden["hist"], strict=True):
        assert got.keys() == want.keys()
        for key, wanted in want.items():
            assert got[key] == pytest.approx(wanted, rel=1e-6, abs=1e-8), f"{key} drifted"
    for key, wanted in golden["w"].items():
        assert state["w"][key] == pytest.approx(wanted, rel=1e-6, abs=1e-8), (
            f"weight {key} drifted"
        )


def test_joint_training_over_two_datasets_writes_one_conditioned_checkpoint(tmp_path):
    """A run with `datasets=[a, b]` trains ONE model, conditioned on each
    dataset's dimensionless groups, and writes a single checkpoint — the joint
    (transfer-learning) path, distinct from N separate single-dataset runs."""
    from naviernet.physics.groups import N_COND, compute_groups
    from naviernet.training import train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            "datasets=[ds_a,ds_b]",
            "run_name=joint",
            *_TINY_TRAIN,
            "training.steps=2",
            "training.log_every=1",
            "training.rebalance_every=1",
        ]
    )
    run_paths = RunPaths.from_config(cfg)
    # Two synthetic datasets with distinct regimes (different wall heat flux), each
    # carrying its own groups in its tensors — as preprocess would have written.
    for name, q_wall in (("ds_a", 2.0), ("ds_b", 5.0)):
        ds_paths = run_paths.for_dataset(name)
        ds_paths.processed_dir.mkdir(parents=True, exist_ok=True)
        groups = compute_groups(make_config([f"experiment.q_wall_W_cm2={q_wall}"]))
        _write_tensors(ds_paths.tensors, list(range(1, 9)), n_event=8, groups=groups)

    model, _, state = train(cfg, run_paths)

    assert model.n_cond == N_COND  # one conditioned model, not two plain ones
    assert state["done"] == 2
    assert len(state["hist"]) == 2
    assert run_paths.checkpoint.exists()  # a single joint checkpoint
    saved = torch.load(run_paths.checkpoint, map_location="cpu", weights_only=False)
    assert saved["datasets"] == ["ds_a", "ds_b"]
    assert saved["n_cond"] == N_COND


@pytest.mark.parametrize("mode", ["weight", "march"])
def test_joint_training_runs_with_causal_weighting_enabled(tmp_path, mode):
    """The causal path is wired into the joint loop too, for BOTH modes: a conditioned
    two-dataset run with causal weighting ON reweights each dataset's collocation
    residual by time -- diverging from the causal-off joint run under an identical
    seed -- and still trains to finite losses (no crash from the per-bin passes)."""
    import math

    from naviernet.training import train

    def run(run_name: str, extra: list[str]):
        cfg = make_config(
            [
                f"paths.root={tmp_path}",
                "datasets=[ds_a,ds_b]",
                f"run_name={run_name}",
                *_TINY_TRAIN,
                "training.steps=3",
                "training.log_every=1",
                "training.seed=0",
                *extra,
            ]
        )
        run_paths = RunPaths.from_config(cfg)
        _stage_joint_datasets(run_paths)
        _, _, state = train(cfg, run_paths)
        return run_paths, state["hist"]

    _, off = run(f"joint-causal-off-{mode}", [])
    run_paths, on = run(
        f"joint-causal-on-{mode}",
        [
            "training.causal_weighting=true",
            f"training.causal_mode={mode}",
            "training.causal_time_chunks=4",
            "training.causal_eps_schedule=[1.0]",
        ],
    )

    assert len(on) == len(off) == 3
    for record in on:
        for name, value in record.items():
            assert math.isfinite(value), f"causal joint {name} was not finite"
    assert _trajectories_differ(on, off), "causal weighting should change the joint trajectory"
    assert run_paths.checkpoint.exists()


def test_single_dataset_evaluate_reports_a_validation_split_iou(tmp_path):
    """A single-series run with a validation split surfaces its in-distribution
    IoU (over the held-out tail frames), rather than silently dropping the metric."""
    import json

    from naviernet.evaluation import evaluate
    from naviernet.training import load_model, train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            "dataset=solo",
            *_TINY_TRAIN,
            "training.steps=2",
            "training.holdout_frame=-1",
            "training.val_fraction=0.25",
            "training.val_strategy=tail",
        ]
    )
    run_paths = RunPaths.from_config(cfg)
    _stage(run_paths)  # 8 event frames

    train(cfg, run_paths)
    model, data, _ = load_model(cfg, run_paths)
    report = evaluate(cfg, model, data, run_paths)

    assert report["validation_frames"] == [7, 8]  # tail 0.25 of 8 frames
    assert 0.0 <= report["iou_val"] <= 1.0
    on_disk = json.loads(run_paths.metrics_json.read_text())
    assert on_disk["iou_val"] == report["iou_val"]


def test_joint_evaluation_reports_per_dataset_iou(tmp_path):
    """A joint run evaluates each dataset with its own conditioning and writes one
    metrics.json carrying per-dataset IoU plus an aggregate."""
    import json

    from naviernet.evaluation import evaluate_joint
    from naviernet.physics.groups import compute_groups
    from naviernet.training import load_joint, train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            "datasets=[ds_a,ds_b]",
            "run_name=joint",
            *_TINY_TRAIN,
            "training.steps=2",
        ]
    )
    run_paths = RunPaths.from_config(cfg)
    for name, q_wall in (("ds_a", 2.0), ("ds_b", 5.0)):
        ds_paths = run_paths.for_dataset(name)
        ds_paths.processed_dir.mkdir(parents=True, exist_ok=True)
        groups = compute_groups(make_config([f"experiment.q_wall_W_cm2={q_wall}"]))
        _write_tensors(ds_paths.tensors, list(range(1, 9)), n_event=8, groups=groups)

    train(cfg, run_paths)
    model, contexts, heldout = load_joint(cfg, run_paths)
    report = evaluate_joint(cfg, model, contexts, run_paths, heldout_datasets=heldout)

    assert set(report["per_dataset"]) == {"ds_a", "ds_b"}
    for name in ("ds_a", "ds_b"):
        assert 0.0 <= report["per_dataset"][name]["iou_mean"] <= 1.0
    assert 0.0 <= report["iou_mean"] <= 1.0  # aggregate over the datasets
    on_disk = json.loads(run_paths.metrics_json.read_text())
    assert on_disk["datasets"] == ["ds_a", "ds_b"]


def _write_joint_datasets(run_paths, specs):
    """Write synthetic tensors for several datasets, each with distinct groups."""
    from naviernet.physics.groups import compute_groups

    for name, q_wall in specs:
        ds_paths = run_paths.for_dataset(name)
        ds_paths.processed_dir.mkdir(parents=True, exist_ok=True)
        groups = compute_groups(make_config([f"experiment.q_wall_W_cm2={q_wall}"]))
        _write_tensors(ds_paths.tensors, list(range(1, 9)), n_event=8, groups=groups)


def test_joint_checkpoint_records_the_training_and_held_out_split(tmp_path):
    """A run with a held-out condition records which datasets trained and which
    were kept out, so evaluation can score transfer separately."""
    from naviernet.training import train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            "datasets=[ds_a,ds_b,ds_c]",
            "heldout_datasets=[ds_c]",
            "run_name=joint",
            *_TINY_TRAIN,
            "training.steps=2",
        ]
    )
    run_paths = RunPaths.from_config(cfg)
    _write_joint_datasets(run_paths, (("ds_a", 2.0), ("ds_b", 5.0), ("ds_c", 8.0)))

    train(cfg, run_paths)

    saved = torch.load(run_paths.checkpoint, map_location="cpu", weights_only=False)
    assert saved["datasets"] == ["ds_a", "ds_b", "ds_c"]
    assert saved["training_datasets"] == ["ds_a", "ds_b"]
    assert saved["heldout_datasets"] == ["ds_c"]


def test_held_out_condition_never_enters_the_loss(tmp_path):
    """Training on [a, b, c] with c held out is byte-for-byte identical to training
    on [a, b] -- proof the held-out dataset contributes nothing to supervision."""
    from naviernet.training import train

    def train_model(root, datasets, heldout):
        overrides = [
            f"paths.root={root}",
            f"datasets=[{','.join(datasets)}]",
            "run_name=joint",
            *_TINY_TRAIN,
            "training.steps=3",
        ]
        if heldout:
            overrides.append(f"heldout_datasets=[{','.join(heldout)}]")
        cfg = make_config(overrides)
        run_paths = RunPaths.from_config(cfg)
        _write_joint_datasets(run_paths, (("ds_a", 2.0), ("ds_b", 5.0), ("ds_c", 8.0)))
        model, _, _ = train(cfg, run_paths)
        return model

    withheld = train_model(tmp_path / "withheld", ["ds_a", "ds_b", "ds_c"], ["ds_c"])
    pair = train_model(tmp_path / "pair", ["ds_a", "ds_b"], [])

    for (name, a), (_, b) in zip(
        withheld.state_dict().items(), pair.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), f"{name} differs -- the held-out dataset leaked into training"


def test_joint_run_with_one_training_dataset_still_checkpoints(tmp_path):
    """Holding out all but one dataset leaves a single training condition; the
    conditioned joint path still runs and writes a checkpoint."""
    from naviernet.physics.groups import N_COND
    from naviernet.training import train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            "datasets=[ds_a,ds_b]",
            "heldout_datasets=[ds_b]",
            "run_name=joint",
            *_TINY_TRAIN,
            "training.steps=2",
        ]
    )
    run_paths = RunPaths.from_config(cfg)
    _write_joint_datasets(run_paths, (("ds_a", 2.0), ("ds_b", 5.0)))

    model, _, state = train(cfg, run_paths)

    assert run_paths.checkpoint.exists()
    assert model.n_cond == N_COND
    saved = torch.load(run_paths.checkpoint, map_location="cpu", weights_only=False)
    assert saved["training_datasets"] == ["ds_a"]
    assert saved["heldout_datasets"] == ["ds_b"]


def test_joint_training_rejects_every_dataset_held_out(tmp_path):
    """Nothing left to train on -> fail loudly (the config guard fires)."""
    from naviernet.training import train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            "datasets=[ds_a,ds_b]",
            "heldout_datasets=[ds_a,ds_b]",
            "run_name=joint",
            *_TINY_TRAIN,
        ]
    )
    run_paths = RunPaths.from_config(cfg)

    with pytest.raises(ValueError, match="training"):
        train(cfg, run_paths)


def test_joint_metrics_report_validation_and_transfer(tmp_path):
    """metrics.json v2: training datasets carry an in-distribution val IoU; a
    held-out condition is scored over every frame as a separate transfer IoU."""
    import json

    from naviernet.evaluation import evaluate_joint
    from naviernet.training import load_joint, train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            "datasets=[ds_a,ds_b,ds_c]",
            "heldout_datasets=[ds_c]",
            "run_name=joint",
            *_TINY_TRAIN,
            "training.steps=2",
            "training.holdout_frame=-1",
            "training.val_fraction=0.25",
            "training.val_strategy=tail",
        ]
    )
    run_paths = RunPaths.from_config(cfg)
    _write_joint_datasets(run_paths, (("ds_a", 2.0), ("ds_b", 5.0), ("ds_c", 8.0)))

    train(cfg, run_paths)
    model, contexts, heldout = load_joint(cfg, run_paths)
    report = evaluate_joint(cfg, model, contexts, run_paths, heldout_datasets=heldout)

    assert report["datasets"] == ["ds_a", "ds_b", "ds_c"]
    assert report["training_datasets"] == ["ds_a", "ds_b"]
    assert report["heldout_datasets"] == ["ds_c"]
    # per_dataset covers the datasets that trained; each has an in-distribution val.
    assert set(report["per_dataset"]) == {"ds_a", "ds_b"}
    for name in ("ds_a", "ds_b"):
        d = report["per_dataset"][name]
        assert d["validation_frames"] == [7, 8]  # tail 0.25 of 8 frames
        assert 0.0 <= d["iou_val"] <= 1.0
    assert 0.0 <= report["val_iou_mean"] <= 1.0
    # transfer: the held-out condition, scored over ALL its frames.
    assert set(report["transfer"]["per_dataset"]) == {"ds_c"}
    assert 0.0 <= report["transfer"]["mean"] <= 1.0

    on_disk = json.loads(run_paths.metrics_json.read_text())
    assert (
        on_disk["transfer"]["per_dataset"]["ds_c"] == report["transfer"]["per_dataset"]["ds_c"]
    )


def test_joint_metrics_omit_transfer_when_nothing_is_held_out(tmp_path):
    from naviernet.evaluation import evaluate_joint
    from naviernet.training import load_joint, train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            "datasets=[ds_a,ds_b]",
            "run_name=joint",
            *_TINY_TRAIN,
            "training.steps=2",
        ]
    )
    run_paths = RunPaths.from_config(cfg)
    _write_joint_datasets(run_paths, (("ds_a", 2.0), ("ds_b", 5.0)))

    train(cfg, run_paths)
    model, contexts, heldout = load_joint(cfg, run_paths)
    report = evaluate_joint(cfg, model, contexts, run_paths, heldout_datasets=heldout)

    assert "transfer" not in report
    assert report["heldout_datasets"] == []
    assert set(report["per_dataset"]) == {"ds_a", "ds_b"}


def test_val_iou_folds_in_the_legacy_holdout_frame(tmp_path):
    """With no split fraction but the legacy holdout frame set, the in-distribution
    val IoU still reports that frame -- the metric is never silently dropped."""
    from naviernet.evaluation import evaluate_joint
    from naviernet.training import load_joint, train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            "datasets=[ds_a,ds_b]",
            "run_name=joint",
            *_TINY_TRAIN,
            "training.steps=2",
            "training.holdout_frame=5",  # camera frame 6
            "training.val_fraction=0.0",
        ]
    )
    run_paths = RunPaths.from_config(cfg)
    _write_joint_datasets(run_paths, (("ds_a", 2.0), ("ds_b", 5.0)))

    train(cfg, run_paths)
    model, contexts, heldout = load_joint(cfg, run_paths)
    report = evaluate_joint(cfg, model, contexts, run_paths, heldout_datasets=heldout)

    for name in ("ds_a", "ds_b"):
        d = report["per_dataset"][name]
        assert d["validation_frames"] == [6]  # only the legacy holdout frame
        assert d["iou_val"] == pytest.approx(d["iou_per_frame"][6])


def test_single_dataset_train_rejects_holding_out_its_only_dataset(tmp_path):
    """The split guard fires on the single-dataset path too (before any I/O), so a
    misconfigured CLI run fails loudly instead of training as if nothing was set."""
    from naviernet.training import train

    cfg = make_config(
        [f"paths.root={tmp_path}", "dataset=solo", "heldout_datasets=[solo]", *_TINY_TRAIN]
    )
    with pytest.raises(ValueError, match="hold out every dataset"):
        train(cfg, RunPaths.from_config(cfg))


def test_joint_evaluation_uses_the_checkpoints_split_not_the_current_config(tmp_path):
    """A standalone evaluate must classify datasets by how the model was trained
    (the checkpoint's held-out split), not by whatever cfg a re-run composes."""
    from naviernet.evaluation import evaluate_joint
    from naviernet.training import load_joint, train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            "datasets=[ds_a,ds_b,ds_c]",
            "heldout_datasets=[ds_c]",
            "run_name=joint",
            *_TINY_TRAIN,
            "training.steps=2",
        ]
    )
    run_paths = RunPaths.from_config(cfg)
    _write_joint_datasets(run_paths, (("ds_a", 2.0), ("ds_b", 5.0), ("ds_c", 8.0)))
    train(cfg, run_paths)

    # Re-compose as a bare `stage=evaluate` would: same run, but no heldout override.
    eval_cfg = make_config(
        [f"paths.root={tmp_path}", "datasets=[ds_a,ds_b,ds_c]", "run_name=joint", *_TINY_TRAIN]
    )
    model, contexts, heldout = load_joint(eval_cfg, run_paths)
    assert heldout == ["ds_c"], "held-out split comes from the checkpoint"
    report = evaluate_joint(eval_cfg, model, contexts, run_paths, heldout_datasets=heldout)

    # ds_c is still scored as transfer, not folded into the training set.
    assert report["heldout_datasets"] == ["ds_c"]
    assert set(report["transfer"]["per_dataset"]) == {"ds_c"}
    assert set(report["per_dataset"]) == {"ds_a", "ds_b"}


def test_joint_training_needs_groups_in_each_datasets_tensors(tmp_path):
    """A dataset preprocessed before groups were recorded can't join a conditioned
    run; the trainer says so instead of silently mis-conditioning."""
    from naviernet.training import train

    cfg = make_config(
        [f"paths.root={tmp_path}", "datasets=[ds_a,ds_b]", "run_name=joint", *_TINY_TRAIN]
    )
    run_paths = RunPaths.from_config(cfg)
    for name in ("ds_a", "ds_b"):
        ds_paths = run_paths.for_dataset(name)
        ds_paths.processed_dir.mkdir(parents=True, exist_ok=True)
        _write_tensors(ds_paths.tensors, list(range(1, 9)), n_event=8)  # no groups

    with pytest.raises(ValueError, match="groups"):
        train(cfg, run_paths)


def test_curriculum_ramps_evaporation_in_as_the_source_penalty_decays():
    """Soft -> hard: at the start the penalty is full and the closure off; by the
    end the closure is full and the penalty gone. Disabled (0) leaves both at 1."""
    from naviernet.training import _curriculum

    assert _curriculum(0, 100) == (1.0, 0.0)
    assert _curriculum(50, 100) == (0.5, 0.5)
    assert _curriculum(100, 100) == (0.0, 1.0)
    assert _curriculum(150, 100) == (0.0, 1.0), "clamped past the end"
    assert _curriculum(50, 0) == (1.0, 1.0), "0 disables the schedule"


def test_loss_schedule_gates_stage_b_physics_until_the_warmup_ends():
    """The Stage-B terms (momentum/energy/evaporation) stay at zero through the
    warm-up, then engage; the gate never touches the Stage-A objective. Absent
    keys keep their static weight (multiplier 1), so post-warm-up they are on."""
    from types import SimpleNamespace

    from naviernet.training import _loss_schedule

    keys = ("mom", "energy", "evap")
    tcfg = SimpleNamespace(curriculum_steps=0, stage_b_warmup_steps=100)

    during = _loss_schedule(50, tcfg, keys)
    assert during["mom"] == during["energy"] == during["evap"] == 0.0
    assert during["src"] == 1.0, "the Stage-A source penalty is not gated"

    at_boundary = _loss_schedule(100, tcfg, keys)
    assert at_boundary["mom"] == 0.0, "the warm-up step itself is still gated (<=)"

    after = _loss_schedule(101, tcfg, keys)
    assert "mom" not in after and "energy" not in after, "engaged -> default weight 1"
    assert after["evap"] == 1.0, "engaged, curriculum off -> full closure"

    no_warmup = _loss_schedule(
        1, SimpleNamespace(curriculum_steps=0, stage_b_warmup_steps=0), keys
    )
    assert "mom" not in no_warmup, "0 warm-up -> Stage-B physics on from step 1"


def test_causal_weights_enforce_temporal_ordering():
    """Causal weighting (Wang et al.): a time chunk is held down until earlier
    chunks are satisfied, so the model learns forward in time. eps=0 is uniform;
    weights decrease with time, depend only on EARLIER losses, and carry no grad."""
    import math

    from naviernet.training import _causal_weights

    losses = torch.tensor([1.0, 1.0, 1.0, 1.0])

    assert torch.allclose(_causal_weights(losses, 0.0), torch.ones(4)), "eps=0 -> uniform"

    w = _causal_weights(losses, 1.0)
    assert w[0].item() == pytest.approx(1.0), "first chunk has no prior -> full weight"
    assert torch.all(w[1:] < w[:-1]), "later chunks weighted down by earlier residuals"
    assert w[2].item() == pytest.approx(math.exp(-2.0)), "w_2 = exp(-(L0+L1)) = exp(-2)"
    assert not w.requires_grad, "weights steer training but carry no gradient"


def test_causal_eps_anneals_across_the_call_window_and_restarts_on_resume():
    """ε steps through the schedule as training progresses over the call's step
    window [first_step, last_step]: the first step uses the smallest ε (near-uniform
    in time), the last the largest, so causal enforcement steepens over the run. An
    empty schedule disables it (ε=0 -> uniform)."""
    from naviernet.training import _causal_eps

    schedule = [1e-2, 1e-1, 1.0]
    assert _causal_eps(1, 1, 300, schedule) == pytest.approx(1e-2), "first step -> smallest ε"
    assert _causal_eps(150, 1, 300, schedule) == pytest.approx(1e-1), "midpoint -> middle ε"
    assert _causal_eps(300, 1, 300, schedule) == pytest.approx(1.0), "last step -> largest ε"
    assert _causal_eps(1, 1, 300, []) == 0.0, "empty schedule -> ε=0 (uniform in time)"

    # Resume regression: a chunk starting at step 301 anneals from the start of the
    # schedule again, rather than dividing an absolute step by a per-call count and
    # snapping straight to the final ε for the whole resumed segment.
    assert _causal_eps(301, 301, 600, schedule) == pytest.approx(1e-2), (
        "resumed chunk restarts ε"
    )
    assert _causal_eps(600, 301, 600, schedule) == pytest.approx(1.0), (
        "resumed chunk reaches max ε"
    )


def test_time_chunks_partition_x_coll_by_ascending_time():
    """The binning is by the time column (index 2), ascending, into leaf tensors
    that carry gradient (so the PDE residuals differentiate through them), covering
    every point exactly once."""
    from naviernet.training import _time_chunks

    torch.manual_seed(0)
    x = torch.rand(20, 3, requires_grad=True)

    chunks = _time_chunks(x, 4)

    assert len(chunks) == 4
    assert sum(c.shape[0] for c in chunks) == 20, "every point lands in exactly one bin"
    for c in chunks:
        assert c.is_leaf and c.requires_grad, "bins are differentiable leaves"
    means = [float(c.detach()[:, 2].mean()) for c in chunks]
    assert means == sorted(means), "bins are ordered earliest -> latest in time"
    # Fewer points than chunks still yields non-empty bins (no empty-bin NaN).
    assert all(c.shape[0] > 0 for c in _time_chunks(x[:3], 8))


def test_causal_collocation_loss_reduces_to_the_uniform_mean_when_eps_is_zero():
    """The no-op guarantee the journey rests on: at ε=0 the causal weights are all
    one, so with equal-size bins the time-causal collocation loss equals the plain
    per-term weighted collocation mean over the same points -- and it still carries
    gradient back to the model parameters."""
    from naviernet.models.pinn import BubblePINN
    from naviernet.physics import registry
    from naviernet.physics.groups import compute_groups
    from naviernet.training import _causal_collocation_loss, _weighted_sum

    cfg = make_config(_TINY_TRAIN)
    torch.manual_seed(0)
    model = BubblePINN(cfg)
    groups = compute_groups(cfg)
    coll_equations = registry.collocation_equations(
        registry.enabled_equations(cfg.model.fields)
    )
    weights = {"vof": 2.0, "div": 0.5, "src": 3.0}  # distinct, so weights must carry through
    schedule: dict[str, float] = {}  # no warm-up gating: every multiplier is 1

    x_coll = torch.rand(16, 3, requires_grad=True)  # 16 points / 4 bins -> equal bins

    causal = _causal_collocation_loss(
        model, x_coll, groups, None, coll_equations, weights, schedule, n_chunks=4, eps=0.0
    )
    ctx = registry.LossContext(model, x_coll, groups=groups)
    uniform = _weighted_sum(
        {e.weight_key: e.term(ctx) for e in coll_equations}, weights, schedule
    )

    assert causal.item() == pytest.approx(uniform.item(), rel=1e-5), (
        "ε=0 -> uniform collocation mean"
    )
    assert causal.requires_grad, "gradient must still flow through the residual"
    causal.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()), (
        "the causal collocation loss trains the model"
    )


def test_causal_weighting_changes_the_stage_a_loss_trajectory(tmp_path):
    """Wiring check for Phase 1: turning causal weighting ON reweights the
    collocation residual by time, so the optimised loss trajectory diverges from
    the uniform-in-time run under an identical seed and sampling -- and the run
    still trains to finite losses (no crash from the per-bin residual passes)."""
    import math

    off = _stage_a_hist(tmp_path, "causal-off", [])
    on = _stage_a_hist(
        tmp_path,
        "causal-on",
        [
            "training.causal_weighting=true",
            "training.causal_time_chunks=4",
            "training.causal_eps_schedule=[1.0]",
        ],
    )

    assert len(on) == len(off) == 3
    for record in on:
        for name, value in record.items():
            assert math.isfinite(value), f"causal {name} was not finite"
    # Same seed, same samples: only the temporal reweighting differs, so the
    # trajectories must diverge on the collocation terms.
    assert _trajectories_differ(on, off), "causal weighting should change the trajectory"


def test_march_active_bins_expands_monotonically_to_full_domain():
    """Time-marching horizon: covers only the earliest time bin at the start of the
    run and expands to the whole domain by ``full_frac`` of the way through, never
    shrinking -- so late times are trained at full weight once the front reaches
    them."""
    from naviernet.training import _march_active_bins

    n = 16
    assert _march_active_bins(0.0, n, 0.5) == 1, "start: only the earliest bin"
    assert _march_active_bins(1.0, n, 0.5) == n, "stays at the full domain afterwards"
    # full_frac IS honoured (not a hardcoded 0.5): a smaller frac reaches the full
    # domain earlier. The `1 + int(...)` discretisation reaches full slightly *before*
    # the nominal frac, so compare across fracs rather than pinning the exact step.
    assert _march_active_bins(0.25, n, 0.25) == n, "smaller full_frac -> full sooner"
    assert _march_active_bins(0.25, n, 0.75) < n, "larger full_frac still expanding at 0.25"
    swept = [_march_active_bins(i / 20, n, 0.5) for i in range(21)]
    assert swept == sorted(swept), "the horizon never shrinks"


def test_time_march_collocation_loss_restricts_to_the_active_time_horizon():
    """Marching's distinguishing behaviour: at the start only the earliest time bin's
    residual enters the loss; once the horizon reaches full_frac it is the whole
    collocation mean. This pins WHICH points are active (earliest-first), which a
    trajectory-only test cannot."""
    from naviernet.models.pinn import BubblePINN
    from naviernet.physics import registry
    from naviernet.physics.groups import compute_groups
    from naviernet.training import _time_chunks, _time_march_collocation_loss, _weighted_sum

    cfg = make_config(_TINY_TRAIN)
    torch.manual_seed(0)
    model = BubblePINN(cfg)
    groups = compute_groups(cfg)
    coll_equations = registry.collocation_equations(
        registry.enabled_equations(cfg.model.fields)
    )
    weights = {"vof": 2.0, "div": 0.5, "src": 3.0}
    schedule: dict[str, float] = {}
    x_coll = torch.rand(16, 3, requires_grad=True)

    def march(progress: float) -> float:
        return _time_march_collocation_loss(
            model, x_coll, groups, None, coll_equations, weights, schedule, 4, progress, 0.5
        ).item()

    def uniform_over(points) -> float:
        ctx = registry.LossContext(model, points, groups=groups)
        return _weighted_sum(
            {e.weight_key: e.term(ctx) for e in coll_equations}, weights, schedule
        ).item()

    earliest_bin = _time_chunks(x_coll, 4)[0]
    assert march(0.0) == pytest.approx(uniform_over(earliest_bin), rel=1e-5), (
        "progress 0 -> only the earliest time bin"
    )
    assert march(1.0) == pytest.approx(uniform_over(x_coll), rel=1e-5), (
        "progress >= full_frac -> the whole domain"
    )
    assert march(0.0) != pytest.approx(march(1.0), rel=1e-3), "the horizon actually restricts"


def test_time_marching_and_soft_weighting_are_distinct_objectives(tmp_path):
    """march and weight must be different objectives, not a silent dispatch
    fall-through: under a shared seed their trajectories diverge from EACH OTHER (both
    also differ from the uniform run, so comparing only to 'off' would not catch a
    mode mix-up)."""
    import math

    weight = _stage_a_hist(
        tmp_path,
        "cmp-weight",
        [
            "training.causal_weighting=true",
            "training.causal_mode=weight",
            "training.causal_time_chunks=4",
            "training.causal_eps_schedule=[1.0]",
        ],
    )
    march = _stage_a_hist(
        tmp_path,
        "cmp-march",
        [
            "training.causal_weighting=true",
            "training.causal_mode=march",
            "training.causal_time_chunks=4",
        ],
    )

    for record in march:
        for name, value in record.items():
            assert math.isfinite(value), f"march {name} was not finite"
    assert _trajectories_differ(weight, march), "march and weight are distinct objectives"


def test_unknown_causal_mode_fails_loudly(tmp_path):
    """An unrecognised causal_mode is a config error, not a silent fall-through to
    one variant -- it must raise so a typo can't quietly train the wrong objective.
    (Validation is the first line of train(), before any dataset I/O, so no staging.)"""
    from naviernet.training import train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            *_TINY_TRAIN,
            "training.steps=2",
            "training.causal_weighting=true",
            "training.causal_mode=bogus",
        ]
    )

    with pytest.raises(ValueError, match="causal_mode"):
        train(cfg, RunPaths.from_config(cfg))


def test_out_of_range_march_full_frac_fails_loudly(tmp_path):
    """causal_march_full_frac<=0 would collapse the horizon to the full domain on the
    first step, silently defeating the curriculum -- so it must raise, like an unknown
    causal_mode does, rather than degrade quietly."""
    from naviernet.training import train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            *_TINY_TRAIN,
            "training.steps=2",
            "training.causal_weighting=true",
            "training.causal_mode=march",
            "training.causal_march_full_frac=0.0",
        ]
    )

    with pytest.raises(ValueError, match="causal_march_full_frac"):
        train(cfg, RunPaths.from_config(cfg))


def test_stage_b_engages_only_at_the_boundary_and_never_when_disabled():
    """The optimiser restart fires at exactly step warmup+1 -- and never when the
    warm-up is disabled (0) or the model has no Stage-B physics, so an ordinary
    Stage-B run is not silently restarted on its first step."""
    from types import SimpleNamespace

    from naviernet.training import _stage_b_engages_at

    keys = ("mom", "energy", "evap")
    tcfg = SimpleNamespace(stage_b_warmup_steps=100)
    assert _stage_b_engages_at(101, tcfg, keys) is True
    assert _stage_b_engages_at(100, tcfg, keys) is False
    assert _stage_b_engages_at(102, tcfg, keys) is False
    # 0 disables the warm-up: it must NOT fire at step 1 (warmup + 1).
    assert _stage_b_engages_at(1, SimpleNamespace(stage_b_warmup_steps=0), keys) is False
    # Nothing to engage on a Stage-A model.
    assert _stage_b_engages_at(101, tcfg, ()) is False


def test_curriculum_schedule_targets_real_weight_keys():
    """The trainer applies the schedule by the names 'src' and 'evap'; guard those
    strings against drifting from the registry's weight keys (else it silently
    no-ops via schedule.get(name, 1.0))."""
    from naviernet.physics.registry import REGISTRY

    keys = {e.weight_key for e in REGISTRY}
    assert {"src", "evap"} <= keys


_TINY_TRAIN = [
    "model.hidden=8",
    "model.layers=2",
    "model.fourier_feats=4",
    "training.n_data=16",
    "training.n_coll=16",
    "training.n_bc=8",
]
_TINY_STAGE_B = [
    "model=stage_b",
    "training=stage_b",
    "model.per_field.p.hidden=8",
    "model.per_field.p.layers=2",
    "model.per_field.T.hidden=8",
    "model.per_field.T.layers=2",
]


def _stage(paths):
    paths.ensure()
    paths.tensors.parent.mkdir(parents=True, exist_ok=True)
    _write_tensors(paths.tensors, list(range(1, 9)), n_event=8)


def _stage_joint_datasets(run_paths, specs=(("ds_a", 2.0), ("ds_b", 5.0))):
    """Stage synthetic datasets for a joint run, each carrying its own groups (a
    distinct wall heat flux -> distinct regime), as preprocess would have written."""
    from naviernet.physics.groups import compute_groups

    for name, q_wall in specs:
        ds_paths = run_paths.for_dataset(name)
        ds_paths.processed_dir.mkdir(parents=True, exist_ok=True)
        groups = compute_groups(make_config([f"experiment.q_wall_W_cm2={q_wall}"]))
        _write_tensors(ds_paths.tensors, list(range(1, 9)), n_event=8, groups=groups)


def _stage_a_hist(tmp_path, run_name: str, extra: list[str]) -> list[dict]:
    """Train a tiny single-dataset Stage-A run (fixed seed, no Stage-B warm-up so the
    optimiser restart never fires) and return its loss history, for comparing loss
    trajectories under the one variable a test toggles via ``extra``."""
    from naviernet.training import train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            f"run_name={run_name}",
            *_TINY_TRAIN,
            "training.steps=3",
            "training.log_every=1",
            "training.stage_b_warmup_steps=0",
            "training.seed=0",
            *extra,
        ]
    )
    paths = RunPaths.from_config(cfg)
    _stage(paths)
    _, _, state = train(cfg, paths)
    return state["hist"]


def _trajectories_differ(a: list[dict], b: list[dict], keys=("vof", "div")) -> bool:
    """True if two loss histories diverge on any collocation term at any logged step.
    Under a shared seed, step 1's recorded means match (identical initial model); the
    divergence shows once the differing gradient has stepped the model."""
    return any(
        a_rec[k] != pytest.approx(b_rec[k], rel=1e-9, abs=1e-12)
        for a_rec, b_rec in zip(a, b, strict=True)
        for k in keys
    )


def test_stage_b_smoke_run_trains_pressure_and_temperature(tmp_path):
    """A tiny Stage-B run builds the p/T networks, activates momentum + energy +
    evaporation, and takes real steps without producing a NaN."""
    import math

    from naviernet.training import train

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            *_TINY_STAGE_B,
            *_TINY_TRAIN,
            "training.steps=2",
            "training.rebalance_every=2",
            "training.curriculum_steps=2",
        ]
    )
    paths = RunPaths.from_config(cfg)
    _stage(paths)

    model, _, state = train(cfg, paths)

    assert {"p", "T"} <= set(model.fields)
    assert {"mom", "energy", "evap"} <= set(state["hist"][-1])
    for record in state["hist"]:
        for name, value in record.items():
            assert math.isfinite(value), f"{name} was not finite"
    assert paths.checkpoint.exists()


def _spy_on_adam(monkeypatch) -> list:
    """Record the lr of every optimiser built during training. The in-run warm-up
    builds a second one when Stage-B physics engages; this proves it fired."""
    import torch

    built: list = []
    real_adam = torch.optim.Adam

    def _spy(*args, **kwargs):
        built.append(kwargs.get("lr"))
        return real_adam(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", _spy)
    return built


def test_stage_b_warmup_restarts_the_optimiser_when_physics_engages(
    tmp_path, caplog, monkeypatch
):
    """The in-run warm-up gates Stage-B physics off, then engages it on a FRESH
    optimiser exactly at step warmup+1. Carrying Adam's Stage-A state across the
    switch collapses the interface, so the restart is the crux of the feature."""
    import logging

    from naviernet.training import train

    built = _spy_on_adam(monkeypatch)
    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            *_TINY_STAGE_B,
            *_TINY_TRAIN,
            "training.steps=4",
            "training.stage_b_warmup_steps=2",
            "training.curriculum_steps=0",
        ]
    )
    paths = RunPaths.from_config(cfg)
    _stage(paths)

    with caplog.at_level(logging.INFO, logger="naviernet.training"):
        train(cfg, paths)

    assert len(built) == 2, "one optimiser at init, one more when physics engages"
    assert "Stage-B physics engaged" in caplog.text


def test_joint_stage_b_warmup_restarts_the_optimiser_at_the_boundary(
    tmp_path, caplog, monkeypatch
):
    """The joint (transfer-learning) path gates and restarts the optimiser exactly
    like the single-dataset path -- the same warm-up, over a conditioned model."""
    import logging

    from naviernet.physics.groups import compute_groups
    from naviernet.training import train

    built = _spy_on_adam(monkeypatch)
    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            "datasets=[ds_a,ds_b]",
            "run_name=joint",
            *_TINY_STAGE_B,
            *_TINY_TRAIN,
            "training.steps=4",
            "training.stage_b_warmup_steps=2",
            "training.curriculum_steps=0",
        ]
    )
    run_paths = RunPaths.from_config(cfg)
    for name, q_wall in (("ds_a", 2.0), ("ds_b", 5.0)):
        ds_paths = run_paths.for_dataset(name)
        ds_paths.processed_dir.mkdir(parents=True, exist_ok=True)
        groups = compute_groups(make_config([f"experiment.q_wall_W_cm2={q_wall}"]))
        _write_tensors(ds_paths.tensors, list(range(1, 9)), n_event=8, groups=groups)

    with caplog.at_level(logging.INFO, logger="naviernet.training"):
        train(cfg, run_paths)

    assert len(built) == 2, "one optimiser at init, one more when physics engages"
    assert "Stage-B physics engaged" in caplog.text


def test_stage_b_warm_starts_from_a_stage_a_checkpoint(tmp_path, caplog):
    """A Stage-A checkpoint seeds a Stage-B run: phi/u/v/s load, p/T and the
    evaporation unknowns initialise fresh, and the step count carries forward."""
    import logging

    from naviernet.training import train

    cfg_a = make_config([f"paths.root={tmp_path}", "training.steps=2", *_TINY_TRAIN])
    paths = RunPaths.from_config(cfg_a)
    _stage(paths)
    _, _, state_a = train(cfg_a, paths)
    assert state_a["done"] == 2 and "p" not in state_a["w"]

    cfg_b = make_config(
        [
            f"paths.root={tmp_path}",
            *_TINY_STAGE_B,
            *_TINY_TRAIN,
            "training.steps=2",
            "training.curriculum_steps=2",
        ]
    )
    with caplog.at_level(logging.INFO, logger="naviernet.training"):
        model, _, state_b = train(cfg_b, RunPaths.from_config(cfg_b))

    # The warm-start branch ran: it loaded phi/u/v/s (strict=False with only the
    # new p/T/evaporation params missing) rather than starting from scratch. The
    # unexpected-key guard would have raised had the checkpoint not matched.
    assert "warm start" in caplog.text
    assert {"p", "T"} <= set(model.fields)
    assert state_b["done"] == 4, "warm start carries the Stage-A step count forward"
    assert {"mom", "energy", "evap"} <= set(state_b["w"])
    # The data loss at the first Stage-B step continues from where Stage A left
    # off (phi loaded), not a fresh-random-net value.
    assert state_b["hist"][0]["data"] == pytest.approx(state_a["hist"][-1]["data"], rel=0.5)


def test_unknown_stage_is_rejected(tiny_cfg):
    with pytest.raises(ValueError, match="unknown stage"):
        Pipeline(tiny_cfg).run("trian")


def test_missing_tensors_give_an_actionable_error(tiny_cfg):
    """Asking to evaluate before preprocessing should say what to run."""
    with pytest.raises(FileNotFoundError, match="stage=train"):
        Pipeline(tiny_cfg).evaluate()


def test_pipeline_creates_the_run_directories(tiny_cfg):
    paths = RunPaths.from_config(tiny_cfg)
    Pipeline(tiny_cfg)
    assert paths.output_dir.is_dir() and paths.figures_dir.is_dir()


# --- frame exclusion --------------------------------------------------------


def test_usable_frames_are_the_window_when_nothing_is_excluded(cfg):
    assert usable_frame_numbers(cfg) == list(range(1, cfg.experiment.n_frames_usable + 1))


def test_excluded_frames_are_dropped_from_the_usable_window():
    cfg = make_config(["experiment.excluded_frames=[3,7]"])

    kept = usable_frame_numbers(cfg)

    assert 3 not in kept and 7 not in kept
    assert kept == [n for n in range(1, cfg.experiment.n_frames_usable + 1) if n not in (3, 7)]


def test_excluding_frames_outside_the_window_changes_nothing(cfg):
    """Frames past `n_frames_usable` were never fed to the model to begin with."""
    beyond = cfg.experiment.n_frames_usable + 1
    excluded = make_config([f"experiment.excluded_frames=[{beyond}]"])

    assert usable_frame_numbers(excluded) == usable_frame_numbers(cfg)


def test_excluding_almost_everything_is_rejected(cfg):
    survivors = list(range(1, MIN_USABLE_FRAMES))  # one short of the floor
    excluded = [n for n in range(1, cfg.experiment.n_frames_usable + 1) if n not in survivors]
    too_few = make_config([f"experiment.excluded_frames=[{','.join(map(str, excluded))}]"])

    with pytest.raises(ValueError, match="at least"):
        usable_frame_numbers(too_few)


def _write_tensors(path, frame_numbers: list[int], n_event: int, groups=None) -> None:
    """A minimal archive: only the keys BubbleDataset reads on construction.

    ``groups`` (when given) is recorded in the meta, as preprocess does, so a
    dataset can join a conditioned multi-dataset run.
    """
    import json

    import numpy as np

    n_t = len(frame_numbers)
    alpha = np.zeros((n_t, 4, 5), dtype=np.float32)
    alpha[:, 1:3, 1:3] = 1.0
    meta = {
        "x_pin_star": 0.5,
        "t_ref_ms": 1.5,
        "n_frames_usable": n_t,
        "n_frames_event": n_event,
        "frame_numbers": frame_numbers,
    }
    if groups is not None:
        meta["groups"] = groups
    np.savez_compressed(
        path,
        alpha=alpha,
        sdf=((0.5 - alpha) * 0.1).astype(np.float32),
        valid=np.ones_like(alpha),
        masks_camera=(alpha > 0.5).astype(np.uint8),
        x_star=np.linspace(0, 1, 5, dtype=np.float32),
        y_star=np.linspace(0, 1, 4, dtype=np.float32),
        t_star=((np.asarray(frame_numbers) - 1) * 0.1).astype(np.float32),
        meta=json.dumps(meta),
    )


def test_holdout_resolves_to_a_row_when_earlier_frames_are_excluded(tiny_cfg):
    """Camera frame 6 sits at row 4 once frame 3 is gone — supervising row 5
    would train on the holdout while still reporting it as held out."""
    from naviernet.data.dataset import BubbleDataset

    paths = RunPaths.from_config(tiny_cfg)
    paths.ensure()
    paths.tensors.parent.mkdir(parents=True, exist_ok=True)
    kept = [1, 2, 4, 5, 6, 7, 8]
    _write_tensors(paths.tensors, kept, n_event=len(kept))

    data = BubbleDataset(tiny_cfg, paths)  # tiny_cfg holds the default holdout

    assert tiny_cfg.training.holdout_frame == 5, "fixture assumes camera frame 6"
    assert data.frame_numbers == kept
    assert data.holdout_row == kept.index(6) == 4
    assert data.holdout_row not in data._ti[data._train_idx], "the holdout row is supervised"


def test_an_excluded_holdout_leaves_no_holdout_row(tiny_cfg):
    from naviernet.data.dataset import BubbleDataset

    paths = RunPaths.from_config(tiny_cfg)
    paths.ensure()
    paths.tensors.parent.mkdir(parents=True, exist_ok=True)
    _write_tensors(paths.tensors, [1, 2, 3, 4, 5, 7, 8], n_event=7)

    data = BubbleDataset(tiny_cfg, paths)

    assert data.holdout_row == -1
    assert len(data._train_idx) > 0, "every row is trainable when none is held out"


def test_event_frames_are_camera_numbers_not_row_indices(tiny_cfg):
    from naviernet.data.dataset import BubbleDataset

    paths = RunPaths.from_config(tiny_cfg)
    paths.ensure()
    paths.tensors.parent.mkdir(parents=True, exist_ok=True)
    _write_tensors(paths.tensors, [1, 2, 4, 5, 6, 7, 11], n_event=6)

    data = BubbleDataset(tiny_cfg, paths)

    assert data.n_event == 6
    assert data.event_frames == [1, 2, 4, 5, 6, 7]


def _dataset(tmp_path, frame_numbers, n_event, overrides=None):
    """A BubbleDataset over a synthetic archive, composed like the CLI would."""
    from naviernet.data.dataset import BubbleDataset

    cfg = make_config([f"paths.root={tmp_path}", *(overrides or [])])
    paths = RunPaths.from_config(cfg)
    paths.ensure()
    paths.tensors.parent.mkdir(parents=True, exist_ok=True)
    _write_tensors(paths.tensors, frame_numbers, n_event=n_event)
    return BubbleDataset(cfg, paths)


def test_no_validation_split_holds_out_only_the_legacy_frame(tmp_path):
    # val_fraction=0 (the default) -> axis A is inert; behaviour is unchanged.
    data = _dataset(tmp_path, [1, 2, 3, 4, 5, 6, 7, 8], 8, ["training.val_fraction=0.0"])

    assert data.split_rows == []
    assert data.split_frames == []


def test_tail_validation_holds_out_the_last_event_frames(tmp_path):
    # 0.25 of 8 event frames -> ceil = 2 held from the tail (extrapolation).
    data = _dataset(
        tmp_path,
        [1, 2, 3, 4, 5, 6, 7, 8],
        8,
        [
            "training.holdout_frame=-1",
            "training.val_fraction=0.25",
            "training.val_strategy=tail",
        ],
    )

    assert data.split_rows == [6, 7]
    assert data.split_frames == [7, 8]
    for row in data.split_rows:
        assert row not in data._ti[data._train_idx], "a val row was supervised"


def test_scatter_validation_spreads_across_the_event(tmp_path):
    # Same count, but interior evenly-spaced frames (interpolation), not the tail.
    data = _dataset(
        tmp_path,
        [1, 2, 3, 4, 5, 6, 7, 8],
        8,
        [
            "training.holdout_frame=-1",
            "training.val_fraction=0.25",
            "training.val_strategy=scatter",
        ],
    )

    assert len(data.split_rows) == 2
    assert data.split_rows != [6, 7], "scatter must not collapse to the tail"
    assert max(data.split_rows) < 7, "scatter holds interior frames, not the last"


def test_validation_split_operates_on_kept_frames_after_exclusions(tmp_path):
    # Excluded frames are already gone from the archive, so the tail split is
    # taken over the frames that remain -- rows, not raw camera indices.
    data = _dataset(
        tmp_path,
        [1, 2, 4, 5, 6, 7, 8],  # camera frame 3 excluded at preprocess
        7,
        [
            "training.holdout_frame=-1",
            "training.val_fraction=0.3",
            "training.val_strategy=tail",
        ],
    )

    # ceil(0.3 * 7) = 3 tail rows -> camera frames 6, 7, 8.
    assert data.split_frames == [6, 7, 8]


def test_validation_split_composes_with_the_legacy_holdout_frame(tmp_path):
    # holdout_frame (camera 6) AND the tail split are both held out -- a union.
    data = _dataset(
        tmp_path,
        [1, 2, 3, 4, 5, 6, 7, 8],
        8,
        [
            "training.holdout_frame=5",
            "training.val_fraction=0.25",
            "training.val_strategy=tail",
        ],
    )

    held = set(data.split_rows) | {data.holdout_row}
    assert data.holdout_row == 5  # camera frame 6
    assert data.split_rows == [6, 7]
    for row in held:
        assert row not in data._ti[data._train_idx]


def test_validation_split_leaves_at_least_one_training_frame(tmp_path):
    # An aggressive fraction still cannot hold out the whole event.
    data = _dataset(
        tmp_path,
        [1, 2, 3, 4],
        4,
        [
            "training.holdout_frame=-1",
            "training.val_fraction=0.9",
            "training.val_strategy=tail",
        ],
    )

    assert len(data.split_rows) == 3, "capped so one training frame survives"
    event_rows = {r for r in data._ti[data._train_idx] if r < data.n_event}
    assert event_rows, "at least one event frame is still supervised"


def test_scatter_validation_rows_are_never_supervised(tmp_path):
    # The interpolation split must also keep its frames out of the training index.
    data = _dataset(
        tmp_path,
        [1, 2, 3, 4, 5, 6, 7, 8],
        8,
        [
            "training.holdout_frame=-1",
            "training.val_fraction=0.4",
            "training.val_strategy=scatter",
        ],
    )

    assert len(data.split_rows) >= 1
    for row in data.split_rows:
        assert row not in data._ti[data._train_idx], "a scatter val row was supervised"


def test_scatter_never_holds_an_endpoint_or_collapses(tmp_path):
    # Even an aggressive fraction keeps both endpoints trained and returns distinct
    # interior rows (no silent collision), so it stays a genuine interpolation test.
    data = _dataset(
        tmp_path,
        [1, 2, 3, 4, 5, 6, 7, 8],
        8,
        [
            "training.holdout_frame=-1",
            "training.val_fraction=0.9",
            "training.val_strategy=scatter",
        ],
    )

    assert len(data.split_rows) == len(set(data.split_rows)), "rows collapsed onto each other"
    assert data.split_rows == sorted(data.split_rows)
    assert 0 not in data.split_rows and (data.n_event - 1) not in data.split_rows
    assert len(data.split_rows) <= data.n_event - 2  # only interior rows are eligible


def test_scatter_on_a_two_frame_event_holds_nothing(tmp_path):
    # No interior frame exists, so scatter cannot hold one out without training on
    # an endpoint -- it holds nothing rather than break its interpolation contract.
    data = _dataset(
        tmp_path,
        [1, 2],
        2,
        [
            "training.holdout_frame=-1",
            "training.val_fraction=0.5",
            "training.val_strategy=scatter",
        ],
    )

    assert data.split_rows == []


def test_tiny_split_floors_to_one_validation_frame(tmp_path):
    # A fraction that would round below one frame is floored up to a single frame.
    data = _dataset(
        tmp_path,
        [1, 2, 3, 4, 5, 6, 7, 8],
        8,
        [
            "training.holdout_frame=-1",
            "training.val_fraction=0.05",
            "training.val_strategy=tail",
        ],
    )

    assert len(data.split_rows) == 1


# --- tests below need the real dataset -------------------------------------


@pytest.mark.needs_data
def test_dataset_shapes_agree(trained):
    _, data = trained
    n_frames, height, width = data.shape

    assert data.sdf.shape == data.alpha.shape == data.valid.shape
    assert len(data.x) == width and len(data.y) == height and len(data.t) == n_frames


@pytest.mark.needs_data
def test_holdout_frame_is_never_sampled(trained, cfg):
    """The generalisation claim rests on this: no supervision from that frame."""
    import numpy as np

    _, data = trained
    rng = np.random.default_rng(0)
    _, _ = data.sample_supervised(4096, rng)

    # The row is resolved from the archive, not assumed to be the config index:
    # excluding an earlier frame shifts it, and supervising the shifted row
    # would train on the very frame the headline IoU calls unseen.
    assert data.frame_numbers[data.holdout_row] == cfg.training.holdout_frame + 1
    assert data.holdout_row not in data._ti[data._train_idx]


@pytest.mark.needs_data
def test_signed_distance_is_negative_inside_the_bubble(trained):
    _, data = trained
    inside = data.alpha > 0.5

    assert data.sdf[inside].max() <= 0.0
    assert data.sdf[~inside].min() >= 0.0


@pytest.mark.needs_data
def test_sampled_points_lie_inside_the_domain(trained):
    import numpy as np

    _, data = trained
    d = data.domain
    points = data.sample_supervised(512, np.random.default_rng(1))[0]

    assert points[:, 0].min() >= d.x_min and points[:, 0].max() <= d.x_max
    assert points[:, 1].min() >= d.y_min and points[:, 1].max() <= d.y_max


@pytest.mark.needs_data
@pytest.mark.slow
def test_holdout_iou_meets_the_published_figure(trained, cfg):
    """The headline result: >0.95 IoU on a frame the model never saw."""
    from naviernet.evaluation import frame_iou

    model, data = trained
    iou = frame_iou(cfg, model, data, data.holdout_row)
    assert iou > 0.95


@pytest.mark.needs_data
@pytest.mark.slow
def test_inferred_nose_speed_matches_the_measurement(trained, cfg):
    """Inferred within 10% of the measured 180 mm/s, with no velocity supervision."""
    from naviernet.evaluation import nose_trajectory

    model, data = trained
    times, nose, _ = nose_trajectory(cfg, model, data)
    speed = torch.tensor(nose).diff() / torch.tensor(times).diff()
    middle = slice(len(times) // 5, 4 * len(times) // 5)
    mm_s = float(speed[middle].mean()) * cfg.scales.U_ref * 1e3

    assert mm_s == pytest.approx(180.0, rel=0.10)


@pytest.mark.needs_data
@pytest.mark.slow
def test_multirun_seed_sweep_produces_independent_runs(cfg, paths, tmp_path):
    """Regression: each --multirun job must own its output directory.

    Previously every sweep job shared ``outputs/<run_name>``, so the second
    seed silently resumed the first job's checkpoint instead of training a fresh
    model. The CLI now binds the output directory to Hydra's per-job runtime
    directory; this drives a real two-seed sweep and checks the jobs stay
    independent.
    """
    import shutil
    import subprocess
    import sys

    # Train reads only the preprocessed tensors, so stage those under a
    # throwaway root -- no raw frames needed.
    processed = tmp_path / "data" / "processed" / cfg.dataset
    processed.mkdir(parents=True)
    shutil.copy(paths.tensors, processed / "tensors.npz")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "naviernet.cli",
            "--multirun",
            "stage=train",
            "training.steps=1",
            "training.log_every=1",
            f"paths.root={tmp_path}",
            "training.seed=0,1",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    checkpoints = sorted((tmp_path / "outputs" / "multirun").rglob("ckpt.pt"))
    assert len(checkpoints) == 2, f"expected two independent runs, got {checkpoints}"

    # A fresh single-step run reports done == 1. If the second job had resumed
    # the first's checkpoint it would report 2.
    for ckpt in checkpoints:
        state = torch.load(ckpt, weights_only=False)["state"]
        assert state["done"] == 1, f"{ckpt} completed {state['done']} steps (resumed?)"
