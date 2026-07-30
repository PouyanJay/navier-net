"""Extrapolation benchmark for the pinn-accuracy journey.

Trains one dataset with a validation split (tail frames held out of supervision)
and reports the per-frame IoU, so the accuracy phases (causal weighting, RBA,
adaptive collocation) can be measured against a fixed baseline. Truth is what
runs: this is the reproducible harness the journey verifies each phase with.

Usage (from the repo root, in the venv):
    .venv/bin/python scripts/bench_extrapolation.py --dataset Series-1 \
        --steps 3000 --rebalance-every 1500 [training.<key>=<val> ...]

Any extra `key=value` args are passed through as Hydra overrides, so a phase can
flip its flag on, e.g. `training.causal_weighting=true`. Recorded baseline
(Series-1, steps 3000, rebalance_every 1500, 10% tail split):
    frame9=0.879 frame10=0.758 frame11=0.60 mean=0.876 val_iou=0.679
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# The API's compose path applies the series' saved conditions + model config, so
# the bench trains exactly what a launched run would.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "apps" / "api" / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Series-1")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--rebalance-every", type=int, default=1500)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="bench-extrap")
    parser.add_argument("overrides", nargs="*", help="extra Hydra overrides, key=value")
    args = parser.parse_args()

    from naviernet_api.services.config_service import compose_cfg_once
    from naviernet_api.services.datasets import series_overrides
    from naviernet_api.settings import Settings

    from naviernet.pipeline import Pipeline

    settings = Settings(repo_root=pathlib.Path.cwd())
    overrides = [
        f"paths.root={settings.repo_root}",
        f"run_name={args.run_name}",
        f"training.steps={args.steps}",
        f"training.rebalance_every={args.rebalance_every}",
        f"training.val_fraction={args.val_fraction}",
        "training.val_strategy=tail",
        "training.holdout_frame=-1",
        f"training.stage_b_warmup_steps={args.steps // 2}",
        f"training.seed={args.seed}",
        "training.device=cpu",
        "training.log_every=1000",
        *series_overrides(settings, args.dataset),
        *args.overrides,
    ]
    cfg = compose_cfg_once(args.dataset, overrides=overrides)
    print(
        f"fields={list(cfg.model.fields)} steps={args.steps} "
        f"rebalance_every={args.rebalance_every} val_fraction={args.val_fraction}"
    )

    pipe = Pipeline(cfg)
    pipe.train(steps=args.steps, on_log=lambda r: None)
    metrics = pipe.evaluate()

    ipf = metrics.get("iou_per_frame", {})

    def frame(k: int) -> float:
        return round(ipf.get(k, ipf.get(str(k), 0.0)), 3)

    frames = sorted(int(k) for k in ipf)
    per_frame = " ".join(f"{k}:{frame(k)}" for k in frames)
    print(f"per-frame IoU: {per_frame}")
    print(
        f"RESULT mean={metrics.get('iou_mean'):.3f} "
        f"val_iou={metrics.get('iou_val'):.3f} "
        f"validation_frames={metrics.get('validation_frames')}"
    )
    _report_root_drift(cfg, pipe)


def _report_root_drift(cfg, pipe) -> None:
    """Per-frame predicted bubble-root position vs the dataset's measured anchor.

    The root drifting downstream on the held-out frames is the diagnosed
    extrapolation failure the hard pin targets; reporting it for every bench run
    (pinned or not) makes the mechanism visible next to the IoU numbers.
    """
    from naviernet.evaluation import predict_alpha, root_position

    model, data = pipe._load()
    x_anchor, _ = data.pin_anchor
    stride = cfg.evaluation.stride
    threshold = cfg.evaluation.threshold
    xs = data.x[::stride]

    drifts = []
    for row, frame_no in enumerate(data.frame_numbers):
        mask = predict_alpha(model, data, data.t[row], stride) > threshold
        root = root_position(mask, xs, x_anchor)
        drifts.append(f"{frame_no}:{root:.3f}({root - x_anchor:+.3f})")
    print(f"root anchor x*={x_anchor:.4f}")
    print(f"per-frame root x*(drift): {' '.join(drifts)}")


if __name__ == "__main__":
    main()
