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


def _perimeter(mask) -> int:
    from scipy import ndimage

    return int((mask ^ ndimage.binary_erosion(mask)).sum())


def _frame_report(cfg, model, data, row: int, x_anchor: float) -> dict:
    """One frame's mechanistic measurements: root drift (the hard pin's check),
    front error (the growth constraints' check), and shape structure (component
    count / perimeter ratio -- the front-geometry check)."""
    from scipy import ndimage

    from naviernet.evaluation import front_position, predict_alpha, root_position

    stride = cfg.evaluation.stride
    xs = data.x[::stride]
    mask = predict_alpha(model, data, data.t[row], stride) > cfg.evaluation.threshold
    measured = data.alpha[row, ::stride, ::stride] > cfg.evaluation.threshold
    _, n_comp = ndimage.label(mask)
    return {
        "frame": data.frame_numbers[row],
        "root": root_position(mask, xs, x_anchor),
        "front": front_position(mask, xs, x_anchor),
        "front_true": front_position(measured, xs, x_anchor),
        "components": n_comp,
        "perim_ratio": _perimeter(mask) / max(_perimeter(measured), 1),
    }


def _report_root_drift(cfg, pipe) -> None:
    """Per-frame mechanistic report next to the IoU numbers (see _frame_report)."""
    from naviernet.training import load_model

    model, data, _ = load_model(cfg, pipe.paths)
    x_anchor, _ = data.pin_anchor
    reports = [
        _frame_report(cfg, model, data, row, x_anchor) for row in range(len(data.frame_numbers))
    ]

    print(f"root anchor x*={x_anchor:.4f}")
    line = " ".join(
        f"{r['frame']}:{r['root']:.3f}({r['root'] - x_anchor:+.3f})" for r in reports
    )
    print(f"per-frame root x*(drift): {line}")
    line = " ".join(
        f"{r['frame']}:{r['front']:.3f}({r['front'] - r['front_true']:+.3f})" for r in reports
    )
    print(f"per-frame front x*(vs measured): {line}")
    line = " ".join(f"{r['frame']}:{r['components']}c/{r['perim_ratio']:.2f}p" for r in reports)
    print(f"per-frame shape (components/perimeter-ratio): {line}")


if __name__ == "__main__":
    main()
