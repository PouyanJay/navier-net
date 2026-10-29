"""Run the trusted-fields acceptance benchmark on a rented GPU through Modal.

The pipeline is unchanged: each remote function runs the ``naviernet`` CLI on a
container that holds this checkout's ``src`` and ``configs`` and the processed
archive of one series, with ``training.device=cuda``. Results come back to the
local ``outputs/`` directory exactly as a local run would write them, and are
kept on a Modal volume as well.

Usage, from the repository root (after ``modal token new`` once)::

    modal run scripts/modal_bench.py --mode time                 # 60 steps, per-step cost
    modal run scripts/modal_bench.py --mode bench --seeds 0,1,2  # the acceptance run
    modal run scripts/modal_bench.py --mode bench --seeds 0 --gpu A100

Each seed is one container; the seeds run in parallel. Only the code, the
configs and the 2.7 MB processed archive leave this machine.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/navier-net"
DATASET = "Series-1-w300"
# The acceptance configuration (docs/plans/trustworthy-hidden-fields-plan.md).
RECIPE = ["experiment=trusted_w300", "model=trusted", "training=trusted"]
RUN_PREFIX = "trust-wpf"
GPU = "A10G"
CPUS = 4
# The exact kinematic condition costs about 70% more per step (about 7 h for
# 12000 steps on the A10); a timeout is not retried, so leave room.
TIMEOUT_S = 12 * 3600

# The processed archive the container trains on: the series' current one, or
# another (NAVIERNET_TENSORS, a path) so a run resumed after a re-preprocess
# keeps the outlines it started on.
TENSORS = Path(
    os.environ.get("NAVIERNET_TENSORS", REPO / "data" / "processed" / DATASET / "tensors.npz")
)
# The traced outline model.measured_outline reads, kept beside the tensors.
OUTLINE = TENSORS.with_name("outline.npz")

app = modal.App("navier-net-bench")

# The container: Debian, Python 3.12, the project's dependencies (the Linux
# torch wheel carries CUDA), and this checkout mounted at run time so every
# `modal run` sees the code as it is on disk.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "opencv-python-headless>=4.8",
        "pillow>=10.0",
        "matplotlib>=3.7",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
    )
    .env(
        {
            "PYTHONPATH": f"{REMOTE_ROOT}/src",
            "NAVIERNET_CONFIG_DIR": f"{REMOTE_ROOT}/configs",
            "OMP_NUM_THREADS": str(CPUS),
        }
    )
    .add_local_dir(REPO / "src", f"{REMOTE_ROOT}/src")
    .add_local_dir(REPO / "configs", f"{REMOTE_ROOT}/configs")
    .add_local_file(
        TENSORS,
        f"{REMOTE_ROOT}/data/processed/{DATASET}/tensors.npz",
    )
)
if OUTLINE.exists():
    image = image.add_local_file(OUTLINE, f"{REMOTE_ROOT}/data/processed/{DATASET}/outline.npz")

# Every run directory is also kept here, so a lost download is not a lost run.
volume = modal.Volume.from_name("navier-net-outputs", create_if_missing=True)
OUTPUTS = f"{REMOTE_ROOT}/outputs"


def _cli(overrides: list[str]) -> None:
    """Run the pipeline CLI on the container, streaming its log."""
    command = [sys.executable, "-m", "naviernet.cli", f"paths.root={REMOTE_ROOT}", *overrides]
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)


def _pack(run_name: str) -> bytes:
    """The run directory as a tar.gz, the shape the local side unpacks."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(f"{OUTPUTS}/{run_name}", arcname=run_name)
    return buffer.getvalue()


# A save every this many steps, so a container lost mid-run resumes from its
# last checkpoint on the volume instead of from nothing.
CHECKPOINT_EVERY = 1000


def _common(run_name: str, seed: int, gpu_steps: list[str]) -> list[str]:
    return [
        f"dataset={DATASET}",
        *RECIPE,
        f"run_name={run_name}",
        f"training.seed={seed}",
        "training.device=cuda",
        f"training.checkpoint_every={CHECKPOINT_EVERY}",
        *gpu_steps,
    ]


@app.function(
    image=image, gpu=GPU, cpu=CPUS, memory=8192, timeout=TIMEOUT_S, volumes={OUTPUTS: volume}
)
def time_steps(seed: int = 0) -> float:
    """Sixty steps with the physics engaged after twenty: the per-step cost."""
    import torch

    run_name = f"{RUN_PREFIX}-timing"
    print("GPU:", torch.cuda.get_device_name(0))
    overrides = _common(
        run_name,
        seed,
        [
            "training.steps=60",
            "training.log_every=20",
            "training.stage_b_warmup_steps=20",
            "training.rebalance_every=20",
            "stage=train",
        ],
    )
    started = time.monotonic()
    _cli(overrides)
    per_step = (time.monotonic() - started) / 60
    print(f"about {per_step:.2f} s per step including start-up")
    return per_step


def _target_steps(extra: list[str]) -> int:
    """The run's step count, read from the recipe composed with the run's own
    overrides, so the target is never typed twice and an override of
    training.steps is honoured."""
    from hydra import compose, initialize_config_dir

    from naviernet.config import register_configs

    register_configs()
    with initialize_config_dir(config_dir=f"{REMOTE_ROOT}/configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"dataset={DATASET}", *RECIPE, *extra])
    return int(cfg.training.steps)


def _completed_steps(run_name: str) -> int:
    """Steps the run's checkpoint on the volume has already done (0 when none)."""
    import torch

    checkpoint = Path(OUTPUTS) / run_name / "checkpoints" / "ckpt.pt"
    if not checkpoint.exists():
        return 0
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state"]
    return int(state["done"])


@app.function(
    image=image, gpu=GPU, cpu=CPUS, memory=8192, timeout=TIMEOUT_S, volumes={OUTPUTS: volume}
)
def bench(
    seed: int, prefix: str = RUN_PREFIX, overrides: str = ""
) -> tuple[str, bytes, str | None]:
    """Train and evaluate (with the audit) one seed; returns its run directory
    and, when the evaluate stage failed, the failure, so a trained checkpoint
    is never lost to an evaluation bug (it can be audited locally).

    A run name already on the volume resumes from its last checkpoint and
    trains only the steps left to the recipe's total. That matters because
    Modal restarts a preempted container with the same input: training.steps
    is additive per call, so a restart that passed it unchanged trained the
    whole count again (trust-wpf6-s0, preempted at 7500, ended at 19000).

    ``overrides`` are extra Hydra overrides, space-separated, applied after the
    recipe to both stages, so an experiment on top of the recipe is recorded in
    the run's own config snapshot."""
    extra = overrides.split()
    run_name = f"{prefix}-s{seed}"
    remaining = _target_steps(extra) - _completed_steps(run_name)
    if remaining > 0:
        print(f"{run_name}: {remaining} steps to train", flush=True)
        _cli(_common(run_name, seed, [*extra, f"training.steps={remaining}", "stage=train"]))
        volume.commit()
    failure = None
    try:
        _cli(_common(run_name, seed, [*extra, "stage=evaluate"]))
    except subprocess.CalledProcessError as exc:
        failure = f"evaluate stage failed: {exc}"
    volume.commit()
    return run_name, _pack(run_name), failure


# --- the step-cost sweep (--mode sweep) ----------------------------------------
# What the next runs train: the trusted recipe with the test settings on top.
SWEEP_EXTRAS = [
    "training.weights.energy_liquid=10.0",
    "training.weights.brinkman=5.0",
    "experiment.footprint_wetting=microlayer",
    "experiment.microlayer_mode=fit",
    "experiment.microlayer_um=0.5",
    "training.growth_closure_weight=1.0",
    "training.growth_closure_stride=6",
    "model.hard_kinematics=true",
]
# The recipe's batch sizes, scaled together by the sweep.
SWEEP_BATCH = {"training.n_coll": 3072, "training.n_data": 3072, "training.n_bc": 512}
# Physics steps timed, after a short warm-up; the first logged physics step is
# skipped so start-up and the handoff are not counted.
SWEEP_WARMUP, SWEEP_FIRST, SWEEP_LAST = 10, 20, 50


def _logged_at(log: str, step: int) -> float:
    """The wall clock of the log line for ``step``, in seconds."""
    import datetime as dt
    import re

    match = re.search(
        rf"^\[([^\]]+)\]\[naviernet\.training\]\[INFO\] - step +{step} \| lr", log, re.M
    )
    if match is None:
        raise RuntimeError(f"no log line for step {step}")
    return dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()


def _time_physics_steps(label: str, scale: int, device: str, threads: int) -> dict:
    """Seconds per physics step of the sweep's recipe at ``scale`` times the
    batch, on this container."""
    run_name = f"sweep-{label}-x{scale}"
    overrides = [
        f"dataset={DATASET}",
        *RECIPE,
        *SWEEP_EXTRAS,
        f"run_name={run_name}",
        f"training.device={device}",
        "training.warmup_cache=false",
        "training.checkpoint_every=0",
        f"training.steps={SWEEP_LAST}",
        f"training.stage_b_warmup_steps={SWEEP_WARMUP}",
        "training.rebalance_every=1000",
        "training.log_every=10",
        *(f"{key}={value * scale}" for key, value in SWEEP_BATCH.items()),
        "stage=train",
    ]
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    command = [sys.executable, "-m", "naviernet.cli", f"paths.root={REMOTE_ROOT}", *overrides]
    done = subprocess.run(command, cwd=REMOTE_ROOT, env=env, capture_output=True, text=True)
    log = done.stdout + done.stderr
    if done.returncode != 0:
        return {"label": label, "scale": scale, "error": log[-1500:]}
    per_step = (_logged_at(log, SWEEP_LAST) - _logged_at(log, SWEEP_FIRST)) / (
        SWEEP_LAST - SWEEP_FIRST
    )
    return {"label": label, "scale": scale, "seconds_per_step": per_step}


@app.function(image=image, gpu=GPU, cpu=CPUS, memory=16384, timeout=3600)
def sweep_gpu(scale: int, label: str = GPU) -> dict:
    import torch

    result = _time_physics_steps(label, scale, "cuda", CPUS)
    result["device_name"] = torch.cuda.get_device_name(0)
    return result


@app.function(image=image, cpu=8, memory=16384, timeout=3600)
def sweep_cpu(scale: int) -> dict:
    return _time_physics_steps("cpu8", scale, "cpu", 8)


def _sweep() -> None:
    """Every machine and batch scale in parallel, then one table."""
    calls = [sweep_gpu.spawn(scale) for scale in (1, 2, 4, 8)]
    for gpu in ("L4", "A100", "H100"):
        calls += [sweep_gpu.with_options(gpu=gpu).spawn(scale, gpu) for scale in (1, 4)]
    calls += [sweep_cpu.spawn(1)]
    for call in calls:
        result = call.get()
        if "error" in result:
            print(
                f"{result['label']:>6} x{result['scale']}: failed\n{result['error']}",
                flush=True,
            )
            continue
        print(
            f"{result['label']:>6} x{result['scale']}: {result['seconds_per_step']:.3f} s per step "
            f"({result.get('device_name', 'cpu')})",
            flush=True,
        )


def _unpack(run_name: str, payload: bytes) -> Path:
    """Write a returned run directory under the local outputs/, never over one."""
    target = REPO / "outputs" / run_name
    if target.exists():
        raise SystemExit(f"{target} exists; move it aside before downloading")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        tar.extractall(REPO / "outputs", filter="data")
    return target


@app.local_entrypoint()
def main(
    mode: str = "time",
    seeds: str = "0,1,2",
    gpu: str = GPU,
    prefix: str = RUN_PREFIX,
    overrides: str = "",
) -> None:
    """``time`` measures the per-step cost; ``bench`` runs the seeds in parallel
    under ``prefix`` (a new prefix for a new configuration; the same one resumes)."""
    # The functions as declared, unless another card is asked for: a function
    # re-optioned at run time is a separate object, and only the declared one
    # has been seen to dispatch a `.map` promptly.
    timing = time_steps if gpu == GPU else time_steps.with_options(gpu=gpu)
    runner = bench if gpu == GPU else bench.with_options(gpu=gpu)
    if mode == "time":
        per_step = timing.remote(0)
        print(
            f"per step: {per_step:.2f} s -> 12000 steps in about {per_step * 12000 / 60:.0f} min"
        )
        return
    if mode == "sweep":
        _sweep()
        return
    if mode != "bench":
        raise SystemExit("mode must be 'time', 'sweep' or 'bench'")
    wanted = [int(s) for s in seeds.split(",") if s.strip()]
    print(
        f"dispatching seeds {wanted} on {gpu} as {prefix}-s* {overrides}".rstrip(), flush=True
    )
    # One seed's failure must not end the app and cancel the others mid-run.
    for result in runner.map(
        wanted,
        kwargs={"prefix": prefix, "overrides": overrides},
        order_outputs=False,
        return_exceptions=True,
    ):
        if isinstance(result, Exception):
            print(f"a seed failed: {result}", flush=True)
            continue
        run_name, payload, failure = result
        print(f"downloaded {_unpack(run_name, payload)}", flush=True)
        if failure:
            print(f"{run_name}: {failure}; audit it locally", flush=True)


if __name__ == "__main__":
    os.execvp("modal", ["modal", "run", __file__, *sys.argv[1:]])
