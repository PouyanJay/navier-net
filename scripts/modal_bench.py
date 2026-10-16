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
TIMEOUT_S = 6 * 3600

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
        REPO / "data" / "processed" / DATASET / "tensors.npz",
        f"{REMOTE_ROOT}/data/processed/{DATASET}/tensors.npz",
    )
)

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


@app.function(
    image=image, gpu=GPU, cpu=CPUS, memory=8192, timeout=TIMEOUT_S, volumes={OUTPUTS: volume}
)
def bench(seed: int, prefix: str = RUN_PREFIX) -> tuple[str, bytes, str | None]:
    """Train and evaluate (with the audit) one seed; returns its run directory
    and, when the evaluate stage failed, the failure, so a trained checkpoint
    is never lost to an evaluation bug (it can be audited locally). A run name
    already on the volume resumes from its last checkpoint."""
    run_name = f"{prefix}-s{seed}"
    _cli(_common(run_name, seed, ["stage=train"]))
    volume.commit()
    failure = None
    try:
        _cli(_common(run_name, seed, ["stage=evaluate"]))
    except subprocess.CalledProcessError as exc:
        failure = f"evaluate stage failed: {exc}"
    volume.commit()
    return run_name, _pack(run_name), failure


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
    mode: str = "time", seeds: str = "0,1,2", gpu: str = GPU, prefix: str = RUN_PREFIX
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
    if mode != "bench":
        raise SystemExit("mode must be 'time' or 'bench'")
    wanted = [int(s) for s in seeds.split(",") if s.strip()]
    print(f"dispatching seeds {wanted} on {gpu} as {prefix}-s*", flush=True)
    # One seed's failure must not end the app and cancel the others mid-run.
    for result in runner.map(
        wanted, kwargs={"prefix": prefix}, order_outputs=False, return_exceptions=True
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
