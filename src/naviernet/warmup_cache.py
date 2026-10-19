"""The shared warm-up: the Stage-A outline fit, trained once and reused.

Steps 1 to ``stage_b_warmup_steps`` fit the outline to the images with the
physics gated off, so every run that agrees on what that stretch executes ends
it in the same state. With ``training.warmup_cache`` on, the first such run
saves its checkpoint at the end of the warm-up under a key of those inputs, and
every later one starts from that checkpoint instead of retraining it.

The key is everything the warm-up reads, and nothing it does not:

* the processed archive, by content, so a change to the traced outlines (or
  to the algorithm that traces them) is a new key;
* the resolved config, minus the bookkeeping fields and the fields that act
  only once the physics engages (:data:`AFTER_WARMUP` and the Stage-B terms'
  weights), so two physics variants on the same outline share one warm-up;
* the source of the modules the warm-up runs, and :data:`WARMUP_CODE_VERSION`
  for the shared modules (the model, the trainer, the residuals) whose edits
  are mostly about the physics stage.

A run that misses the cache trains its warm-up as a call of its own and then
resumes, exactly as a run that hits it does, so the two paths produce the
same run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path

from omegaconf import OmegaConf

from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths

log = get_logger(__name__)

# Bump when a change to pinn.py, training.py, residuals.py or registry.py alters
# anything the warm-up executes: the outline, the velocity or source the Stage-A
# terms read, those terms, the sampler, the rebalancer, the loop. A physics-only
# change leaves it alone. Over-bumping costs one warm-up; under-bumping reuses a
# stale one, so when in doubt, bump.
WARMUP_CODE_VERSION = 1

# Modules the warm-up runs and the physics stage rarely touches: hashed whole.
WARMUP_MODULES = (
    "data/dataset.py",
    "models/geometry.py",
    "models/layers.py",
    "physics/cap_supervision.py",
    "physics/front_velocity.py",
    "physics/kinematics.py",
    "physics/mass_balance.py",
    "physics/nose_supervision.py",
    "physics/root_supervision.py",
)

# Config fields that never act before the physics engages, or only record the
# run: left out of the key. The Stage-B terms' weights are left out too (see
# _key_config); the trainer resets every weight to its configured value at the
# handoff, so a warm-up rebalanced under one of them is the same warm-up.
AFTER_WARMUP = (
    "run_name",
    "stage",
    "paths",
    "evaluation",
    "training.steps",
    "training.log_every",
    "training.checkpoint_every",
    "training.device",
    "training.warmup_cache",
    "training.temperature_stage_from",
    "training.physics_lr_halflife",
    "training.growth_closure_weight",
    "training.growth_closure_stride",
    "training.growth_closure_outline_gradient",
    "training.frozen_weighting",
    "training.one_way_advection",
    "model.momentum_inertia",
    "model.hard_kinematics",
    "model.kinematic_blend_length",
    "model.kinematic_side",
    "experiment.footprint_wetting",
    "experiment.microlayer_mode",
    "experiment.microlayer_um",
    "experiment.footprint_age_ms",
    "experiment.wet_fraction_mode",
    "experiment.wet_fraction",
)

CACHE_DIR = "_warmup"


def _drop(tree: dict, dotted: str) -> None:
    *parents, leaf = dotted.split(".")
    for name in parents:
        tree = tree.get(name)
        if not isinstance(tree, dict):
            return
    tree.pop(leaf, None)


def _key_config(cfg, stage_b_keys: tuple[str, ...]) -> dict:
    tree = OmegaConf.to_container(cfg, resolve=True)
    for dotted in AFTER_WARMUP:
        _drop(tree, dotted)
    for name in stage_b_keys:
        _drop(tree, f"training.weights.{name}")
    return tree


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def warmup_key(cfg, paths: RunPaths, stage_b_keys: tuple[str, ...]) -> tuple[str, dict]:
    """The cache key of this run's warm-up and the record of what went into it."""
    package = Path(__file__).resolve().parent
    record = {
        "tensors_sha256": _file_digest(paths.tensors),
        "code_version": WARMUP_CODE_VERSION,
        "modules_sha256": {name: _file_digest(package / name) for name in WARMUP_MODULES},
        "config": _key_config(cfg, stage_b_keys),
    }
    blob = json.dumps(record, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16], record


def cache_dir(cfg, paths: RunPaths) -> Path:
    """Where this dataset's cached warm-ups live: beside the runs, so a Modal
    volume that keeps the runs keeps them too."""
    return paths.output_dir.parent / CACHE_DIR / str(cfg.dataset)


def enabled(cfg) -> bool:
    return bool(getattr(cfg.training, "warmup_cache", False))


def reuse_or_train(
    cfg,
    paths: RunPaths,
    steps: int,
    stage_b_keys: tuple[str, ...],
    train_steps: Callable[[int], object],
) -> int:
    """Put the end-of-warm-up checkpoint in place for a fresh run, from the
    cache or by training it with ``train_steps``, and return the steps left of
    ``steps``. A run that already has a checkpoint, a run without a warm-up or
    physics to engage, and a run too short to finish its warm-up are left to
    train as they are."""
    warmup = int(cfg.training.stage_b_warmup_steps)
    if paths.checkpoint.exists() or warmup <= 0 or not stage_b_keys or steps < warmup:
        return steps
    key, record = warmup_key(cfg, paths, stage_b_keys)
    entry = cache_dir(cfg, paths) / key
    cached = entry / "ckpt.pt"
    if cached.exists():
        paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, paths.checkpoint)
        log.info("warm-up reused from %s (%d steps not retrained)", entry, warmup)
        return steps - warmup
    log.info("warm-up %s not cached: training its %d steps once", key, warmup)
    train_steps(warmup)
    entry.mkdir(parents=True, exist_ok=True)
    partial = entry / "ckpt.pt.tmp"
    shutil.copyfile(paths.checkpoint, partial)
    (entry / "meta.json").write_text(
        json.dumps(
            {
                **record,
                "key": key,
                "first_run": str(cfg.run_name),
                "device": str(cfg.training.device),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    partial.replace(cached)
    log.info("warm-up cached at %s", entry)
    return steps - warmup
