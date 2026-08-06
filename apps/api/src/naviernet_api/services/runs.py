"""Reading runs from `outputs/`.

A "run" is a directory under `outputs/` that the pipeline produced. This module
locates its artifacts through the reused `RunPaths` layout (constructed directly,
no Hydra composition needed) and reads the JSON the pipeline already writes. It
performs no training; the only run it mutates is one it deletes wholesale
(:func:`delete_run`).
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths
from naviernet_api.models import ArtifactFlags, RunDetail, RunSummary
from naviernet_api.settings import Settings

if TYPE_CHECKING:
    from omegaconf import DictConfig

log = get_logger(__name__)

# Run ids and dataset names become directory names, so constrain both hard
# (defense against path traversal; see SECURITY.md §3).
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Directory names under outputs/ that are not individual runs.
_NON_RUN_DIRS = {"multirun"}

# Served figure files must be PNG names inside the run's figures dir, at most
# one dataset subdirectory deep (joint runs render per-dataset figures).
_FIGURE_NAME_RE = re.compile(r"^(?:[A-Za-z0-9._-]+/)?[A-Za-z0-9._-]+\.png$")


def _safe_run_dir(settings: Settings, run_id: str) -> Path | None:
    """Resolve a run id to its directory, or None if invalid / missing."""
    # "." matches the character class but would resolve to outputs/ itself,
    # collapsing per-run scoping (same guard as dataset ids; SECURITY.md §3).
    if not _RUN_ID_RE.match(run_id) or run_id in {".", ".."}:
        return None
    outputs = settings.outputs_dir.resolve()
    run_dir = (outputs / run_id).resolve()
    if not run_dir.is_relative_to(outputs) or not run_dir.is_dir():
        return None
    return run_dir


def delete_run(settings: Settings, run_id: str) -> bool:
    """Remove a run's output directory. Returns True if it existed.

    Confined to `outputs/` (SECURITY.md §3); an invalid or unknown id is a no-op.
    """
    run_dir = _safe_run_dir(settings, run_id)
    if run_dir is None:
        return False
    shutil.rmtree(run_dir)
    log.info("deleted run %s", run_id)
    return True


def _run_paths(settings: Settings, run_id: str, dataset: str | None) -> RunPaths:
    """RunPaths for a run, reusing the pipeline's artifact layout."""
    ds = dataset or run_id
    return RunPaths(
        raw_dir=settings.data_raw_dir / ds,
        processed_dir=settings.repo_root / "data" / "processed" / ds,
        output_dir=settings.outputs_dir / run_id,
    )


def _run_paths_or_none(settings: Settings, run_id: str) -> RunPaths | None:
    """RunPaths for a validated, existing run; the common preamble of the
    artifact readers below."""
    run_dir = _safe_run_dir(settings, run_id)
    if run_dir is None:
        return None
    dataset = _dataset_of(run_dir, _read_json(run_dir / "metrics.json"))
    return _run_paths(settings, run_id, dataset)


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        # A corrupt artifact must not masquerade as an absent one silently.
        log.warning("could not read %s: %s", path, exc)
        return None


def _read_hydra_config(run_dir: Path) -> dict | None:
    """The resolved config snapshot Hydra wrote for the run."""
    snapshot = run_dir / ".hydra" / "config.yaml"
    if not snapshot.is_file():
        return None
    from omegaconf import OmegaConf

    try:
        return OmegaConf.to_container(OmegaConf.load(snapshot), resolve=True)  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001 (any parse failure means "unreadable")
        log.warning("could not read config snapshot in %s: %s", run_dir, exc)
        return None


def run_paths_for(settings: Settings, run_id: str) -> RunPaths | None:
    """RunPaths for a validated, existing run (None for bad/unknown ids)."""
    return _run_paths_or_none(settings, run_id)


def load_run_config(settings: Settings, run_id: str) -> DictConfig | None:
    """The run's own config, schema-merged and re-pinned to this repository.

    Merging the snapshot over the structured schema means unknown keys or
    wrong types fail here instead of deep inside the pipeline. Returned
    mutable so callers can override run-scoped values before locking it.
    """
    run_dir = _safe_run_dir(settings, run_id)
    if run_dir is None:
        return None
    snapshot = run_dir / ".hydra" / "config.yaml"
    if not snapshot.is_file():
        return None
    from omegaconf import OmegaConf

    from naviernet.config.schema import Config

    try:
        cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(snapshot))
    except Exception as exc:  # noqa: BLE001 (any parse/merge failure means "unreadable")
        log.warning("could not load config snapshot for %s: %s", run_id, exc)
        return None
    cfg.paths.root = str(settings.repo_root)
    cfg.training.device = "cpu"  # the server never schedules onto an accelerator
    return cfg


def _dataset_of(run_dir: Path, metrics: dict | None) -> str | None:
    """The run's dataset name, validated so it is safe to use in a path.

    The value comes from user-influenced sources (`metrics.json`, the Hydra
    `dataset=` override recorded in the snapshot), so it is checked against the
    same allowlist as a run id before it can flow into `data/processed/<ds>`.
    """
    candidate: str | None = None
    if metrics and metrics.get("dataset"):
        candidate = str(metrics["dataset"])
    else:
        config = _read_hydra_config(run_dir)
        if config and config.get("dataset"):
            candidate = str(config["dataset"])
    if candidate is not None and not _RUN_ID_RE.match(candidate):
        log.warning("ignoring unsafe dataset name %r in %s", candidate, run_dir)
        return None
    return candidate


def _load_checkpoint_state(checkpoint: Path) -> dict | None:
    """Deserialize a checkpoint, or None if absent/unreadable.

    The checkpoint is a first-party artifact this repo produced, so loading it
    with `weights_only=False` is acceptable (SECURITY.md §1). Torch is imported
    lazily so it never costs anything on paths that don't need it. A corrupt or
    unexpected checkpoint is logged rather than silently swallowed.
    """
    if not checkpoint.is_file():
        return None
    import torch

    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (RuntimeError, EOFError, OSError, ValueError) as exc:
        log.warning("could not read checkpoint %s: %s", checkpoint, exc)
        return None


def _checkpoint_steps(checkpoint: Path) -> int | None:
    """Completed training steps from the checkpoint's run state."""
    state = _load_checkpoint_state(checkpoint)
    done = state.get("state", {}).get("done") if state else None
    return int(done) if done is not None else None


def _datasets_of(run_dir: Path, metrics: dict | None) -> list[str]:
    """Every dataset a run spans: the v2 `datasets` list for a joint run, else
    the single dataset. Names are validated like run ids (they become paths)."""
    if metrics and isinstance(metrics.get("datasets"), list):
        names = [str(name) for name in metrics["datasets"]]
        safe = [name for name in names if _RUN_ID_RE.match(name)]
        for name in set(names) - set(safe):
            log.warning("ignoring unsafe dataset name %r in %s", name, run_dir)
        if safe:
            return safe
    dataset = _dataset_of(run_dir, metrics)
    return [dataset] if dataset is not None else []


def _heldout_of(run_dir: Path, metrics: dict | None) -> list[str]:
    """The run's held-out (axis-B) datasets, validated like every dataset name."""
    if not metrics or not isinstance(metrics.get("heldout_datasets"), list):
        return []
    names = [str(name) for name in metrics["heldout_datasets"]]
    safe = [name for name in names if _RUN_ID_RE.match(name)]
    for name in set(names) - set(safe):
        log.warning("ignoring unsafe held-out dataset name %r in %s", name, run_dir)
    return safe


def _run_date(run_dir: Path) -> str | None:
    """When the run last changed, as an ISO UTC timestamp (directory mtime —
    runs record no explicit timestamp of their own)."""
    try:
        stamp = run_dir.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat(timespec="seconds")


def _live_state(run_id: str):
    """The run-manager job for this run, if the server launched it.

    Imported lazily: run_manager imports this module at module level, so a
    top-level import here would be a cycle.
    """
    from naviernet_api.services import run_manager

    return run_manager.status(run_id)


def list_runs(settings: Settings, datasets: set[str] | None = None) -> list[RunSummary]:
    """Every run directory under `outputs/`, newest name last (sorted).

    With `datasets`, only runs spanning at least one of those datasets are
    returned — the project-scoped listing (a run with no identifiable dataset
    cannot belong to a project, so it is excluded from scoped listings).
    """
    if not settings.outputs_dir.is_dir():
        return []

    summaries: list[RunSummary] = []
    for run_dir in sorted(settings.outputs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name in _NON_RUN_DIRS:
            continue
        metrics = _read_json(run_dir / "metrics.json")
        spanned = _datasets_of(run_dir, metrics)
        if datasets is not None and not set(spanned) & datasets:
            continue
        summaries.append(_summarize_run(settings, run_dir, metrics, spanned))
    return summaries


def _summarize_run(
    settings: Settings, run_dir: Path, metrics: dict | None, spanned: list[str]
) -> RunSummary:
    """One run's list row: live job state wins over the on-disk view."""
    run_id = run_dir.name
    dataset = _dataset_of(run_dir, metrics)
    paths = _run_paths(settings, run_id, dataset)
    live = _live_state(run_id)
    if live is not None and live.state in ("queued", "running"):
        status = "running"
    elif live is not None and live.state == "error":
        status = "failed"
    else:
        status = "trained" if paths.checkpoint.is_file() else "empty"
    config = _read_hydra_config(run_dir)
    per_frame = metrics.get("iou_per_frame") if metrics else None
    return RunSummary(
        id=run_id,
        dataset=dataset,
        status=status,
        iou_holdout=metrics.get("iou_holdout") if metrics else None,
        datasets=spanned,
        heldout_datasets=_heldout_of(run_dir, metrics),
        val_iou_mean=metrics.get("val_iou_mean") if metrics else None,
        iou_val=metrics.get("iou_val") if metrics else None,
        iou_mean=metrics.get("iou_mean") if metrics else None,
        # Written as {frame: iou} by the evaluator and as a bare list by older
        # runs; both answer "how many frames was this mean taken over".
        n_frames=len(per_frame) if isinstance(per_frame, (dict, list)) else None,
        seed=_seed_of(config),
        recipe=recipe_of(config),
        date=_run_date(run_dir),
        steps_done=live.steps_done if status == "running" else None,
        steps_total=live.steps_total if status == "running" else None,
    )


def _seed_of(config: dict | None) -> int | None:
    """The seed the run actually trained with.

    Worth reading rather than parsing off the id: only sweep children are named
    by the machine (``<sweep_id>-s<seed>``). A hand-named run's ``-s2`` suffix is
    a label someone typed, and several in this repository disagree with the seed
    their own config recorded.
    """
    seed = (config or {}).get("training", {}).get("seed")
    return int(seed) if isinstance(seed, (int, float)) else None


# What a run did differently, in the vocabulary the Solver's rail uses. The
# interface treatment comes first (it is the run's formulation, and every run has
# one), then the opt-in extras, each named only when it is actually on. A run at
# the recommended recipe therefore reads as one chip, not eight.
_EXTRAS: tuple[tuple[str, str, str], ...] = (
    ("model", "hard_pin", "pin"),
    ("model", "allow_pinch", "pinch"),
    ("model", "evolving_width", "evwidth"),
    ("training", "causal_weighting", "causal"),
    ("training", "adaptive_collocation", "adaptive"),
    ("training", "kinematics", "kinematics"),
    ("training", "front_velocity", "front-v"),
)


def recipe_of(config: dict | None) -> list[str] | None:
    """The short recipe chips for a run, or None when it recorded no config.

    None is not the same as ``[]``: an empty list says "the recommended recipe,
    exactly", while None says the run predates the config snapshot and the UI
    must not guess at what it ran.
    """
    if not config:
        return None
    model = config.get("model") or {}
    training = config.get("training") or {}
    if model.get("sharp_interface"):
        chips = ["sharp"]
    elif model.get("front_geometry"):
        chips = ["front"]
    else:
        chips = ["diffuse"]
    chips += [
        label
        for group, key, label in _EXTRAS
        if (model if group == "model" else training).get(key)
    ]
    if training.get("weighting") not in (None, "gradnorm"):
        chips.append(str(training["weighting"]).upper())
    return chips


def read_dataset_and_metrics(
    settings: Settings, run_id: str
) -> tuple[str | None, dict | None] | None:
    """The run's dataset name and metrics only; cheap (no checkpoint load)."""
    run_dir = _safe_run_dir(settings, run_id)
    if run_dir is None:
        return None
    metrics = _read_json(run_dir / "metrics.json")
    return _dataset_of(run_dir, metrics), metrics


def get_run(settings: Settings, run_id: str) -> RunDetail | None:
    """Full detail for one run, or None if it doesn't exist / id is invalid."""
    run_dir = _safe_run_dir(settings, run_id)
    if run_dir is None:
        return None

    metrics = _read_json(run_dir / "metrics.json")
    dataset = _dataset_of(run_dir, metrics)
    paths = _run_paths(settings, run_id, dataset)

    # Joint runs render figures into per-dataset subdirectories; list them
    # with their relative names so the client can request them back verbatim.
    figures = (
        sorted(
            str(p.relative_to(paths.figures_dir))
            for pattern in ("*.png", "*/*.png")
            for p in paths.figures_dir.glob(pattern)
        )
        if paths.figures_dir.is_dir()
        else []
    )
    artifacts = ArtifactFlags(
        checkpoint=paths.checkpoint.is_file(),
        metrics=metrics is not None,
        groups=paths.groups_json.is_file(),
        video=paths.video.is_file(),
        figures=figures,
    )

    return RunDetail(
        id=run_id,
        dataset=dataset,
        status="trained" if artifacts.checkpoint else "empty",
        steps=_checkpoint_steps(paths.checkpoint),
        metrics=metrics,
        config=_read_hydra_config(run_dir),
        artifacts=artifacts,
    )


def read_groups(settings: Settings, run_id: str) -> dict | None:
    """The dimensionless groups sub-dict from `dimensionless_groups.json`."""
    paths = _run_paths_or_none(settings, run_id)
    if paths is None:
        return None
    payload = _read_json(paths.groups_json)
    return payload.get("groups") if payload else None


class _Artifact(NamedTuple):
    """Where one of the evaluate stage's artifacts lives: the single-run path,
    the per-dataset path a joint run writes, and what to call it in a log."""

    shared: Path
    per_dataset: Callable[[str], Path]
    kind: str


def _dataset_scoped_json(artifact: _Artifact, dataset: str | None) -> dict | None:
    """One of the evaluate stage's artifacts, optionally scoped to a joint run's
    dataset. The name is validated because it becomes part of a path."""
    if dataset is None:
        return _read_json(artifact.shared)
    if not _RUN_ID_RE.match(dataset):
        log.warning("rejecting unsafe %s dataset name %r", artifact.kind, dataset)
        return None
    return _read_json(artifact.per_dataset(dataset))


def read_trajectory(settings: Settings, run_id: str, dataset: str | None = None) -> dict | None:
    """The growth-kinematics arrays the evaluate stage wrote, or None."""
    paths = _run_paths_or_none(settings, run_id)
    if paths is None:
        return None
    return _dataset_scoped_json(
        _Artifact(paths.trajectory_json, paths.trajectory_json_for, "trajectory"),
        dataset,
    )


def read_front_velocity(
    settings: Settings, run_id: str, dataset: str | None = None
) -> dict | None:
    """The front-velocity report the evaluate stage wrote, or None.

    Absent for any run evaluated before this artifact existed, which the caller
    reports as "re-run evaluate" rather than as an error.
    """
    paths = _run_paths_or_none(settings, run_id)
    if paths is None:
        return None
    return _dataset_scoped_json(
        _Artifact(
            paths.front_velocity_json,
            paths.front_velocity_json_for,
            "front-velocity",
        ),
        dataset,
    )


def read_loss_history(settings: Settings, run_id: str) -> list[dict] | None:
    """The per-log-step loss records saved in the checkpoint's run state."""
    paths = _run_paths_or_none(settings, run_id)
    if paths is None:
        return None
    state = _load_checkpoint_state(paths.checkpoint)
    hist = state.get("state", {}).get("hist") if state else None
    return [dict(record) for record in hist] if hist is not None else None


def figure_path(settings: Settings, run_id: str, name: str) -> Path | None:
    """Path to a figure PNG, confined to the run's figures dir (SECURITY.md §3)."""
    # ".." matches the name character class; reject it before it can resolve
    # upward (the containment check below would too — defense in depth).
    if not _FIGURE_NAME_RE.match(name) or ".." in name.split("/"):
        return None
    paths = _run_paths_or_none(settings, run_id)
    if paths is None:
        return None
    figures_dir = paths.figures_dir.resolve()
    path = (figures_dir / name).resolve()
    if not path.is_relative_to(figures_dir) or not path.is_file():
        return None
    return path


def video_path(settings: Settings, run_id: str) -> Path | None:
    """Path to the run's rendered video, or None if absent."""
    paths = _run_paths_or_none(settings, run_id)
    if paths is None:
        return None
    return paths.video if paths.video.is_file() else None


def checkpoint_path(settings: Settings, run_id: str) -> Path | None:
    """Path to the run's checkpoint, or None if absent."""
    paths = _run_paths_or_none(settings, run_id)
    if paths is None:
        return None
    return paths.checkpoint if paths.checkpoint.is_file() else None


def tensors_path(settings: Settings, run_id: str) -> Path | None:
    """Path to the preprocessed tensors for the run's dataset, or None.

    Confined to `data/processed/` (SECURITY.md §3): the dataset name is derived
    data, so the resolved path is checked to stay inside the processed root even
    though `_dataset_of` already validates the name.
    """
    paths = _run_paths_or_none(settings, run_id)
    if paths is None:
        return None
    processed_root = (settings.repo_root / "data" / "processed").resolve()
    tensors = paths.tensors.resolve()
    if not tensors.is_relative_to(processed_root) or not tensors.is_file():
        return None
    return tensors
