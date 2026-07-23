"""Reading runs from `outputs/`.

A "run" is a directory under `outputs/` that the pipeline produced. This module
locates its artifacts through the reused `RunPaths` layout (constructed directly —
no Hydra composition needed) and reads the JSON the pipeline already writes. It
performs no training and never mutates a run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths
from naviernet_api.models import ArtifactFlags, RunDetail, RunSummary
from naviernet_api.settings import Settings

log = get_logger(__name__)

# Run ids and dataset names become directory names, so constrain both hard
# (defense against path traversal — see SECURITY.md §3).
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Directory names under outputs/ that are not individual runs.
_NON_RUN_DIRS = {"multirun"}

# Served figure files must be simple PNG names inside the run's figures dir.
_FIGURE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.png$")


def _safe_run_dir(settings: Settings, run_id: str) -> Path | None:
    """Resolve a run id to its directory, or None if invalid / missing."""
    # "." matches the character class but would resolve to outputs/ itself,
    # collapsing per-run scoping (same guard as dataset ids — SECURITY.md §3).
    if not _RUN_ID_RE.match(run_id) or run_id in {".", ".."}:
        return None
    outputs = settings.outputs_dir.resolve()
    run_dir = (outputs / run_id).resolve()
    if not run_dir.is_relative_to(outputs) or not run_dir.is_dir():
        return None
    return run_dir


def _run_paths(settings: Settings, run_id: str, dataset: str | None) -> RunPaths:
    """RunPaths for a run, reusing the pipeline's artifact layout."""
    ds = dataset or run_id
    return RunPaths(
        raw_dir=settings.data_raw_dir / ds,
        processed_dir=settings.repo_root / "data" / "processed" / ds,
        output_dir=settings.outputs_dir / run_id,
    )


def _run_paths_or_none(settings: Settings, run_id: str) -> RunPaths | None:
    """RunPaths for a validated, existing run — the common preamble of the
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
    except (json.JSONDecodeError, OSError):
        return None


def _read_hydra_config(run_dir: Path) -> dict | None:
    """The resolved config snapshot Hydra wrote for the run."""
    snapshot = run_dir / ".hydra" / "config.yaml"
    if not snapshot.is_file():
        return None
    from omegaconf import OmegaConf

    try:
        return OmegaConf.to_container(OmegaConf.load(snapshot), resolve=True)  # type: ignore[return-value]
    except Exception:
        return None


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


def list_runs(settings: Settings) -> list[RunSummary]:
    """Every run directory under `outputs/`, newest name last (sorted)."""
    if not settings.outputs_dir.is_dir():
        return []

    summaries: list[RunSummary] = []
    for run_dir in sorted(settings.outputs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name in _NON_RUN_DIRS:
            continue
        run_id = run_dir.name
        metrics = _read_json(run_dir / "metrics.json")
        dataset = _dataset_of(run_dir, metrics)
        paths = _run_paths(settings, run_id, dataset)
        summaries.append(
            RunSummary(
                id=run_id,
                dataset=dataset,
                status="trained" if paths.checkpoint.is_file() else "empty",
                iou_holdout=(metrics or {}).get("iou_holdout"),
            )
        )
    return summaries


def read_dataset_and_metrics(
    settings: Settings, run_id: str
) -> tuple[str | None, dict | None] | None:
    """The run's dataset name and metrics only — cheap (no checkpoint load)."""
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

    figures = (
        sorted(p.name for p in paths.figures_dir.glob("*.png"))
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
    if not _FIGURE_NAME_RE.match(name):
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
