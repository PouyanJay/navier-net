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

from naviernet.utils.paths import RunPaths
from naviernet_api.models import ArtifactFlags, RunDetail, RunSummary
from naviernet_api.settings import Settings

# Run ids become directory names, so constrain them hard (defense against path
# traversal — see SECURITY.md §3).
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Directory names under outputs/ that are not individual runs.
_NON_RUN_DIRS = {"multirun"}


def _safe_run_dir(settings: Settings, run_id: str) -> Path | None:
    """Resolve a run id to its directory, or None if invalid / missing."""
    if not _RUN_ID_RE.match(run_id):
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
    if metrics and metrics.get("dataset"):
        return str(metrics["dataset"])
    config = _read_hydra_config(run_dir)
    if config and config.get("dataset"):
        return str(config["dataset"])
    return None


def _checkpoint_steps(checkpoint: Path) -> int | None:
    """Completed training steps from the checkpoint's run state.

    The checkpoint is a first-party artifact this repo produced, so loading it
    with `weights_only=False` is acceptable (SECURITY.md §1). Torch is imported
    lazily so it never costs anything on the list path.
    """
    if not checkpoint.is_file():
        return None
    import torch

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return int(state["state"]["done"])
    except Exception:
        return None


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
