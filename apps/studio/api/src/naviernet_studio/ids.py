"""Run and dataset identifiers become directory names, so they are validated
and confined before they touch the filesystem."""

from __future__ import annotations

import json
import re
from pathlib import Path

from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths
from naviernet_studio.settings import Settings

log = get_logger(__name__)

SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
# Directories under outputs/ that are not individual runs.
NON_RUN_DIRS = frozenset({"multirun"})


def is_safe_name(name: str) -> bool:
    return bool(SAFE_NAME.match(name)) and name not in {".", ".."}


def run_dir(settings: Settings, run_id: str) -> Path | None:
    """The run's directory, or None for an invalid, foreign or missing id."""
    if not is_safe_name(run_id):
        return None
    outputs = settings.outputs_dir.resolve()
    candidate = (outputs / run_id).resolve()
    if not candidate.is_relative_to(outputs) or not candidate.is_dir():
        return None
    return candidate


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None


def dataset_of(directory: Path) -> str | None:
    """The run's dataset, from metrics.json or its config snapshot, validated."""
    metrics = read_json(directory / "metrics.json")
    candidate = str(metrics["dataset"]) if metrics and metrics.get("dataset") else None
    if candidate is None:
        snapshot = directory / ".hydra" / "config.yaml"
        if snapshot.is_file():
            from omegaconf import OmegaConf

            raw = OmegaConf.load(snapshot)
            if raw.get("dataset"):
                candidate = str(raw["dataset"])
    if candidate is not None and not is_safe_name(candidate):
        log.warning("ignoring unsafe dataset name %r in %s", candidate, directory)
        return None
    return candidate


def run_paths(settings: Settings, run_id: str) -> RunPaths | None:
    directory = run_dir(settings, run_id)
    if directory is None:
        return None
    dataset = dataset_of(directory) or run_id
    return RunPaths(
        raw_dir=settings.data_raw_dir / dataset,
        processed_dir=settings.data_processed_dir / dataset,
        output_dir=directory,
    )


def list_run_ids(settings: Settings) -> list[str]:
    """Every run directory that holds a checkpoint, newest first."""
    outputs = settings.outputs_dir
    if not outputs.is_dir():
        return []
    found = [
        d
        for d in outputs.iterdir()
        if d.is_dir() and d.name not in NON_RUN_DIRS and is_safe_name(d.name)
        if (d / "checkpoints" / "ckpt.pt").is_file()
    ]
    found.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in found]
