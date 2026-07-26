"""Project metadata: one JSON file per project under `projects/`.

A project is the platform's scoping unit: a uuid identity with an editable
name and description, linked to a dataset under `data/raw/` once its first
sequence is uploaded. The file is the source of truth (no database), matching
the platform's filesystem-first architecture.

Datasets that predate the projects layer are materialized into project files
the first time projects are listed, so every project is immediately editable.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from naviernet.utils.logging import get_logger
from naviernet_api.models import ProjectSummary, ProjectUpdate
from naviernet_api.services import datasets as datasets_service
from naviernet_api.services import run_manager
from naviernet_api.services import runs as runs_service
from naviernet_api.settings import Settings

log = get_logger(__name__)

# Serializes every read-modify-write on the project files (same idiom as
# run_manager): concurrent PATCHes must not lose edits, and concurrent lists
# must not materialize the same legacy dataset twice.
_lock = threading.Lock()

_PROJECT_ID_RE = re.compile(r"^[0-9a-f]{32}$")  # uuid4().hex
MAX_NAME_CHARS = 120
MAX_DESCRIPTION_CHARS = 2000
MAX_DATASETS = 50

# Purpose line for materialized legacy datasets (the platform's experiment).
_LEGACY_DESCRIPTION = (
    "Reconstruct the hidden velocity and volume-fraction fields of a confined "
    "vapor slug from its high-speed image sequence."
)


class ProjectError(ValueError):
    """A rejected project operation (bad name, unknown dataset, …)."""


class ProjectInUseError(ProjectError):
    """A delete blocked because a training run is live on the project's data."""


def is_valid_project_id(project_id: str) -> bool:
    # Ids are generated as uuid4().hex; anything else is rejected before it can
    # reach the filesystem (SECURITY.md §3; ids become file names).
    return bool(_PROJECT_ID_RE.match(project_id))


def _path(settings: Settings, project_id: str) -> Path:
    return settings.projects_dir / f"{project_id}.json"


def _read(path: Path) -> ProjectSummary | None:
    try:
        return ProjectSummary.model_validate(json.loads(path.read_text()))
    except (OSError, ValueError) as exc:  # unreadable or malformed; surface, don't crash
        log.warning("skipping unreadable project file %s: %s", path.name, exc)
        return None


def _write(settings: Settings, project: ProjectSummary) -> None:
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a crash mid-write can't leave a truncated file.
    path = _path(settings, project.id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(project.model_dump_json(indent=2))
    tmp.replace(path)


def _validate_metadata(name: str, description: str) -> tuple[str, str]:
    """Returns the normalized (name, description), or raises ProjectError."""
    name = name.strip()
    description = description.strip()
    if not name:
        raise ProjectError("project name must not be empty")
    if len(name) > MAX_NAME_CHARS:
        raise ProjectError(f"project name is limited to {MAX_NAME_CHARS} characters")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise ProjectError(
            f"project description is limited to {MAX_DESCRIPTION_CHARS} characters"
        )
    return name, description


def _now() -> str:
    # Fixed-width microseconds: isoformat() omits the field when it is zero,
    # which would break the lexicographic "oldest first" ordering.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def list_projects(settings: Settings) -> list[ProjectSummary]:
    """All projects, oldest first.

    Deliberate CQS exception: a dataset with no project yet is materialized
    into one here (a lazy, idempotent migration of pre-projects data), so the
    read has a write side effect. Serialized by the module lock so two
    concurrent lists cannot mint two projects for the same dataset.
    """
    with _lock:
        projects = []
        if settings.projects_dir.is_dir():
            projects = [
                project
                for path in sorted(settings.projects_dir.glob("*.json"))
                if (project := _read(path)) is not None
            ]

        linked = {ds for project in projects for ds in project.datasets}
        for dataset in datasets_service.list_datasets(settings):
            if dataset.id not in linked:
                projects.append(_materialize_dataset(settings, dataset.id))

    return sorted(projects, key=lambda project: (project.created_at, project.id))


def _materialize_dataset(settings: Settings, dataset_id: str) -> ProjectSummary:
    """A project file for a dataset that predates the projects layer."""
    project = ProjectSummary(
        id=uuid.uuid4().hex,
        name=dataset_id,
        description=_LEGACY_DESCRIPTION,
        datasets=[dataset_id],
        created_at=_now(),
    )
    _write(settings, project)
    log.info("materialized project %s for legacy dataset %s", project.id, dataset_id)
    return project


def get_project(settings: Settings, project_id: str) -> ProjectSummary | None:
    if not is_valid_project_id(project_id):
        return None
    path = _path(settings, project_id)
    return _read(path) if path.is_file() else None


def create_project(settings: Settings, name: str, description: str = "") -> ProjectSummary:
    """A new empty project: identity + metadata, no data attached yet."""
    name, description = _validate_metadata(name, description)
    project = ProjectSummary(
        id=uuid.uuid4().hex,
        name=name,
        description=description,
        created_at=_now(),
    )
    with _lock:
        _write(settings, project)
    return project


def update_project(
    settings: Settings, project_id: str, payload: ProjectUpdate
) -> ProjectSummary | None:
    """Apply the payload's explicitly-set fields; returns None if unknown.

    Only fields the client actually sent change, so `{"datasets": null}`
    clears the series list while an omitted field is left alone.
    """
    fields = payload.model_dump(exclude_unset=True)
    if fields.get("name", "") is None:
        raise ProjectError("project name must not be empty")
    if fields.get("description", "") is None:
        fields["description"] = ""  # explicit null clears the description
    if "datasets" in fields and fields["datasets"] is None:
        fields["datasets"] = []  # explicit null clears the series list
    if "datasets" in fields:
        # Order-preserving dedup with a sanity cap, mirroring the metadata limits.
        fields["datasets"] = list(dict.fromkeys(fields["datasets"]))
        if len(fields["datasets"]) > MAX_DATASETS:
            raise ProjectError(f"projects are limited to {MAX_DATASETS} series")

    for dataset in fields.get("datasets") or []:
        if datasets_service.get_dataset_summary(settings, dataset) is None:
            raise ProjectError(f"dataset {dataset!r} does not exist")

    with _lock:
        project = get_project(settings, project_id)
        if project is None:
            return None
        name, description = _validate_metadata(
            fields.get("name", project.name), fields.get("description", project.description)
        )
        updated = project.model_copy(
            update={**fields, "name": name, "description": description}
        )
        _write(settings, updated)
    return updated


def _datasets_of_other_projects(settings: Settings, exclude_id: str) -> set[str]:
    """Every dataset referenced by a project other than `exclude_id`.

    Reads the project files directly (no materialization), so it is safe to call
    while already holding `_lock` — unlike `list_projects`, which takes it.
    """
    shared: set[str] = set()
    if settings.projects_dir.is_dir():
        for path in settings.projects_dir.glob("*.json"):
            if path.stem == exclude_id:
                continue
            other = _read(path)
            if other is not None:
                shared.update(other.datasets)
    return shared


def _try_remove(
    delete: Callable[[Settings, str], Any], settings: Settings, target: str, *, label: str
) -> None:
    """Run one filesystem delete, logging (not raising) if it fails, so a single
    unremovable resource cannot abort the rest of a project's cascade."""
    try:
        delete(settings, target)
    except OSError as exc:  # permission, file held open, races with a writer
        log.error("could not remove %s while deleting project: %s", label, exc)


def delete_project(settings: Settings, project_id: str) -> ProjectSummary | None:
    """Delete a project and the data and outputs it exclusively owns.

    A dataset still referenced by another project — and that dataset's runs —
    is preserved; only series this project alone holds have their raw and
    processed directories and their runs removed. Returns the deleted project,
    or None if the id is unknown (mirrors `update_project`, so the route can 404).

    Raises `ProjectInUseError` if a training run is live on one of the owned
    datasets: deleting its tensors mid-run would corrupt the run (and the worker
    would recreate the "deleted" output dir). Cancel the run first.

    The lock is held across the filesystem cascade on purpose: it stops a
    concurrent `update_project` from re-attaching an owned dataset to another
    project between the ownership check and the delete. With a single training
    slot and few projects the cascade is brief, so the availability cost is
    acceptable; the alternative (I/O outside the lock) reopens that race.
    """
    with _lock:
        project = get_project(settings, project_id)
        if project is None:
            return None
        shared = _datasets_of_other_projects(settings, project_id)
        owned = [dataset for dataset in project.datasets if dataset not in shared]

        in_use = sorted(set(owned) & run_manager.active_datasets())
        if in_use:
            raise ProjectInUseError(
                f"a training run is in progress on {in_use[0]!r}; "
                "cancel it before deleting this project"
            )

        # Best-effort: one stubborn file must not abort the cascade and leave the
        # project record pointing at half-deleted data. Log what could not be
        # removed, then always drop the record last.
        for run in runs_service.list_runs(settings):
            if run.dataset in owned:
                _try_remove(runs_service.delete_run, settings, run.id, label=f"run {run.id}")
        for dataset in owned:
            _try_remove(
                datasets_service.delete_dataset, settings, dataset, label=f"dataset {dataset}"
            )
        _path(settings, project_id).unlink(missing_ok=True)
    log.info("deleted project %s (%d owned dataset(s) removed)", project_id, len(owned))
    return project
