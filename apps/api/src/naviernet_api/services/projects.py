"""Project metadata: one JSON file per project under `projects/`.

A project is the platform's scoping unit — a uuid identity with an editable
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
from datetime import datetime, timezone
from pathlib import Path

from naviernet.utils.logging import get_logger
from naviernet_api.models import ProjectSummary, ProjectUpdate
from naviernet_api.services import datasets as datasets_service
from naviernet_api.settings import Settings

log = get_logger(__name__)

# Serializes every read-modify-write on the project files (same idiom as
# run_manager): concurrent PATCHes must not lose edits, and concurrent lists
# must not materialize the same legacy dataset twice.
_lock = threading.Lock()

_PROJECT_ID_RE = re.compile(r"^[0-9a-f]{32}$")  # uuid4().hex
MAX_NAME_CHARS = 120
MAX_DESCRIPTION_CHARS = 2000

# Purpose line for materialized legacy datasets (the platform's experiment).
_LEGACY_DESCRIPTION = (
    "Reconstruct the hidden velocity and volume-fraction fields of a confined "
    "vapor slug from its high-speed image sequence."
)


class ProjectError(ValueError):
    """A rejected project operation (bad name, unknown dataset, …)."""


def is_valid_project_id(project_id: str) -> bool:
    # Ids are generated as uuid4().hex; anything else is rejected before it can
    # reach the filesystem (SECURITY.md §3 — ids become file names).
    return bool(_PROJECT_ID_RE.match(project_id))


def _path(settings: Settings, project_id: str) -> Path:
    return settings.projects_dir / f"{project_id}.json"


def _read(path: Path) -> ProjectSummary | None:
    try:
        return ProjectSummary.model_validate(json.loads(path.read_text()))
    except (OSError, ValueError) as exc:  # unreadable or malformed — surface, don't crash
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

        linked = {project.dataset for project in projects if project.dataset}
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
        dataset=dataset_id,
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

    Only fields the client actually sent change, so `{"dataset": null}`
    detaches a dataset while an omitted field is left alone.
    """
    fields = payload.model_dump(exclude_unset=True)
    if fields.get("name", "") is None:
        raise ProjectError("project name must not be empty")
    if fields.get("description", "") is None:
        fields["description"] = ""  # explicit null clears the description

    dataset = fields.get("dataset")
    if dataset is not None and datasets_service.get_dataset_summary(settings, dataset) is None:
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
