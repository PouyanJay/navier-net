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
import uuid
from datetime import datetime, timezone
from pathlib import Path

from naviernet.utils.logging import get_logger
from naviernet_api.models import ProjectSummary
from naviernet_api.services import datasets as datasets_service
from naviernet_api.settings import Settings

log = get_logger(__name__)

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
    _path(settings, project.id).write_text(project.model_dump_json(indent=2))


def _validate_metadata(name: str, description: str) -> str:
    """Returns the normalized name, or raises ProjectError."""
    name = name.strip()
    if not name:
        raise ProjectError("project name must not be empty")
    if len(name) > MAX_NAME_CHARS:
        raise ProjectError(f"project name is limited to {MAX_NAME_CHARS} characters")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise ProjectError(
            f"project description is limited to {MAX_DESCRIPTION_CHARS} characters"
        )
    return name


def _now() -> str:
    # Microsecond precision: creation order stays stable even for projects
    # created back-to-back (the list endpoint sorts by this field).
    return datetime.now(timezone.utc).isoformat()


def list_projects(settings: Settings) -> list[ProjectSummary]:
    """All projects, materializing any dataset that has no project yet."""
    projects = []
    if settings.projects_dir.is_dir():
        projects = [
            p
            for path in sorted(settings.projects_dir.glob("*.json"))
            if (p := _read(path)) is not None
        ]

    linked = {p.dataset for p in projects if p.dataset}
    for dataset in datasets_service.list_datasets(settings):
        if dataset.id not in linked:
            projects.append(_materialize_dataset(settings, dataset.id))

    return sorted(projects, key=lambda p: (p.created_at, p.id))


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
    project = ProjectSummary(
        id=uuid.uuid4().hex,
        name=_validate_metadata(name, description),
        description=description,
        created_at=_now(),
    )
    _write(settings, project)
    return project


def update_project(
    settings: Settings,
    project_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    dataset: str | None = None,
) -> ProjectSummary | None:
    """Apply the given fields; returns None for an unknown project."""
    project = get_project(settings, project_id)
    if project is None:
        return None

    updated = project.model_copy(
        update={
            "name": name if name is not None else project.name,
            "description": description if description is not None else project.description,
            "dataset": dataset if dataset is not None else project.dataset,
        }
    )
    updated.name = _validate_metadata(updated.name, updated.description)
    if dataset is not None and datasets_service.get_dataset_summary(settings, dataset) is None:
        raise ProjectError(f"dataset {dataset!r} does not exist")

    _write(settings, updated)
    return updated
