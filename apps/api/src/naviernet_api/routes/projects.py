"""Project endpoints: list, create, and edit the workspace's projects."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from naviernet_api.models import ProjectCreate, ProjectSummary, ProjectUpdate
from naviernet_api.services import projects as projects_service
from naviernet_api.services.projects import ProjectError
from naviernet_api.settings import Settings, get_settings

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=list[ProjectSummary])
def list_projects(settings: Settings = Depends(get_settings)) -> list[ProjectSummary]:
    """Every project, oldest first. Legacy datasets appear as projects too."""
    return projects_service.list_projects(settings)


@router.post("", response_model=ProjectSummary, status_code=201)
def create_project(
    payload: ProjectCreate, settings: Settings = Depends(get_settings)
) -> ProjectSummary:
    """Create an empty project: a new uuid plus the given name/description."""
    try:
        return projects_service.create_project(settings, payload.name, payload.description)
    except ProjectError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{project_id}", response_model=ProjectSummary)
def get_project(project_id: str, settings: Settings = Depends(get_settings)) -> ProjectSummary:
    """One project by id."""
    project = projects_service.get_project(settings, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"project {project_id!r} not found")
    return project


@router.patch("/{project_id}", response_model=ProjectSummary)
def update_project(
    project_id: str, payload: ProjectUpdate, settings: Settings = Depends(get_settings)
) -> ProjectSummary:
    """Edit a project's metadata, or attach (`"dataset": id`) / detach
    (`"dataset": null`) its dataset. Omitted fields are left unchanged."""
    try:
        updated = projects_service.update_project(settings, project_id, payload)
    except ProjectError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail=f"project {project_id!r} not found")
    return updated
