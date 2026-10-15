"""The run manifest: `outputs/<run_id>/run.json`.

Every artifact a run writes (checkpoint, metrics, figures, the config snapshot)
is a *product* of training. The manifest is the one file that records the run's
IDENTITY instead: which project owns it, which series it spans, and when it
started. It is written before the first training step, so a run that fails
mid-way still knows what it was.

Why it exists: project membership used to be joined through the dataset name
every time the runs list was read, which made it retroactive (re-attaching a
series to a second project moved that series' entire run history with it) and
lossy (a run whose dataset could not be identified belonged to no project at
all). A stored id decides it once, at launch.

Read failures are never fatal: a missing, unreadable, or future-versioned
manifest degrades to None and the caller falls back to the dataset join, which
is exactly how every run written before this file is handled.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from naviernet.utils.logging import get_logger
from naviernet_api.models import RunLaunchRequest, RunManifest
from naviernet_api.settings import Settings

log = get_logger(__name__)

MANIFEST_FILE = "run.json"

# The version this module writes. A file claiming a HIGHER version was written by
# a newer server against a layout this one does not know, so it is ignored rather
# than half-read. v2 added `config_sha256`, the digest of the config snapshot,
# so a run is matched to its exact inputs.
SCHEMA_VERSION = 2

# Dataset names become path segments (`data/processed/<name>`), so they carry the
# same allowlist the run and dataset ids do (SECURITY.md §3). The manifest is
# first-party, but its dataset list originates in a launch request.
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(
    output_dir: Path,
    run_id: str,
    request: RunLaunchRequest,
    *,
    config_sha256: str | None = None,
) -> RunManifest:
    """Record the run's identity beside its artifacts, and return it: the
    project the launch resolved, the series the request spans, and the digest
    of the inputs it composed to.

    Write-then-rename, like the project files: a crash mid-write cannot leave a
    truncated manifest that would read as a run belonging nowhere.
    """
    manifest = RunManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        project_id=request.project,
        datasets=request.resolved_datasets(),
        heldout_datasets=list(request.heldout_datasets or []),
        created_at=_now(),
        config_sha256=config_sha256,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(manifest.model_dump_json(indent=2))
    tmp.replace(path)
    log.info("wrote manifest for run %s (project %s)", run_id, request.project or "-")
    return manifest


def read(run_dir: Path) -> RunManifest | None:
    """The run's manifest, or None when it has none this reader can trust."""
    path = run_dir / MANIFEST_FILE
    if not path.is_file():
        return None  # a run from before the manifest; the caller falls back
    try:
        manifest = RunManifest.model_validate(json.loads(path.read_text()))
    except (OSError, ValueError) as exc:
        log.warning("ignoring unreadable manifest in %s: %s", run_dir.name, exc)
        return None
    if manifest.schema_version > SCHEMA_VERSION:
        log.warning(
            "ignoring manifest in %s: schema v%d is newer than v%d",
            run_dir.name,
            manifest.schema_version,
            SCHEMA_VERSION,
        )
        return None
    unsafe = [name for name in manifest.datasets if not _NAME_RE.match(name)]
    if unsafe:
        log.warning("ignoring manifest in %s: unsafe dataset name %r", run_dir.name, unsafe[0])
        return None
    return manifest


def project_for_datasets(settings: Settings, datasets: list[str]) -> str | None:
    """The project a run over these datasets belongs to, or None if undecidable.

    Undecidable means either nothing references the datasets or several projects
    do -- a joint run spanning two projects' series has no single owner, and
    guessing one would be worse than recording none. Callers that need certainty
    send the project explicitly.

    `list_projects` is used rather than a raw directory scan so a series that
    predates the projects layer is materialized into its project first; without
    that, the very first run on such a series would record no owner.
    """
    if not datasets:
        return None
    # Imported here, not at module scope: services.projects imports run_manager,
    # which imports this module (same cycle-avoiding idiom as runs._live_state).
    from naviernet_api.services import projects as projects_service

    wanted = set(datasets)
    owners = {
        project.id
        for project in projects_service.list_projects(settings)
        if wanted & set(project.datasets)
    }
    if len(owners) == 1:
        return owners.pop()
    if owners:
        log.info("datasets %s span %d projects; recording no owner", datasets, len(owners))
    return None
