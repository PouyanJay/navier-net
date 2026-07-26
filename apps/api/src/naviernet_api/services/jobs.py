"""Driving the preprocess stage in the background.

A minimal in-process job registry: one preprocess job per dataset, run on a
daemon thread, its state polled over HTTP. Preprocess is quick (seconds), so no
streaming is needed; this is the seed of the fuller run manager Phase 4 adds for
training.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from naviernet.utils.logging import get_logger
from naviernet_api.models import PreprocessStatus
from naviernet_api.services import datasets as datasets_service
from naviernet_api.settings import Settings

log = get_logger(__name__)


@dataclass
class _Job:
    state: str = "running"
    message: str | None = None


_jobs: dict[str, _Job] = {}
_lock = threading.Lock()


def _has_qc(settings: Settings, dataset: str) -> bool:
    return datasets_service.qc_path(settings, dataset) is not None


def status(settings: Settings, dataset: str) -> PreprocessStatus:
    """Current preprocessing state for a dataset."""
    with _lock:
        job = _jobs.get(dataset)
    return PreprocessStatus(
        dataset=dataset,
        state=job.state if job else "idle",
        message=job.message if job else None,
        has_qc=_has_qc(settings, dataset),
    )


def start_preprocess(settings: Settings, dataset: str) -> PreprocessStatus:
    """Kick off preprocessing in the background (idempotent while running)."""
    with _lock:
        job = _jobs.get(dataset)
        already_running = job is not None and job.state == "running"
        if not already_running:
            _jobs[dataset] = _Job(state="running")
    # Spawn and report OUTSIDE the lock: status() re-acquires it, and the lock is
    # not reentrant.
    if not already_running:
        thread = threading.Thread(
            target=_run, args=(settings, dataset), name=f"preprocess-{dataset}", daemon=True
        )
        thread.start()
    return status(settings, dataset)


def _run(settings: Settings, dataset: str) -> None:
    """Background worker: compose the config and run the preprocess stage."""
    try:
        from naviernet.pipeline import Pipeline
        from naviernet_api.services.config_service import compose_cfg
        from naviernet_api.services.datasets import series_overrides

        # paths.root is pinned to the repo so data/ and outputs/ resolve
        # regardless of the server's working directory. The series' saved
        # conditions and frame exclusions apply here too, so preprocessing sees
        # its real Δt and drops the frames the user marked.
        cfg = compose_cfg(
            dataset,
            overrides=[
                f"paths.root={settings.repo_root}",
                *series_overrides(settings, dataset),
            ],
        )
        Pipeline(cfg).preprocess()
        with _lock:
            _jobs[dataset] = _Job(state="done")
        log.info("preprocess done for %s", dataset)
    except Exception as exc:  # noqa: BLE001 (report any failure back to the client)
        log.exception("preprocess failed for %s", dataset)
        with _lock:
            _jobs[dataset] = _Job(state="error", message=str(exc))
