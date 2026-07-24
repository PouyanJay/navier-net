"""Launching and tracking training runs in the background.

The fuller sibling of `jobs.py` (which drives the quick preprocess stage): a
training run takes minutes-to-hours, so each launched run gets a registry entry
holding its live state plus an append-only event buffer (loss records, console
lines, status transitions) that the SSE endpoint replays and follows. The
registry is process-local; the filesystem stays the source of truth for the
run's artifacts, and a finished run is served from disk like any other.

Lock discipline (same as `jobs.py`): the lock is not reentrant, so state is
set inside it and threads are spawned / statuses reported outside it.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from omegaconf import DictConfig

from naviernet.utils.logging import get_logger
from naviernet_api.models import RunJobStatus, RunLaunchRequest
from naviernet_api.services import datasets as datasets_service
from naviernet_api.services import runs as runs_service
from naviernet_api.settings import Settings

log = get_logger(__name__)

# Training is CPU-bound and long-running; one run at a time keeps the box
# responsive and the console unambiguous.
MAX_CONCURRENT_RUNS = 1

# Finished registry entries kept for late SSE joins. Older runs are evicted;
# their artifacts (checkpoint, metrics, solver_console.log) live on disk. This
# bounds the process's memory across many launches: each job's event buffer is
# itself bounded (log_every >= 10 over steps <= 20000).
MAX_FINISHED_JOBS = 8

_RunState = Literal["queued", "running", "done", "error"]
_TERMINAL_STATES = ("done", "error")


class LaunchRejected(Exception):
    """A launch request that is well-formed but cannot be honored."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class _RunJob:
    dataset: str | None
    state: _RunState = "running"
    stage: str | None = None
    message: str | None = None
    steps_done: int = 0
    steps_total: int = 0
    events: list[dict] = field(default_factory=list)


_jobs: dict[str, _RunJob] = {}
_lock = threading.Lock()


class _ConsoleHandler(logging.Handler):
    """Mirrors the worker thread's pipeline log lines into the run's events.

    Filtering on the emitting thread id gives an exact per-run console: other
    threads (request handlers, a preprocess job) share the `naviernet` loggers
    but never leak into this run's stream.
    """

    def __init__(self, run_id: str, thread_id: int):
        super().__init__(level=logging.INFO)
        self._run_id = run_id
        self._thread_id = thread_id

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self._thread_id:
            return
        tone = "err" if record.levelno >= logging.WARNING else None
        _append_event(self._run_id, "log", {"line": record.getMessage(), "tone": tone})


# -- registry access --------------------------------------------------------


def _append_event(run_id: str, name: str, data: dict) -> None:
    with _lock:
        job = _jobs.get(run_id)
        if job is not None:
            job.events.append({"event": name, "data": data})


def _status_snapshot(run_id: str, job: _RunJob) -> RunJobStatus:
    return RunJobStatus(
        run_id=run_id,
        dataset=job.dataset,
        state=job.state,
        stage=job.stage,
        message=job.message,
        steps_done=job.steps_done,
        steps_total=job.steps_total,
    )


def _emit_status(run_id: str) -> None:
    """Append the job's current status to its own event stream."""
    with _lock:
        job = _jobs.get(run_id)
        if job is not None:
            job.events.append(
                {"event": "status", "data": _status_snapshot(run_id, job).model_dump()}
            )


def _console(run_id: str, line: str, tone: str | None = None) -> None:
    _append_event(run_id, "log", {"line": line, "tone": tone})


def status(run_id: str) -> RunJobStatus | None:
    """The live status of a run launched by this process, or None."""
    with _lock:
        job = _jobs.get(run_id)
        return _status_snapshot(run_id, job) if job else None


def active_run() -> RunJobStatus | None:
    """The currently running job, if any (there is at most one)."""
    with _lock:
        for run_id, job in _jobs.items():
            if job.state == "running":
                return _status_snapshot(run_id, job)
    return None


def events_since(run_id: str, cursor: int) -> tuple[list[dict], int, bool] | None:
    """Events after ``cursor`` plus the new cursor and whether the run is over.

    A job's state only turns terminal in the same lock acquisition that appends
    its final status event (`_finish` / `abort_queued`), so ``terminal=True``
    guarantees the final event is included in (or preceded) this batch; a
    consumer that drains until terminal never misses it.
    """
    with _lock:
        job = _jobs.get(run_id)
        if job is None:
            return None
        events = job.events[cursor:]
        new_cursor = len(job.events)
        terminal = job.state in _TERMINAL_STATES
    return events, new_cursor, terminal


# -- launching --------------------------------------------------------------


def launch(settings: Settings, request: RunLaunchRequest) -> RunJobStatus:
    """Validate a launch request, reserve the slot, and start the worker."""
    if request.resume:
        run_id, dataset = _validate_resume(settings, request)
        request.dataset = dataset  # resolved from the run's own artifacts
    else:
        run_id, dataset = None, _validate_new_run(settings, request)

    with _lock:
        if _slot_busy():
            raise LaunchRejected(409, "a training run is already in progress")
        if run_id is None:
            # Minting stats the filesystem while holding the lock; one .exists()
            # per launch, accepted so id reservation and registration are atomic.
            run_id = _mint_run_id(settings)
        _jobs[run_id] = _RunJob(dataset=dataset, stage="train")
        _evict_finished_jobs()
    # Spawn and report OUTSIDE the lock: both re-acquire it.
    thread = threading.Thread(
        target=_worker, args=(settings, run_id, request), name=f"train-{run_id}", daemon=True
    )
    thread.start()
    result = status(run_id)
    if result is None:  # unreachable: the job was registered above
        raise RuntimeError(f"run {run_id!r} vanished from the registry")
    return result


def _slot_busy() -> bool:
    """Whether the single training slot is claimed. Call under the lock.

    Queued jobs count: a sweep reserves all its children upfront so nothing
    can slip into the slot between two of its children.
    """
    return sum(1 for job in _jobs.values() if job.state in ("queued", "running")) >= (
        MAX_CONCURRENT_RUNS
    )


def _evict_finished_jobs() -> None:
    """Drop the oldest finished entries beyond the retention cap. Call under
    the lock; insertion order makes the registry oldest-first."""
    finished = [run_id for run_id, job in _jobs.items() if job.state in _TERMINAL_STATES]
    for run_id in finished[: max(0, len(finished) - MAX_FINISHED_JOBS)]:
        del _jobs[run_id]


# -- sweep support: reserved (queued) children run by the sweep's own thread --


def reserve_children(settings: Settings, child_ids: list[str], dataset: str) -> None:
    """Atomically claim the training slot with a sweep's children (queued).

    Registering every child upfront means the sweep owns the slot end-to-end;
    an unrelated launch cannot slip in between two children.
    """
    with _lock:
        if _slot_busy():
            raise LaunchRejected(409, "a training run is already in progress")
        for child_id in child_ids:
            _jobs[child_id] = _RunJob(dataset=dataset, state="queued")
        _evict_finished_jobs()


def execute_child(settings: Settings, run_id: str, request: RunLaunchRequest) -> RunJobStatus:
    """Run one reserved child to completion in the calling thread."""
    with _lock:
        job = _jobs.get(run_id)
        if job is None or job.state != "queued":
            raise RuntimeError(f"child {run_id!r} is not reserved")
        job.state = "running"
        job.stage = "train"
    _worker(settings, run_id, request)
    result = status(run_id)
    if result is None:  # unreachable: children are never evicted while queued/running
        raise RuntimeError(f"child {run_id!r} vanished from the registry")
    return result


def abort_queued(run_ids: list[str], message: str) -> None:
    """Flip still-queued children to error, each with its terminal status event."""
    for run_id in run_ids:
        with _lock:
            job = _jobs.get(run_id)
            if job is None or job.state != "queued":
                continue
            job.state = "error"
            job.message = message
            job.events.append(
                {"event": "status", "data": _status_snapshot(run_id, job).model_dump()}
            )


def validate_trainable_dataset(settings: Settings, dataset: str | None) -> str:
    """The dataset id, after checking a run could train on it."""
    dataset = dataset or ""
    if not datasets_service.is_valid_dataset_id(dataset):
        raise LaunchRejected(404, f"unknown dataset {dataset!r}")
    if datasets_service.tensors_path(settings, dataset) is None:
        raise LaunchRejected(
            409, f"dataset {dataset!r} is not preprocessed; run preprocessing first"
        )
    return dataset


def _validate_new_run(settings: Settings, request: RunLaunchRequest) -> str:
    """The dataset for a new run, after checking it is trainable."""
    return validate_trainable_dataset(settings, request.dataset)


def _validate_resume(settings: Settings, request: RunLaunchRequest) -> tuple[str, str | None]:
    """The (run_id, dataset) to resume, after checking it can be resumed."""
    run_id = request.run_id or ""
    if runs_service.checkpoint_path(settings, run_id) is None:
        raise LaunchRejected(409, f"run {run_id!r} has no checkpoint to resume from")
    if not _snapshot_path(settings, run_id).is_file():
        raise LaunchRejected(409, f"run {run_id!r} has no config snapshot; cannot resume")
    with _lock:
        job = _jobs.get(run_id)
        if job is not None and job.state == "running":
            raise LaunchRejected(409, f"run {run_id!r} is already running")
    found = runs_service.read_dataset_and_metrics(settings, run_id)
    dataset = found[0] if found else None
    return run_id, dataset


def _mint_run_id(settings: Settings) -> str:
    """A unique run id: training auto-resumes any existing checkpoint, so a new
    run must never land in an existing run's directory."""
    base = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    candidate, n = base, 2
    while candidate in _jobs or (settings.outputs_dir / candidate).exists():
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def _snapshot_path(settings: Settings, run_id: str) -> Path:
    return settings.outputs_dir / run_id / ".hydra" / "config.yaml"


# -- the worker -------------------------------------------------------------


def _set_stage(run_id: str, stage: str) -> None:
    with _lock:
        job = _jobs.get(run_id)
        if job is not None:
            job.stage = stage
    _emit_status(run_id)
    _console(run_id, f"[stage] {stage}", tone="em")


def _worker(settings: Settings, run_id: str, request: RunLaunchRequest) -> None:
    """Background worker: configure, train, evaluate, render, finalize."""
    from naviernet.pipeline import Pipeline

    handler = _ConsoleHandler(run_id, threading.get_ident())
    pipeline_logger = logging.getLogger("naviernet")
    pipeline_logger.addHandler(handler)
    try:
        cfg, base_steps = _configure(settings, run_id, request)
        _announce_start(run_id, base_steps, request)
        pipeline = Pipeline(cfg)
        if not request.resume:
            _write_snapshot(cfg, pipeline.paths.output_dir)
        _run_stages(pipeline, run_id, request)
        _finish(settings, run_id, error=None)
        log.info("run %s finished", run_id)
    except Exception as exc:  # noqa: BLE001 (report any failure to the client)
        log.exception("run %s failed", run_id)
        _finish(settings, run_id, error=exc)
    finally:
        pipeline_logger.removeHandler(handler)


def _announce_start(run_id: str, base_steps: int, request: RunLaunchRequest) -> None:
    """Record the step budget and open the console with the launch banner."""
    with _lock:
        job = _jobs.get(run_id)
        if job is None:
            return
        job.steps_done = base_steps
        job.steps_total = base_steps + request.steps
        dataset = job.dataset
    verb = "resuming" if request.resume else "starting"
    origin = f" · dataset {dataset}" if dataset else ""
    _console(run_id, f"[naviernet] {verb} run {run_id}{origin} · {request.steps} steps", "dim")
    _emit_status(run_id)


def _run_stages(pipeline, run_id: str, request: RunLaunchRequest) -> None:
    """Drive the pipeline stages, streaming loss records into the event buffer."""

    def on_hist(record: dict) -> None:
        with _lock:
            job = _jobs.get(run_id)
            if job is not None:
                job.steps_done = int(record["step"])
                job.events.append({"event": "hist", "data": record})

    pipeline.train(steps=request.steps, on_log=on_hist)
    _set_stage(run_id, "evaluate")
    pipeline.evaluate()
    if request.render:
        _set_stage(run_id, "figures")
        pipeline.figures()
        _set_stage(run_id, "video")
        pipeline.video()


def _finish(settings: Settings, run_id: str, error: Exception | None) -> None:
    """Finalize the run: transcript first, then the terminal state and status
    event under ONE lock acquisition, so a stream that observes a terminal
    state is guaranteed to have the final status event in its buffer."""
    if error is None:
        line = f"[naviernet] run complete; checkpoint at outputs/{run_id}/checkpoints/ckpt.pt"
        tone = "ok"
    else:
        line, tone = f"[naviernet] run failed: {error}", "err"
    _console(run_id, line, tone)
    _dump_console(settings, run_id)
    with _lock:
        job = _jobs.get(run_id)
        if job is None:
            return
        job.stage = None
        if error is None:
            job.state = "done"
            job.steps_done = job.steps_total  # hist only records every log_every steps
        else:
            job.state = "error"
            job.message = str(error)
        job.events.append(
            {"event": "status", "data": _status_snapshot(run_id, job).model_dump()}
        )


def _dump_console(settings: Settings, run_id: str) -> None:
    """Persist the session's console transcript into the run directory.

    CLI runs get a Hydra log file; this is the API-launched equivalent, so a
    run's transcript survives the process that streamed it. Appended per
    session, so a resumed run keeps its earlier transcript too.
    """
    with _lock:
        job = _jobs.get(run_id)
        lines = [e["data"]["line"] for e in job.events if e["event"] == "log"] if job else []
    run_dir = settings.outputs_dir / run_id
    if not lines or not run_dir.is_dir():
        return  # nothing to persist, or the run never got a directory
    try:
        with (run_dir / "solver_console.log").open("a") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as exc:
        log.warning("could not persist console for %s: %s", run_id, exc)


def _configure(
    settings: Settings, run_id: str, request: RunLaunchRequest
) -> tuple[DictConfig, int]:
    """The composed config and already-completed step count for this launch."""
    if request.resume:
        return _resume_config(settings, run_id, request)

    dataset = request.dataset
    if dataset is None:  # unreachable: enforced by the model validator + launch()
        raise RuntimeError("new run reached the worker without a dataset")
    overrides = [
        f"paths.root={settings.repo_root}",
        f"run_name={run_id}",
        f"training.steps={request.steps}",
        f"training.lr={request.lr}",
        f"training.lr_halflife={request.lr_halflife}",
        f"training.n_data={request.n_data}",
        f"training.n_coll={request.n_coll}",
        f"training.n_bc={request.n_bc}",
        f"training.holdout_frame={request.holdout_frame}",
        f"training.rebalance_every={request.rebalance_every}",
        f"training.log_every={request.log_every}",
        f"training.seed={request.seed}",
        "training.device=cpu",  # the server never schedules onto an accelerator
        f"training.weights.data={request.weights.data}",
        f"training.weights.vof={request.weights.vof}",
        f"training.weights.div={request.weights.div}",
        f"training.weights.src={request.weights.src}",
        f"training.weights.bc={request.weights.bc}",
    ]
    from naviernet_api.services.config_service import compose_cfg_once
    from naviernet_api.services.datasets import series_overrides

    # The series' saved conditions and frame exclusions travel with every run.
    overrides.extend(series_overrides(settings, dataset))
    return compose_cfg_once(dataset, overrides=overrides), 0


def _resume_config(
    settings: Settings, run_id: str, request: RunLaunchRequest
) -> tuple[DictConfig, int]:
    """Reload the run's own config so a resumed run keeps its architecture."""
    import torch
    from omegaconf import OmegaConf

    # Schema-merged and re-pinned to this repo; unreadable snapshots fail
    # loudly here, not mid-training.
    cfg = runs_service.load_run_config(settings, run_id)
    if cfg is None:
        raise RuntimeError(f"run {run_id!r} has no readable config snapshot")
    cfg.training.steps = request.steps  # the one value a resume overrides
    OmegaConf.set_readonly(cfg, True)  # match compose_cfg's contract

    checkpoint = settings.outputs_dir / run_id / "checkpoints" / "ckpt.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return cfg, int(state["state"]["done"])


def _write_snapshot(cfg, output_dir: Path) -> None:
    """Persist the composed config like Hydra does for CLI runs, so the run's
    detail view (and a later resume) can read it back."""
    from omegaconf import OmegaConf

    snapshot_dir = output_dir / ".hydra"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, snapshot_dir / "config.yaml")
