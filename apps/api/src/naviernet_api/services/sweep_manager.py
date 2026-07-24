"""Driving a seed sweep as a sequence of managed runs.

A sweep is the API-side equivalent of the CLI's `--multirun training.seed=...`:
the same configuration composed once per seed, each child trained and evaluated
in order. Children are ordinary run_manager jobs with flat ids
(`<sweep_id>-s<seed>`), so every per-run endpoint — stream, status, detail,
loss history — works on them unchanged, and the comparison view reads them like
any other run. The sweep registry here only ties them together.

All children are reserved (queued) upfront, so the sweep owns the single
training slot end-to-end; the sweep's own thread then runs them one by one.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from naviernet.utils.logging import get_logger
from naviernet_api.models import RunJobStatus, SweepLaunchRequest, SweepStatus
from naviernet_api.services import run_manager
from naviernet_api.services import runs as runs_service
from naviernet_api.settings import Settings

log = get_logger(__name__)

# Finished sweep entries kept for the UI to re-read; children live in the
# run_manager registry / on disk regardless.
MAX_FINISHED_SWEEPS = 4

_SweepState = Literal["running", "done", "error"]


@dataclass
class _Sweep:
    dataset: str
    seeds: list[int]
    child_ids: list[str]
    state: _SweepState = "running"
    message: str | None = None


_sweeps: dict[str, _Sweep] = {}
_lock = threading.Lock()


def _evicted_child(settings: Settings, sweep: _Sweep, child_id: str) -> RunJobStatus:
    """A child that left the run registry, reconstructed from its artifacts.

    Only finished children are ever evicted (queued/running block eviction),
    so the checkpoint on disk decides between done and error — the sweep must
    never quietly shrink.
    """
    trained = runs_service.checkpoint_path(settings, child_id) is not None
    return RunJobStatus(
        run_id=child_id,
        dataset=sweep.dataset,
        state="done" if trained else "error",
        message=None if trained else "child finished without a checkpoint",
    )


def _snapshot(settings: Settings, sweep_id: str, sweep: _Sweep) -> SweepStatus:
    children = [
        run_manager.status(child_id) or _evicted_child(settings, sweep, child_id)
        for child_id in sweep.child_ids
    ]
    return SweepStatus(
        sweep_id=sweep_id,
        dataset=sweep.dataset,
        state=sweep.state,
        message=sweep.message,
        seeds=sweep.seeds,
        children=children,
    )


def status(settings: Settings, sweep_id: str) -> SweepStatus | None:
    """The live status of a sweep launched by this process, or None."""
    with _lock:
        sweep = _sweeps.get(sweep_id)
    return _snapshot(settings, sweep_id, sweep) if sweep else None


def active_sweep(settings: Settings) -> SweepStatus | None:
    """The currently running sweep, if any (there is at most one)."""
    with _lock:
        entry = next(
            ((sweep_id, s) for sweep_id, s in _sweeps.items() if s.state == "running"), None
        )
    return _snapshot(settings, *entry) if entry else None


def launch(settings: Settings, request: SweepLaunchRequest) -> SweepStatus:
    """Validate, reserve every child, and start the sweep worker."""
    dataset = run_manager.validate_trainable_dataset(settings, request.dataset)
    # Minting and registration are not one critical section: two same-second
    # launches could mint the same id, but the loser is rejected by
    # reserve_children's single-slot check before it ever touches _sweeps.
    with _lock:
        sweep_id = _mint_sweep_id(settings, request.seeds)
    child_ids = [f"{sweep_id}-s{seed}" for seed in request.seeds]

    run_manager.reserve_children(settings, child_ids, dataset)
    with _lock:
        _sweeps[sweep_id] = _Sweep(
            dataset=dataset, seeds=list(request.seeds), child_ids=child_ids
        )
        _evict_finished_sweeps()

    thread = threading.Thread(
        target=_worker, args=(settings, sweep_id, request), name=sweep_id, daemon=True
    )
    thread.start()
    result = status(settings, sweep_id)
    if result is None:  # unreachable: the sweep was registered above
        raise RuntimeError(f"sweep {sweep_id!r} vanished from the registry")
    return result


def _worker(settings: Settings, sweep_id: str, request: SweepLaunchRequest) -> None:
    """Run the sweep's children sequentially; abort the rest on a failure."""
    with _lock:
        sweep = _sweeps[sweep_id]
    try:
        for seed, child_id in zip(sweep.seeds, sweep.child_ids, strict=True):
            child_request = request.model_copy(update={"seed": seed})
            final = run_manager.execute_child(settings, child_id, child_request)
            if final.state != "done":
                raise RuntimeError(f"child {child_id} failed: {final.message}")
        with _lock:
            sweep.state = "done"
        log.info("sweep %s finished (%d children)", sweep_id, len(sweep.child_ids))
    except Exception as exc:  # noqa: BLE001 — report any failure to the client
        log.exception("sweep %s failed", sweep_id)
        with _lock:
            sweep.state = "error"
            sweep.message = str(exc)
        run_manager.abort_queued(sweep.child_ids, f"sweep {sweep_id} aborted: {exc}")


def _mint_sweep_id(settings: Settings, seeds: list[int]) -> str:
    """A sweep id whose child directories don't collide with existing runs."""
    base = datetime.now().strftime("sweep-%Y%m%d-%H%M%S")
    candidate, n = base, 2
    while candidate in _sweeps or any(
        (settings.outputs_dir / f"{candidate}-s{seed}").exists() for seed in seeds
    ):
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def _evict_finished_sweeps() -> None:
    """Drop the oldest finished sweeps beyond the retention cap. Call under
    the lock; insertion order makes the registry oldest-first."""
    finished = [sweep_id for sweep_id, sweep in _sweeps.items() if sweep.state != "running"]
    for sweep_id in finished[: max(0, len(finished) - MAX_FINISHED_SWEEPS)]:
        del _sweeps[sweep_id]
