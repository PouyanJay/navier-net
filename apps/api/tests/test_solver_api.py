"""Launching, streaming, and resuming training runs over the API.

These are integration tests: a POST really composes a config and trains the
real PINN for a couple of steps on the synthetic tensors the fixtures stage, and
the SSE stream is read to completion against the live background thread.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# A deliberately tiny run: 2 steps over small batches finishes in ~a second.
TINY_RUN = {
    "dataset": "highest_t",
    "steps": 2,
    "n_data": 16,
    "n_coll": 16,
    "n_bc": 8,
    "log_every": 10,
    "render": False,
}


def read_stream(client: TestClient, run_id: str) -> list[dict]:
    """Read the run's SSE stream to completion into (event, data) records.

    Minimal parser: assumes single-line `data:` payloads (what sse-starlette
    emits for our compact JSON events); multi-line data would need joining.
    """
    events: list[dict] = []
    name = None
    with client.stream("GET", f"/api/runs/{run_id}/stream") as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and name is not None:
                events.append({"event": name, "data": json.loads(line.split(":", 1)[1])})
                name = None
    return events


def test_launch_trains_evaluates_and_streams(client: TestClient, repo_root: Path):
    """The walking skeleton: POST → background train+evaluate → SSE → artifacts."""
    response = client.post("/api/runs", json=TINY_RUN)
    assert response.status_code == 202
    launched = response.json()
    run_id = launched["run_id"]
    assert launched["state"] == "running"
    assert launched["dataset"] == "highest_t"

    # Reading the stream to completion IS the wait for the background thread.
    events = read_stream(client, run_id)
    by_name = {}
    for event in events:
        by_name.setdefault(event["event"], []).append(event["data"])

    final = by_name["status"][-1]
    assert final["state"] == "done", f"run failed: {final.get('message')}"
    assert final["steps_done"] == final["steps_total"] == 2

    # Live loss records flowed while training (not read back from the checkpoint).
    hist = by_name["hist"]
    assert len(hist) >= 1
    assert {"step", "lr", "data", "vof", "div", "src", "bc"} <= set(hist[0])

    # Console lines flowed too — both the manager's and the pipeline's own.
    lines = [record["line"] for record in by_name["log"]]
    assert any("starting run" in line for line in lines)
    assert any("training steps" in line for line in lines)

    # The run is now a first-class run: listed, detailed, and on disk.
    run_dir = repo_root / "outputs" / run_id
    assert (run_dir / "checkpoints" / "ckpt.pt").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / ".hydra" / "config.yaml").is_file()
    # The streamed console is also persisted as the run's transcript.
    assert "training steps" in (run_dir / "solver_console.log").read_text()
    listed = {run["id"]: run for run in client.get("/api/runs").json()}
    assert listed[run_id]["status"] == "trained"
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["steps"] == 2
    assert detail["config"]["training"]["steps"] == 2


def test_resume_continues_from_the_checkpoint(client: TestClient, repo_root: Path):
    """Resuming an API-launched run adds steps onto its existing checkpoint."""
    run_id = client.post("/api/runs", json=TINY_RUN).json()["run_id"]
    read_stream(client, run_id)  # drain = wait for completion

    response = client.post(
        "/api/runs", json={"resume": True, "run_id": run_id, "steps": 2, "render": False}
    )
    assert response.status_code == 202
    resumed = response.json()
    assert resumed["run_id"] == run_id
    assert resumed["dataset"] == "highest_t"

    events = read_stream(client, run_id)
    final = [e["data"] for e in events if e["event"] == "status"][-1]
    assert final["state"] == "done", f"resume failed: {final.get('message')}"
    # 2 original + 2 resumed steps, visible in the status and the checkpoint.
    assert final["steps_total"] == 4
    assert client.get(f"/api/runs/{run_id}").json()["steps"] == 4
    # The resumed session picked up training where the checkpoint left off.
    lines = [e["data"]["line"] for e in events if e["event"] == "log"]
    assert any("resuming run" in line for line in lines)
    assert any("training steps 3-4" in line for line in lines)


def test_holdout_none_trains_on_all_frames(client: TestClient, repo_root: Path):
    """holdout_frame=-1 supervises every frame; the holdout metric is absent."""
    run_id = client.post("/api/runs", json={**TINY_RUN, "holdout_frame": -1}).json()["run_id"]
    events = read_stream(client, run_id)
    final = [e["data"] for e in events if e["event"] == "status"][-1]
    assert final["state"] == "done", f"run failed: {final.get('message')}"
    metrics = client.get(f"/api/runs/{run_id}").json()["metrics"]
    assert metrics["iou_holdout"] is None
    assert metrics["holdout_frame"] is None


def test_launch_is_rejected_while_a_run_is_active(client: TestClient):
    """One training run at a time: a second launch is refused with 409."""
    from naviernet_api.services import run_manager

    # Seed a running job directly (the established registry-test idiom) — no
    # thread, so nothing races the autouse registry-clearing fixture.
    run_manager._jobs["existing-run"] = run_manager._RunJob(dataset="highest_t")

    second = client.post("/api/runs", json=TINY_RUN)
    assert second.status_code == 409
    assert "already in progress" in second.json()["detail"]

    active = client.get("/api/runs/active").json()
    assert active is not None and active["run_id"] == "existing-run"


def test_a_failing_run_reports_error_over_the_stream(client: TestClient, repo_root: Path):
    """A worker failure surfaces as state=error with a message, not a hang."""
    broken = repo_root / "data" / "processed" / "broken"
    broken.mkdir(parents=True)
    (broken / "tensors.npz").write_bytes(b"not an archive")

    run_id = client.post("/api/runs", json={**TINY_RUN, "dataset": "broken"}).json()["run_id"]
    events = read_stream(client, run_id)
    final = [e["data"] for e in events if e["event"] == "status"][-1]
    assert final["state"] == "error"
    assert final["message"]
    assert client.get(f"/api/runs/{run_id}/status").json()["state"] == "error"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param({**TINY_RUN, "steps": 0}, 422, id="steps-below-range"),
        pytest.param({**TINY_RUN, "steps": 100_000}, 422, id="steps-above-range"),
        pytest.param({**TINY_RUN, "lr": 0}, 422, id="lr-must-be-positive"),
        pytest.param({**TINY_RUN, "weights": {"data": -1}}, 422, id="negative-weight"),
        pytest.param({**TINY_RUN, "dataset": None}, 422, id="new-run-needs-dataset"),
        pytest.param({"resume": True, "steps": 2}, 422, id="resume-needs-run-id"),
        pytest.param({**TINY_RUN, "dataset": "../evil"}, 404, id="traversal-shaped-dataset"),
        pytest.param({**TINY_RUN, "dataset": "."}, 404, id="dot-dataset"),
        pytest.param({**TINY_RUN, "dataset": "sample"}, 409, id="not-preprocessed"),
        pytest.param(
            {"resume": True, "run_id": "no-such-run", "steps": 2}, 409, id="resume-unknown-run"
        ),
        pytest.param({"resume": True, "run_id": ".", "steps": 2}, 409, id="resume-dot-run-id"),
        pytest.param(
            {"resume": True, "run_id": "scratch", "steps": 2}, 409, id="resume-no-checkpoint"
        ),
    ],
)
def test_launch_rejections(client: TestClient, payload: dict, expected: int):
    """Bounds and preconditions reject bad requests with the right status."""
    response = client.post("/api/runs", json=payload)
    assert response.status_code == expected


def test_stream_and_status_unknown_run(client: TestClient):
    """Streams and statuses exist only for runs this server launched."""
    assert client.get("/api/runs/no-such-run/status").status_code == 404
    assert client.get("/api/runs/no-such-run/stream").status_code == 404
    assert client.get("/api/runs/active").json() is None


def test_stream_replays_fully_after_the_run_finished(client: TestClient):
    """A late subscriber still gets the whole story (events are replayed)."""
    run_id = client.post("/api/runs", json=TINY_RUN).json()["run_id"]
    read_stream(client, run_id)  # first reader drains to completion

    events = read_stream(client, run_id)  # late join: full replay, then EOF
    names = {event["event"] for event in events}
    assert {"status", "hist", "log"} <= names
    assert [e["data"] for e in events if e["event"] == "status"][-1]["state"] == "done"
