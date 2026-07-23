"""Launching, streaming, and resuming training runs over the API.

These are integration tests: a POST really composes a config and trains the
real PINN for a couple of steps on the synthetic tensors the fixtures stage, and
the SSE stream is read to completion against the live background thread.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    """Read the run's SSE stream to completion into (event, data) records."""
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
    listed = {run["id"]: run for run in client.get("/api/runs").json()}
    assert listed[run_id]["status"] == "trained"
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["steps"] == 2
    assert detail["config"]["training"]["steps"] == 2
