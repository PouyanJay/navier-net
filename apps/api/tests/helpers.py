"""Shared helpers for the solver/sweep API tests."""

from __future__ import annotations

import json

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


def final_status(events: list[dict]) -> dict:
    """The last status payload in a drained event stream."""
    return [e["data"] for e in events if e["event"] == "status"][-1]
