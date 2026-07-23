"""Seed sweeps over the API: sequential children, slot ownership, failure abort.

Integration tests: a sweep really trains its children (2 steps each) on the
synthetic tensors and each child is a first-class run afterwards.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from helpers import TINY_RUN, final_status, read_stream

TINY_SWEEP = {**TINY_RUN, "seeds": [0, 1]}


def test_sweep_trains_children_sequentially(client: TestClient, repo_root: Path):
    """POST /api/sweeps → both children train+evaluate → both are real runs."""
    response = client.post("/api/sweeps", json=TINY_SWEEP)
    assert response.status_code == 202
    sweep = response.json()
    sweep_id = sweep["sweep_id"]
    child_ids = [child["run_id"] for child in sweep["children"]]
    assert child_ids == [f"{sweep_id}-s0", f"{sweep_id}-s1"]

    # Draining each child's stream in order IS the wait for the sweep.
    for child_id in child_ids:
        events = read_stream(client, child_id)
        final = final_status(events)
        assert final["state"] == "done", f"{child_id} failed: {final.get('message')}"

    finished = client.get(f"/api/sweeps/{sweep_id}").json()
    assert finished["state"] == "done"
    assert [c["state"] for c in finished["children"]] == ["done", "done"]

    # Each child is an ordinary, comparable run: listed, evaluated, seeded.
    listed = {run["id"] for run in client.get("/api/runs").json()}
    assert set(child_ids) <= listed
    for child_id, seed in zip(child_ids, (0, 1), strict=True):
        detail = client.get(f"/api/runs/{child_id}").json()
        assert detail["steps"] == 2
        assert detail["metrics"] is not None  # evaluate ran → comparison data
        assert detail["config"]["training"]["seed"] == seed
        history = client.get(f"/api/runs/{child_id}/loss-history").json()
        assert len(history) >= 1


def test_sweep_owns_the_slot_end_to_end(client: TestClient):
    """While a sweep is queued/running, ordinary launches are refused."""
    from naviernet_api.services import run_manager

    # A reserved (queued) child claims the slot even before it starts training.
    run_manager._jobs["some-sweep-s0"] = run_manager._RunJob(
        dataset="highest_t", state="queued"
    )

    run = client.post("/api/runs", json=TINY_RUN)
    assert run.status_code == 409
    sweep = client.post("/api/sweeps", json=TINY_SWEEP)
    assert sweep.status_code == 409


def test_sweep_child_failure_aborts_the_rest(client: TestClient, repo_root: Path):
    """A failing child marks the sweep error and the remaining children abort."""
    broken = repo_root / "data" / "processed" / "broken"
    broken.mkdir(parents=True)
    (broken / "tensors.npz").write_bytes(b"not an archive")

    sweep = client.post("/api/sweeps", json={**TINY_SWEEP, "dataset": "broken"}).json()
    first, second = [child["run_id"] for child in sweep["children"]]

    events = read_stream(client, first)
    assert final_status(events)["state"] == "error"
    # The second child never runs; its stream replays a terminal abort status.
    aborted = final_status(read_stream(client, second))
    assert aborted["state"] == "error"
    assert "aborted" in (aborted["message"] or "")

    status = client.get(f"/api/sweeps/{sweep['sweep_id']}").json()
    assert status["state"] == "error"
    assert status["message"]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param({**TINY_SWEEP, "seeds": [0, 0]}, 422, id="duplicate-seeds"),
        pytest.param({**TINY_SWEEP, "seeds": []}, 422, id="no-seeds"),
        pytest.param({**TINY_SWEEP, "seeds": [0, 1, 2, 3, 4, 5, 6]}, 422, id="too-many-seeds"),
        pytest.param({**TINY_SWEEP, "seeds": [-1]}, 422, id="negative-seed"),
        pytest.param(
            {**TINY_SWEEP, "resume": True, "run_id": "x"}, 422, id="sweep-cannot-resume"
        ),
        pytest.param({**TINY_SWEEP, "dataset": "nope"}, 409, id="not-preprocessed"),
        pytest.param({**TINY_SWEEP, "dataset": "../evil"}, 404, id="traversal-dataset"),
    ],
)
def test_sweep_rejections(client: TestClient, payload: dict, expected: int):
    """Bounds and preconditions reject bad sweep requests with the right status."""
    response = client.post("/api/sweeps", json=payload)
    assert response.status_code == expected


def test_unknown_sweep_and_active(client: TestClient):
    assert client.get("/api/sweeps/no-such-sweep").status_code == 404
    assert client.get("/api/sweeps/active").json() is None
