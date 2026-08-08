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
from helpers import TINY_RUN, read_stream


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


def test_launch_joint_run_trains_and_evaluates_every_dataset(
    client: TestClient, repo_root: Path
):
    """A `datasets` list trains ONE conditioned model over all of them and writes
    one metrics.json with each dataset's IoU — the transfer-learning path."""
    import json

    from conftest import write_synthetic_tensors

    # A second preprocessed dataset alongside the fixture's `highest_t`.
    second = repo_root / "data" / "processed" / "second"
    second.mkdir(parents=True)
    write_synthetic_tensors(second / "tensors.npz")

    joint = {**TINY_RUN, "dataset": None, "datasets": ["highest_t", "second"]}
    response = client.post("/api/runs", json=joint)
    assert response.status_code == 202
    launched = response.json()
    assert launched["dataset"] == "highest_t"  # the primary series of the run

    run_id = launched["run_id"]
    events = read_stream(client, run_id)
    final = [e["data"] for e in events if e["event"] == "status"][-1]
    assert final["state"] == "done", f"joint run failed: {final.get('message')}"

    metrics = json.loads((repo_root / "outputs" / run_id / "metrics.json").read_text())
    assert set(metrics["per_dataset"]) == {"highest_t", "second"}
    assert metrics["datasets"] == ["highest_t", "second"]


def test_joint_run_renders_per_dataset_figures(
    client: TestClient, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """A rendered joint run writes each dataset's figures into its own subdir,
    lists them with relative names, and serves them back. ffmpeg is kept off
    PATH so the joint video path exercises its skip-not-crash guard."""
    from PIL import Image

    from conftest import write_synthetic_tensors

    monkeypatch.setenv("PATH", str(repo_root))  # no ffmpeg → video skips

    second = repo_root / "data" / "processed" / "second"
    second.mkdir(parents=True)
    write_synthetic_tensors(second / "tensors.npz")
    # Raw frames for both datasets, sized to the synthetic y_roi geometry.
    for name in ("highest_t", "second"):
        raw = repo_root / "data" / "raw" / name
        raw.mkdir(parents=True, exist_ok=True)
        for n in range(1, 13):
            Image.new("L", (16, 140), color=10 * n).save(raw / f"{n}.tif", format="TIFF")

    joint = {
        **TINY_RUN,
        "dataset": None,
        "datasets": ["highest_t", "second"],
        "render": True,
    }
    run_id = client.post("/api/runs", json=joint).json()["run_id"]
    events = read_stream(client, run_id)
    final = [e["data"] for e in events if e["event"] == "status"][-1]
    assert final["state"] == "done", f"joint render run failed: {final.get('message')}"

    figures_dir = repo_root / "outputs" / run_id / "figures"
    for name in ("highest_t", "second"):
        assert (figures_dir / name / "trajectories.png").is_file()
        assert (figures_dir / name / "pinn_on_images_all.png").is_file()
    listed = client.get(f"/api/runs/{run_id}").json()["artifacts"]["figures"]
    assert "second/trajectories.png" in listed
    served = client.get(f"/api/runs/{run_id}/figures/second/trajectories.png")
    assert served.status_code == 200
    assert client.get(f"/api/runs/{run_id}/figures/../secret.png").status_code == 404
    # No ffmpeg on PATH: the joint video skipped rather than failing the run.
    assert not (repo_root / "outputs" / run_id / "video" / "second" / "growth.mp4").exists()


def test_launch_joint_run_with_a_held_out_condition(client: TestClient, repo_root: Path):
    """A joint run can keep whole conditions out of training: they never enter the
    loss and are scored over every frame as a separate transfer IoU."""
    import json

    from conftest import write_synthetic_tensors

    for name in ("second", "third"):
        processed = repo_root / "data" / "processed" / name
        processed.mkdir(parents=True)
        write_synthetic_tensors(processed / "tensors.npz")

    joint = {
        **TINY_RUN,
        "dataset": None,
        "datasets": ["highest_t", "second", "third"],
        "heldout_datasets": ["third"],
        "val_fraction": 0.25,
        "val_strategy": "tail",
    }
    response = client.post("/api/runs", json=joint)
    assert response.status_code == 202
    run_id = response.json()["run_id"]

    events = read_stream(client, run_id)
    final = [e["data"] for e in events if e["event"] == "status"][-1]
    assert final["state"] == "done", f"joint run failed: {final.get('message')}"

    metrics = json.loads((repo_root / "outputs" / run_id / "metrics.json").read_text())
    assert metrics["training_datasets"] == ["highest_t", "second"]
    assert metrics["heldout_datasets"] == ["third"]
    assert set(metrics["per_dataset"]) == {"highest_t", "second"}
    assert set(metrics["transfer"]["per_dataset"]) == {"third"}
    # Per-frame transfer scores travel too, for the agreement charts.
    assert set(metrics["transfer"]["per_frame"]) == {"third"}
    assert len(metrics["transfer"]["per_frame"]["third"]) > 0

    # The launch overrides reached the run's own config snapshot.
    cfg = client.get(f"/api/runs/{run_id}").json()["config"]
    assert cfg["training"]["val_fraction"] == pytest.approx(0.25)
    assert cfg["training"]["val_strategy"] == "tail"
    assert list(cfg["heldout_datasets"]) == ["third"]


def test_joint_run_lists_its_validation_iou_as_the_headline(
    client: TestClient, repo_root: Path
):
    """metrics.json v2 has no `iou_holdout`; the runs list carries the joint
    run's in-distribution validation IoU explicitly so its row isn't blank."""
    from conftest import write_synthetic_tensors

    processed = repo_root / "data" / "processed" / "second"
    processed.mkdir(parents=True)
    write_synthetic_tensors(processed / "tensors.npz")

    joint = {
        **TINY_RUN,
        "dataset": None,
        "datasets": ["highest_t", "second"],
        "val_fraction": 0.2,
    }
    run_id = client.post("/api/runs", json=joint).json()["run_id"]
    read_stream(client, run_id)

    listed = {run["id"]: run for run in client.get("/api/runs").json()}
    detail = client.get(f"/api/runs/{run_id}").json()
    assert listed[run_id]["iou_holdout"] is None  # v2 metrics have no holdout frame
    assert listed[run_id]["val_iou_mean"] == detail["metrics"]["val_iou_mean"]
    assert listed[run_id]["datasets"] == ["highest_t", "second"]


def test_launch_rejects_holding_out_every_dataset(client: TestClient, repo_root: Path):
    """Nothing left to train on is a 400, before anything is scheduled."""
    from conftest import write_synthetic_tensors

    processed = repo_root / "data" / "processed" / "second"
    processed.mkdir(parents=True)
    write_synthetic_tensors(processed / "tensors.npz")

    joint = {
        **TINY_RUN,
        "dataset": None,
        "datasets": ["highest_t", "second"],
        "heldout_datasets": ["highest_t", "second"],
    }
    response = client.post("/api/runs", json=joint)
    assert response.status_code == 400
    assert "hold out" in response.json()["detail"]


def test_launch_rejects_a_held_out_dataset_not_in_the_run(client: TestClient):
    """A held-out condition must be one of the run's own datasets."""
    joint = {**TINY_RUN, "heldout_datasets": ["ghost"]}
    response = client.post("/api/runs", json=joint)
    assert response.status_code == 400
    assert "not in the run" in response.json()["detail"]


def test_joint_launch_rejects_an_unpreprocessed_dataset(client: TestClient):
    """Every dataset in a joint run must be preprocessed; one that isn't is a 409,
    before anything is scheduled."""
    # `sample` has raw frames in the fixture but was never preprocessed.
    joint = {**TINY_RUN, "dataset": None, "datasets": ["highest_t", "sample"]}
    response = client.post("/api/runs", json=joint)
    assert response.status_code == 409
    assert "preprocess" in response.json()["detail"]


def test_holdout_none_trains_on_all_frames(client: TestClient, repo_root: Path):
    """holdout_frame=-1 supervises every frame; the holdout metric is absent."""
    run_id = client.post("/api/runs", json={**TINY_RUN, "holdout_frame": -1}).json()["run_id"]
    events = read_stream(client, run_id)
    final = [e["data"] for e in events if e["event"] == "status"][-1]
    assert final["state"] == "done", f"run failed: {final.get('message')}"
    metrics = client.get(f"/api/runs/{run_id}").json()["metrics"]
    assert metrics["iou_holdout"] is None
    assert metrics["holdout_frame"] is None


def test_stage_b_run_gets_an_automatic_in_run_warm_up(
    client: TestClient, repo_root: Path, sample_processed: Path
):
    """Enabling Stage-B physics makes a launched run train the Stage-A objective
    alone for the first half, then engage momentum/energy/evaporation -- the warm
    start done inside one run, with no extra step for the user."""
    client.put("/api/physics/sample", json={"enabled": ["energy", "mom"], "weights": {}})

    run_id = client.post(
        "/api/runs", json={**TINY_RUN, "dataset": "sample", "steps": 4}
    ).json()["run_id"]
    final = [e["data"] for e in read_stream(client, run_id) if e["event"] == "status"][-1]
    assert final["state"] == "done", f"run failed: {final.get('message')}"

    cfg = client.get(f"/api/runs/{run_id}").json()["config"]
    assert cfg["training"]["stage_b_warmup_steps"] == 2  # half the run
    assert {"p", "T"} <= set(cfg["model"]["fields"])  # the Stage-B heads are present


def test_stage_a_run_has_no_warm_up_gate(client: TestClient, repo_root: Path):
    """A Stage-A run (no Stage-B physics enabled) is never gated: the warm-up is
    zero, so training is unchanged."""
    run_id = client.post("/api/runs", json=TINY_RUN).json()["run_id"]
    read_stream(client, run_id)
    cfg = client.get(f"/api/runs/{run_id}").json()["config"]
    assert cfg["training"]["stage_b_warmup_steps"] == 0


def test_launch_with_rba_and_adaptive_collocation_composes_the_config(
    client: TestClient, repo_root: Path
):
    """The advanced accuracy controls travel from the request to the composed config:
    a run launched with RBA weighting + residual-adaptive collocation trains and its
    saved config reflects both."""
    run_id = client.post(
        "/api/runs",
        json={**TINY_RUN, "weighting": "rba", "adaptive_collocation": True},
    ).json()["run_id"]
    final = [e["data"] for e in read_stream(client, run_id) if e["event"] == "status"][-1]
    assert final["state"] == "done", f"run failed: {final.get('message')}"

    cfg = client.get(f"/api/runs/{run_id}").json()["config"]
    assert cfg["training"]["weighting"] == "rba"
    assert cfg["training"]["adaptive_collocation"] is True


@pytest.mark.parametrize("mode", ["weight", "march"])
def test_launch_with_causal_weighting_composes_and_trains(
    client: TestClient, repo_root: Path, mode: str
):
    """Causal weighting travels through (both modes) and the run actually completes --
    the config snapshot is written before training, so assert `done` too, not just the
    composed value."""
    run_id = client.post(
        "/api/runs",
        json={**TINY_RUN, "causal_weighting": True, "causal_mode": mode},
    ).json()["run_id"]
    final = [e["data"] for e in read_stream(client, run_id) if e["event"] == "status"][-1]
    assert final["state"] == "done", f"run failed: {final.get('message')}"

    cfg = client.get(f"/api/runs/{run_id}").json()["config"]
    assert cfg["training"]["causal_weighting"] is True
    assert cfg["training"]["causal_mode"] == mode


def test_launch_with_hard_pin_and_kinematics_composes_and_trains(
    client: TestClient, repo_root: Path
):
    """The hard root pin and the kinematic growth constraints travel from the
    request to the composed config and the run completes. The evap-floor weight
    stays at the platform default 0 unless explicitly sent.

    The front geometry is the default now and pins the root exactly by
    construction, so a run that wants the SOFT pin has to say so by turning it
    off -- the two are mutually exclusive.
    """
    run_id = client.post(
        "/api/runs",
        json={
            **TINY_RUN,
            "front_geometry": False,
            "hard_pin": True,
            "kinematics": True,
            "kin_margin_frac": 0.5,
        },
    ).json()["run_id"]
    final = [e["data"] for e in read_stream(client, run_id) if e["event"] == "status"][-1]
    assert final["state"] == "done", f"run failed: {final.get('message')}"

    cfg = client.get(f"/api/runs/{run_id}").json()["config"]
    assert cfg["model"]["hard_pin"] is True
    assert cfg["model"]["pin_d_ref"] == pytest.approx(0.1)
    assert cfg["training"]["kinematics"] is True
    assert cfg["training"]["kin_margin_frac"] == pytest.approx(0.5)
    assert cfg["training"]["kin_weight_balance"] == pytest.approx(1.0)
    assert cfg["training"]["kin_weight_evap"] == pytest.approx(0.0), (
        "platform must not enable the evap floor"
    )


def test_launch_with_front_geometry_composes_and_trains(client: TestClient, repo_root: Path):
    """The R3 capsule interface travels from the request to the composed config
    and the run completes."""
    run_id = client.post(
        "/api/runs", json={**TINY_RUN, "front_geometry": True, "causal_weighting": True}
    ).json()["run_id"]
    final = [e["data"] for e in read_stream(client, run_id) if e["event"] == "status"][-1]
    assert final["state"] == "done", f"run failed: {final.get('message')}"

    cfg = client.get(f"/api/runs/{run_id}").json()["config"]
    assert cfg["model"]["front_geometry"] is True
    assert cfg["model"]["hard_pin"] is False


def test_launch_defaults_leave_pin_and_kinematics_off(client: TestClient, repo_root: Path):
    """An unchanged request keeps both features off -- byte-for-byte behavior."""
    run_id = client.post("/api/runs", json=TINY_RUN).json()["run_id"]
    read_stream(client, run_id)

    cfg = client.get(f"/api/runs/{run_id}").json()["config"]
    assert cfg["model"]["hard_pin"] is False
    assert cfg["training"]["kinematics"] is False


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
        pytest.param(
            {**TINY_RUN, "weighting": "rba", "causal_weighting": True},
            422,
            id="rba-incompatible-with-causal",
        ),
        pytest.param(
            {**TINY_RUN, "adaptive_collocation": True},
            422,
            id="adaptive-requires-rba",
        ),
        pytest.param({**TINY_RUN, "weighting": "bogus"}, 422, id="unknown-weighting"),
        pytest.param({**TINY_RUN, "pin_d_ref": 0}, 422, id="pin-d-ref-must-be-positive"),
        pytest.param({**TINY_RUN, "pin_d_ref": 2.1}, 422, id="pin-d-ref-above-range"),
        pytest.param({**TINY_RUN, "kin_margin_frac": -0.1}, 422, id="kin-margin-below-range"),
        pytest.param({**TINY_RUN, "kin_weight_mono": 101}, 422, id="kin-weight-above-range"),
        pytest.param({**TINY_RUN, "kin_weight_evap": -1}, 422, id="kin-evap-below-range"),
        pytest.param(
            {**TINY_RUN, "front_geometry": True, "hard_pin": True},
            422,
            id="front-geometry-excludes-hard-pin",
        ),
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


def test_joint_run_writes_per_dataset_trajectories(client: TestClient, repo_root: Path):
    """A joint run records growth kinematics per spanned dataset (including the
    held-out one) and serves them via ?dataset=; the unscoped path stays 404
    because a joint run has no single trajectory."""
    from conftest import write_synthetic_tensors

    processed = repo_root / "data" / "processed" / "second"
    processed.mkdir(parents=True)
    write_synthetic_tensors(processed / "tensors.npz")

    joint = {
        **TINY_RUN,
        "dataset": None,
        "datasets": ["highest_t", "second"],
        "heldout_datasets": ["second"],
        "val_fraction": 0.2,
    }
    run_id = client.post("/api/runs", json=joint).json()["run_id"]
    read_stream(client, run_id)

    for name in ("highest_t", "second"):
        assert (repo_root / "outputs" / run_id / f"trajectory_{name}.json").is_file()
        body = client.get(f"/api/runs/{run_id}/trajectory", params={"dataset": name}).json()
        assert body["t_ms"] and body["nose_um"]
        assert body["measured"]["t_ms"]

    assert client.get(f"/api/runs/{run_id}/trajectory").status_code == 404
    assert (
        client.get(f"/api/runs/{run_id}/trajectory", params={"dataset": "../evil"}).status_code
        == 404
    )


def test_launch_rejects_sharp_interface_without_the_front(client):
    """There is no front to sample without the geometry: 422 at the boundary with
    an actionable message, not a failure deep in the worker."""
    r = client.post(
        "/api/runs",
        json={**TINY_RUN, "front_geometry": False, "sharp_interface": True},
    )
    assert r.status_code == 422
    assert "front_geometry" in r.text


def test_launch_rejects_pinching_without_the_front(client):
    r = client.post(
        "/api/runs", json={**TINY_RUN, "front_geometry": False, "allow_pinch": True}
    )
    assert r.status_code == 422
    assert "front_geometry" in r.text


def test_launch_rejects_measured_front_velocity_without_the_front(client):
    """The term supervises the explicit front's own normal speed; without the
    geometry there is no such speed to supervise."""
    r = client.post(
        "/api/runs", json={**TINY_RUN, "front_geometry": False, "front_velocity": True}
    )
    assert r.status_code == 422
    assert "front_geometry" in r.text


def test_launch_rejects_measuring_the_front_and_then_ignoring_it(client):
    """Both weights at zero is a run that does the measurement and throws it
    away -- a silent no-op wearing a switched-on flag."""
    r = client.post(
        "/api/runs",
        json={**TINY_RUN, "front_velocity": True, "fv_weight": 0, "fv_apex_weight": 0},
    )
    assert r.status_code == 422
    assert "fv_weight" in r.text


def test_launch_with_measured_front_velocity_composes_and_trains(client, repo_root):
    """End to end through the API: the flag reaches the trainer, the run
    completes, and both measured terms appear in its history."""
    run_id = client.post(
        "/api/runs",
        json={**TINY_RUN, "front_geometry": True, "front_velocity": True},
    ).json()["run_id"]
    events = read_stream(client, run_id)

    final = [e["data"] for e in events if e["event"] == "status"][-1]
    assert final["state"] == "done", final
    history = [e["data"] for e in events if e["event"] == "hist"]
    assert history and {"fv_normal", "fv_apex"} <= set(history[0])
    assert (repo_root / "outputs" / run_id / "metrics.json").is_file()


def test_launch_rejects_a_sharpening_schedule_with_no_target(client):
    r = client.post("/api/runs", json={**TINY_RUN, "alpha_eps_anneal_steps": 100})
    assert r.status_code == 422
    assert "alpha_eps_final" in r.text


def test_launch_with_sharp_interface_composes_and_trains(client, repo_root):
    """The R4 recipe end to end through the API: the run completes and its
    metrics carry the physics diagnostics IoU cannot make.

    The jump condition reads the liquid pressure, so the series must be on the
    Stage-B field set first -- exactly the order a user goes through in the UI
    (there via PUT /api/physics/<series>; here written straight to the series'
    model config, which is the file that endpoint edits and every compose site
    reads).
    """
    (repo_root / "data" / "raw" / TINY_RUN["dataset"]).mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "raw" / TINY_RUN["dataset"] / "model.json").write_text(
        json.dumps({"enabled": ["mom"]})
    )
    run_id = client.post(
        "/api/runs",
        json={**TINY_RUN, "front_geometry": True, "sharp_interface": True},
    ).json()["run_id"]
    final = [e["data"] for e in read_stream(client, run_id) if e["event"] == "status"][-1]
    assert final["state"] == "done", f"run failed: {final.get('message')}"


def test_launch_rejects_film_pressure_without_the_sharp_interface(client):
    """It corrects the Young-Laplace jump; without that condition there is
    nothing for it to correct."""
    r = client.post(
        "/api/runs",
        json={**TINY_RUN, "sharp_interface": False, "film_pressure": True},
    )
    assert r.status_code == 422
    assert "sharp_interface" in r.text


def test_the_defaults_are_the_recommended_physics_recipe(client, repo_root):
    """Hitting Run without touching anything must give the best-known recipe.

    On a Stage-B series that is the full set: the capsule front, the interface
    conditions on it, the film-pressure correction, and a superheat that can
    deplete. Every measured gain in this line of work came from these.
    """
    import json

    from naviernet_api.models import RunLaunchRequest
    from naviernet_api.services.run_manager import _interface_overrides
    from naviernet_api.settings import Settings

    raw = repo_root / "data" / "raw" / TINY_RUN["dataset"]
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "model.json").write_text(json.dumps({"enabled": ["mom", "energy"]}))

    request = RunLaunchRequest(**TINY_RUN)
    assert request.front_geometry is True, "the capsule front is the default"
    assert request.evap_closure_two_way is True

    resolved = _interface_overrides(Settings(repo_root=repo_root), TINY_RUN["dataset"], request)
    assert "model.sharp_interface=true" in resolved
    assert "model.film_pressure=true" in resolved
    assert "model.depletable_superheat=true" in resolved


def test_a_stage_a_series_simply_does_not_get_the_stage_b_recipe(client, repo_root):
    """The conditions read fields a Stage-A series does not train, so the default
    does not apply there. A default not matching is not a silent downgrade -- and
    the run still launches, which is the point."""
    from naviernet_api.models import RunLaunchRequest
    from naviernet_api.services.run_manager import _interface_overrides
    from naviernet_api.settings import Settings

    resolved = _interface_overrides(
        Settings(repo_root=repo_root), TINY_RUN["dataset"], RunLaunchRequest(**TINY_RUN)
    )
    assert "model.sharp_interface=false" in resolved
    assert "model.film_pressure=false" in resolved
    assert "model.depletable_superheat=false" in resolved


def test_asking_explicitly_for_an_unsupported_condition_is_rejected(client):
    """A default that does not fit is silent; an explicit ASK that cannot be
    honoured is an error naming the fix."""
    r = client.post("/api/runs", json={**TINY_RUN, "sharp_interface": True})
    assert r.status_code == 422
    assert "'p'" in r.text and "Stage-B" in r.text


def test_launch_rejects_cap_freedom_without_the_front(client):
    """A free level set has no caps to free; the flag only means something on the
    geometric construction."""
    r = client.post(
        "/api/runs", json={**TINY_RUN, "front_geometry": False, "cap_freedom": True}
    )
    assert r.status_code == 422
    assert "front_geometry" in r.text


@pytest.mark.parametrize("bad", [-0.1, 1.0, 3.0])
def test_launch_rejects_a_cap_delta_outside_its_bound(client, bad):
    """The bound is what stops a cap folding through itself, so the API holds it
    rather than letting the geometry raise deep in the worker."""
    r = client.post("/api/runs", json={**TINY_RUN, "cap_freedom": True, "cap_delta": bad})
    assert r.status_code == 422


def test_cap_freedom_reaches_the_run_it_launched(client: TestClient):
    """The flag has to travel all the way into the run's own config, or a run
    trains a circular cap while the UI says otherwise."""
    launched = client.post(
        "/api/runs", json={**TINY_RUN, "cap_freedom": True, "cap_delta": 0.3}
    )
    assert launched.status_code == 202
    run_id = launched.json()["run_id"]
    read_stream(client, run_id)  # drain = wait for completion

    cfg = client.get(f"/api/runs/{run_id}").json()["config"]
    assert cfg["model"]["cap_freedom"] is True
    assert cfg["model"]["cap_delta"] == pytest.approx(0.3)


def test_launch_rejects_liquid_film_without_the_sharp_interface(client):
    """The film rides on the explicit front the sharp-interface conditions
    sample; without them there is nothing to ride on."""
    r = client.post(
        "/api/runs",
        json={**TINY_RUN, "sharp_interface": False, "liquid_film": True},
    )
    assert r.status_code == 422
    assert "sharp_interface" in r.text


def test_liquid_film_is_opt_in_and_travels_when_asked(client, repo_root):
    """Unbenched physics stays OUT of the default recipe -- the resolved flags
    carry liquid_film=false untouched -- and an explicit ask on a series that
    supports it composes the override."""
    import json

    from naviernet_api.models import RunLaunchRequest
    from naviernet_api.services.run_manager import _interface_overrides
    from naviernet_api.settings import Settings

    raw = repo_root / "data" / "raw" / TINY_RUN["dataset"]
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "model.json").write_text(json.dumps({"enabled": ["mom", "energy"]}))
    settings = Settings(repo_root=repo_root)

    silent = _interface_overrides(settings, TINY_RUN["dataset"], RunLaunchRequest(**TINY_RUN))
    assert "model.liquid_film=false" in silent, "opt-in: the recipe must not adopt it"

    asked = _interface_overrides(
        settings,
        TINY_RUN["dataset"],
        RunLaunchRequest(**{**TINY_RUN, "liquid_film": True}),
    )
    assert "model.liquid_film=true" in asked
    assert "model.sharp_interface=true" in asked


def test_asking_for_the_liquid_film_on_a_stage_a_series_is_rejected(client):
    """It needs the temperature field to deplete by; a Stage-A series has none."""
    r = client.post("/api/runs", json={**TINY_RUN, "liquid_film": True})
    assert r.status_code == 422
    assert "liquid_film" in r.text
