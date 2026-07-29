"""Read-only artifact endpoints: groups, validation, loss history, figures, video."""

from __future__ import annotations

import json

import pytest
from naviernet_api.services import runs as runs_service
from naviernet_api.settings import Settings


def test_groups_returns_dimensionless_groups(client):
    r = client.get("/api/runs/demo_run/groups")
    assert r.status_code == 200
    groups = r.json()
    assert groups["Re"] == pytest.approx(215.5)
    assert groups["bretherton_film_um"] == pytest.approx(4.875)


def test_validation_composes_metrics_and_groups(client):
    r = client.get("/api/runs/demo_run/validation")
    assert r.status_code == 200
    v = r.json()

    assert v["nose_speed_inferred_mm_s"] == pytest.approx(177.0)
    assert v["nose_speed_measured_mm_s"] == pytest.approx(180.0)  # documented
    assert v["nose_speed_error_pct"] == pytest.approx(100 * 3 / 180, rel=1e-3)
    assert v["bretherton_film_um"] == pytest.approx(4.875)
    assert v["reynolds"] == pytest.approx(215.5)
    assert v["iou_holdout"] == pytest.approx(0.968)
    assert v["holdout_frame"] == 6  # int, exact


def test_loss_history_returns_records(client):
    r = client.get("/api/runs/demo_run/loss-history")
    assert r.status_code == 200
    history = r.json()
    assert history[0]["step"] == 200  # int, exact
    assert history[0]["vof"] == pytest.approx(0.04)


def test_figure_is_served_as_png(client):
    r = client.get("/api/runs/demo_run/figures/trajectories.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"


def test_missing_figure_is_404(client):
    assert client.get("/api/runs/demo_run/figures/nope.png").status_code == 404


def test_video_is_served(client):
    r = client.get("/api/runs/demo_run/video")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"


@pytest.mark.parametrize(
    "evil", ["../../metrics.json", "..%2f..", "sub/dir.png", "no-extension"]
)
def test_figure_path_traversal_is_rejected(repo_root, evil):
    """A crafted figure name must not escape the run's figures dir."""
    settings = Settings(repo_root=repo_root)
    assert runs_service.figure_path(settings, "demo_run", evil) is None


def test_unsafe_dataset_name_cannot_escape_processed_root(repo_root):
    """A malicious `dataset` field must not steer tensors_path outside data/processed."""
    settings = Settings(repo_root=repo_root)
    evil = repo_root / "outputs" / "evil"
    evil.mkdir(parents=True)
    (evil / "metrics.json").write_text(json.dumps({"dataset": "../../../../etc"}))

    # The unsafe name is rejected, so it is never surfaced or used in a path.
    dataset, _ = runs_service.read_dataset_and_metrics(settings, "evil")
    assert dataset is None
    assert runs_service.tensors_path(settings, "evil") is None


def test_checkpoint_download(client):
    r = client.get("/api/runs/demo_run/checkpoint")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"


def test_tensors_download(client):
    r = client.get("/api/runs/demo_run/tensors")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"


@pytest.mark.parametrize(
    "endpoint",
    ["groups", "validation", "video", "checkpoint", "tensors", "loss-history", "figures/x.png"],
)
def test_artifacts_of_unknown_run_are_404(client, endpoint):
    assert client.get(f"/api/runs/ghost/{endpoint}").status_code == 404


def test_validation_carries_the_two_axis_summary_for_a_joint_run(client, repo_root):
    """A joint run's validation surfaces axis A (in-distribution), axis B
    (transfer), and the per-dataset agreement — not just the v1 fields."""
    import json

    joint = repo_root / "outputs" / "joint_run"
    (joint / "checkpoints").mkdir(parents=True)
    (joint / "metrics.json").write_text(
        json.dumps(
            {
                "datasets": ["highest_t", "second"],
                "training_datasets": ["highest_t"],
                "heldout_datasets": ["second"],
                "iou_mean": 0.953,
                "val_iou_mean": 0.941,
                "per_dataset": {
                    "highest_t": {
                        "iou_mean": 0.958,
                        "iou_val": 0.941,
                        "validation_frames": [8, 9, 10],
                        "iou_per_frame": {"0": 0.96, "8": 0.94},
                    }
                },
                "transfer": {"per_dataset": {"second": 0.903}, "mean": 0.903},
            }
        )
    )

    body = client.get("/api/runs/joint_run/validation").json()

    assert body["val_iou_mean"] == 0.941
    assert body["transfer_iou_mean"] == 0.903
    assert body["transfer_per_dataset"] == {"second": 0.903}
    assert body["training_datasets"] == ["highest_t"]
    assert body["heldout_datasets"] == ["second"]
    assert body["per_dataset"]["highest_t"]["iou_val"] == 0.941
    assert body["per_dataset"]["highest_t"]["validation_frames"] == [8, 9, 10]


def test_validation_single_run_carries_axis_a_when_split_was_used(client):
    """v1 metrics with iou_val (single-series axis A) surface it unchanged."""
    body = client.get("/api/runs/demo_run/validation").json()

    # demo_run's fixture metrics have no val split: the axis fields are null,
    # and the legacy holdout fields still populate.
    assert body["val_iou_mean"] is None
    assert body["iou_holdout"] == 0.968


def test_iou_csv_exports_frames_with_roles(client):
    response = client.get("/api/runs/demo_run/export/iou.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    lines = response.text.strip().splitlines()
    assert lines[0] == "dataset,camera_frame,iou,role"
    assert any("highest_t,6,0.968,holdout" in line for line in lines)


def test_loss_csv_exports_the_checkpoint_history(client):
    response = client.get("/api/runs/demo_run/export/loss.csv")
    assert response.status_code == 200
    lines = response.text.strip().splitlines()
    assert lines[0].startswith("step,lr,")
    assert lines[1].startswith("200,")


def test_exports_404_when_the_artifact_is_absent(client):
    assert client.get("/api/runs/scratch/export/loss.csv").status_code == 404
    assert client.get("/api/runs/nope/export/iou.csv").status_code == 404
