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

    assert body["val_iou_mean"] == pytest.approx(0.941)
    assert body["transfer_iou_mean"] == pytest.approx(0.903)
    assert body["transfer_per_dataset"] == {"second": pytest.approx(0.903)}
    assert body["training_datasets"] == ["highest_t"]
    assert body["heldout_datasets"] == ["second"]
    assert body["per_dataset"]["highest_t"]["iou_val"] == pytest.approx(0.941)
    assert body["per_dataset"]["highest_t"]["validation_frames"] == [8, 9, 10]


def test_validation_single_run_carries_axis_a_when_split_was_used(client):
    """v1 metrics with iou_val (single-series axis A) surface it unchanged."""
    body = client.get("/api/runs/demo_run/validation").json()

    # demo_run's fixture metrics have no val split: the axis fields are null,
    # and the legacy holdout fields still populate.
    assert body["val_iou_mean"] is None
    assert body["iou_holdout"] == pytest.approx(0.968)


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


def test_front_velocity_csv_exports_every_series_in_both_units(client):
    """One flat file for a paper's analysis: the whole-front speeds and the
    profile, each row carrying the reporting unit and the SI one."""
    response = client.get("/api/runs/demo_run/export/front-velocity.csv")
    assert response.status_code == 200
    lines = response.text.strip().splitlines()

    assert lines[0] == "series,t_ms,s,v_um_per_ms,v_m_per_s,heldout"
    series = {line.split(",")[0] for line in lines[1:]}
    assert series == {
        "nose_speed",
        "nose_speed_measured",
        "apex_vx",
        "apex_vx_measured",
        "apex_vy",
        "apex_vy_measured",
        "profile_model",
        "profile_measured",
    }
    # µm/ms and m/s on the same row, so the file needs no conversion to read.
    assert "nose_speed,0.0,,120.5,0.1205," in response.text
    # The pair that spans a held-out frame says so.
    assert "nose_speed_measured,0.25,,131.0,0.131,True" in response.text


def test_front_velocity_csv_keeps_a_suppressed_bin_as_a_row(client):
    """The nose cap's measurement is deliberately absent, not missing. Dropping
    the row would say the position does not exist; an empty value says it was
    looked at and nothing trustworthy was found."""
    text = client.get("/api/runs/demo_run/export/front-velocity.csv").text

    assert "profile_measured,0.0,0.25,3.0,0.003,False" in text
    assert "profile_measured,0.0,0.75,,,False" in text
    # The model's own value is still there at the same position.
    assert "profile_model,0.0,0.75,120.0,0.12,False" in text


def test_exports_404_when_the_artifact_is_absent(client):
    assert client.get("/api/runs/scratch/export/loss.csv").status_code == 404
    assert client.get("/api/runs/nope/export/iou.csv").status_code == 404
    assert client.get("/api/runs/scratch/export/front-velocity.csv").status_code == 404
