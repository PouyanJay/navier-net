"""Dataset endpoints: read (list/detail/groups/qc/frames) and write (upload)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from naviernet_api.services import datasets as datasets_service
from naviernet_api.services import jobs as jobs_service
from naviernet_api.settings import Settings


def test_list_datasets(client):
    r = client.get("/api/datasets")
    assert r.status_code == 200
    ids = {d["id"]: d for d in r.json()}
    assert "sample" in ids
    assert ids["sample"]["n_frames"] == 3
    assert ids["sample"]["processed"] is False


def test_dataset_detail_has_operating_conditions(client):
    r = client.get("/api/datasets/sample")
    assert r.status_code == 200
    detail = r.json()
    assert detail["n_frames"] == 3
    # Conditions come from the composed (default experiment) config.
    assert detail["conditions"]["fluid"] == "FC-72"
    assert detail["conditions"]["channel_width_um"] == pytest.approx(300.0)
    # Config holds the 0-based tensor index (5); the API reports camera frame 6.
    assert detail["holdout_frame"] == 6


def test_live_groups_are_computed(client):
    r = client.get("/api/datasets/sample/groups")
    assert r.status_code == 200
    groups = r.json()
    assert groups["Re"] == pytest.approx(215.5, rel=1e-3)
    assert groups["bretherton_film_um"] == pytest.approx(4.875, rel=1e-3)


def test_qc_is_404_before_preprocessing(client):
    assert client.get("/api/datasets/sample/qc").status_code == 404


def test_frame_preview_is_png(client):
    r = client.get("/api/datasets/sample/frames/1")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG")


def test_unknown_dataset_is_404(client):
    assert client.get("/api/datasets/nope").status_code == 404
    assert client.get("/api/datasets/nope/frames/1").status_code == 404


def test_upload_accepts_valid_tiffs(client, tiff_bytes):
    files = [("files", (f"frame{i}.tif", tiff_bytes, "image/tiff")) for i in range(2)]
    r = client.post("/api/datasets/fresh/upload", files=files)
    assert r.status_code == 200
    assert r.json()["n_frames"] == 2
    # Saved under server-generated names, listable.
    assert client.get("/api/datasets/fresh").json()["n_frames"] == 2


def test_upload_rejects_non_tiff(client):
    files = [("files", ("evil.tif", b"not a tiff at all", "image/tiff"))]
    r = client.post("/api/datasets/fresh2/upload", files=files)
    assert r.status_code == 400
    assert "TIFF" in r.json()["detail"]


@pytest.mark.parametrize("bad_id", ["../evil", "a/b", "with space"])
def test_upload_rejects_unsafe_dataset_id(repo_root, bad_id, tiff_bytes):
    """A crafted dataset id must not escape data/raw/ (SECURITY.md §3)."""
    settings = Settings(repo_root=repo_root)
    with pytest.raises(datasets_service.UploadError):
        datasets_service.save_frames(settings, bad_id, [tiff_bytes])


def test_upload_rejects_too_many_frames(repo_root, tiff_bytes):
    settings = Settings(repo_root=repo_root)
    too_many = [tiff_bytes] * (datasets_service.MAX_FRAMES + 1)
    with pytest.raises(datasets_service.UploadError, match="too many"):
        datasets_service.save_frames(settings, "big", too_many)


def test_preprocess_status_is_idle_before_start(client):
    r = client.get("/api/datasets/sample/preprocess")
    assert r.status_code == 200
    assert r.json()["state"] == "idle"


def test_start_preprocess_marks_running_and_runs_the_worker(repo_root, monkeypatch):
    """A fresh start marks the job running and actually spawns the worker."""
    settings = Settings(repo_root=repo_root)
    ran = threading.Event()
    # Patch the narrow _run seam (not global threading) so a real thread starts
    # but does trivial work.
    monkeypatch.setattr(jobs_service, "_run", lambda _s, _d: ran.set())

    result = jobs_service.start_preprocess(settings, "sample")
    assert result.state == "running"
    assert ran.wait(5), "the background worker did not run"


def test_start_preprocess_does_not_restart_a_running_job(repo_root, monkeypatch):
    """A second start while running is a no-op (no second worker spawned)."""
    settings = Settings(repo_root=repo_root)
    spawns: list[str] = []
    monkeypatch.setattr(jobs_service, "_run", lambda _s, d: spawns.append(d))
    # Seed a running job so the next start must not spawn.
    jobs_service._jobs["sample"] = jobs_service._Job(state="running")

    result = jobs_service.start_preprocess(settings, "sample")
    assert result.state == "running"
    assert spawns == []


def test_start_preprocess_via_http_route(client, monkeypatch):
    """The POST route itself wires through to a running job."""
    monkeypatch.setattr(jobs_service, "_run", lambda _s, _d: None)
    r = client.post("/api/datasets/sample/preprocess")
    assert r.status_code == 200
    assert r.json()["state"] == "running"


@pytest.mark.parametrize("bad_id", [".", "..", "../evil", "a b"])
def test_unsafe_dataset_ids_are_invalid(bad_id):
    """ "." / ".." collapse to the data root; others escape — none are valid
    (SECURITY.md §3). ("." can't be sent through the httpx TestClient, which
    normalizes it away, so the guard is asserted at the service level.)"""
    assert datasets_service.is_valid_dataset_id(bad_id) is False
    assert datasets_service._raw_dir(Settings(repo_root=Path("/tmp")), bad_id) is None


def test_groups_of_unknown_dataset_is_404(client):
    """Live groups require the dataset to exist, not just a syntactically valid id."""
    assert client.get("/api/datasets/made-up/groups").status_code == 404


def test_reupload_replaces_the_sequence(client, tiff_bytes):
    def upload(count: int):
        files = [("files", (f"{i}.tif", tiff_bytes, "image/tiff")) for i in range(count)]
        return client.post("/api/datasets/reup/upload", files=files)

    assert upload(3).json()["n_frames"] == 3
    # A shorter re-upload must not leave stale frames 4..N behind.
    assert upload(2).json()["n_frames"] == 2


def test_upload_rejects_undecodable_tiff(client):
    """Magic bytes alone aren't enough — the body must actually decode."""
    fake = b"II*\x00" + b"\x00" * 32  # TIFF magic, garbage body
    files = [("files", ("1.tif", fake, "image/tiff"))]
    r = client.post("/api/datasets/broken/upload", files=files)
    assert r.status_code == 400
    assert "decodable" in r.json()["detail"]


def test_save_frames_empty_and_oversize(repo_root, tiff_bytes, monkeypatch):
    settings = Settings(repo_root=repo_root)
    with pytest.raises(datasets_service.UploadError, match="no files"):
        datasets_service.save_frames(settings, "x", [])
    monkeypatch.setattr(datasets_service, "MAX_FRAME_BYTES", 2)
    with pytest.raises(datasets_service.UploadError, match="exceeds"):
        datasets_service.save_frames(settings, "x", [tiff_bytes])


@pytest.mark.needs_data
@pytest.mark.slow
def test_preprocess_write_path_end_to_end(tmp_path, monkeypatch):
    """Upload the real frames, drive the real preprocess, and produce tensors + QC."""
    import shutil
    import time

    source = Path("data/raw/highest_t")
    if not source.is_dir():
        pytest.skip("real highest_t frames not present")

    raw = tmp_path / "data" / "raw" / "highest_t"
    raw.mkdir(parents=True)
    for tif in source.glob("*.tif"):
        shutil.copy(tif, raw / tif.name)

    settings = Settings(repo_root=tmp_path)
    jobs_service.start_preprocess(settings, "highest_t")
    for _ in range(120):
        status = jobs_service.status(settings, "highest_t")
        if status.state != "running":
            break
        time.sleep(0.5)

    assert status.state == "done", status.message
    assert (tmp_path / "data" / "processed" / "highest_t" / "tensors.npz").is_file()
    assert status.has_qc


def test_patch_conditions_saves_and_recomputes_groups(client):
    baseline = client.get("/api/datasets/sample/groups").json()

    r = client.patch("/api/datasets/sample/conditions", json={"U_ref": 0.4})
    assert r.status_code == 200
    body = r.json()
    assert body["conditions"]["U_ref_m_s"] == pytest.approx(0.4)
    # Re scales with the reference velocity — doubling U_ref must double it.
    assert body["groups"]["Re"] == pytest.approx(2 * baseline["Re"], rel=1e-6)

    # The edit persists into the detail + groups endpoints (and is flagged).
    detail = client.get("/api/datasets/sample").json()
    assert detail["conditions"]["U_ref_m_s"] == pytest.approx(0.4)
    assert detail["conditions_set"] is True
    assert client.get("/api/datasets/sample/groups").json()["Re"] == pytest.approx(
        body["groups"]["Re"]
    )


def test_patch_conditions_rejects_non_physical_values(client):
    r = client.patch("/api/datasets/sample/conditions", json={"dt_frame_ms": -1})
    assert r.status_code == 400
    assert "dt_frame_ms" in r.json()["detail"]


def test_patch_conditions_unknown_dataset_is_404(client):
    assert (
        client.patch("/api/datasets/nope/conditions", json={"dt_frame_ms": 1}).status_code
        == 404
    )


def test_conditions_overrides_reach_the_run_config(repo_root, tiff_bytes):
    settings = Settings(repo_root=repo_root)
    datasets_service.save_conditions(settings, "sample", {"U_ref": 0.5})
    overrides = datasets_service.conditions_overrides(settings, "sample")
    assert overrides == ["scales.U_ref=0.5"]


def test_dataset_summary_carries_frame_size(client):
    ids = {d["id"]: d for d in client.get("/api/datasets").json()}
    # The fixture writes small real TIFFs; their true size must be reported.
    assert ids["sample"]["frame_px"] is not None
    width, height = ids["sample"]["frame_px"]
    assert width > 0 and height > 0


def test_qc_data_has_all_three_checks(client):
    r = client.get("/api/datasets/highest_t/qc-data")
    assert r.status_code == 200
    qc = r.json()
    kin = qc["kinematics"]
    assert len(kin["t_ms"]) == len(kin["length_um"]) == 11
    assert isinstance(kin["fit_slope_mm_s"], (int, float))
    assert len(qc["interface"]["frames"]) == 6  # every 2nd of 11 frames
    assert qc["interface"]["frames"][0]["contours"]  # real polylines
    sdf = qc["sdf"]
    assert sdf["frame_index"] == 5
    assert len(sdf["values"]) > 0 and len(sdf["values"][0]) > 0


def test_qc_data_before_preprocessing_is_404(client):
    assert client.get("/api/datasets/sample/qc-data").status_code == 404
