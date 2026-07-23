"""Dataset endpoints: read (list/detail/groups/qc/frames) and write (upload)."""

from __future__ import annotations

import threading

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
