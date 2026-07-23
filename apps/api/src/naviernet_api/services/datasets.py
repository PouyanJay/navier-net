"""Reading and writing datasets under `data/raw/<id>/`.

Read: list datasets, their operating conditions (from the composed config), QC
figure, and per-frame previews. Write: validated image-sequence upload. Driving
preprocess lives in :mod:`naviernet_api.services.jobs`.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from naviernet.utils.logging import get_logger
from naviernet_api.models import DatasetDetail, DatasetSummary, OperatingConditions
from naviernet_api.services.config_service import compose_cfg
from naviernet_api.settings import Settings

log = get_logger(__name__)

_DATASET_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_FRAME_RE = re.compile(r"^(\d+)\.tif$")

# Upload limits (SECURITY.md §2, §4): reject before decoding.
TIFF_MAGIC = (b"II*\x00", b"MM\x00*")  # little- / big-endian TIFF
MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_FRAMES = 200


class UploadError(ValueError):
    """A rejected upload (bad type, too large, too many, bad id)."""


def is_valid_dataset_id(dataset: str) -> bool:
    # Exclude "." / ".." explicitly: they match the character class but "." would
    # resolve to the data root itself (collapsing per-dataset scoping) — SECURITY.md §3.
    return bool(_DATASET_RE.match(dataset)) and dataset not in {".", ".."}


def _confined(root: Path, dataset: str) -> Path | None:
    """`root/dataset` if the id is valid and the resolved path stays inside root."""
    if not is_valid_dataset_id(dataset):
        return None
    root = root.resolve()
    path = (root / dataset).resolve()
    return path if path.is_relative_to(root) and path != root else None


def _raw_dir(settings: Settings, dataset: str) -> Path | None:
    """The dataset's raw dir if the id is valid and confined; may not yet exist."""
    return _confined(settings.data_raw_dir, dataset)


def _processed_dir(settings: Settings, dataset: str) -> Path | None:
    return _confined(settings.repo_root / "data" / "processed", dataset)


def _count_frames(raw_dir: Path) -> int:
    if not raw_dir.is_dir():
        return 0
    return sum(1 for p in raw_dir.iterdir() if _FRAME_RE.match(p.name))


def _is_processed(settings: Settings, dataset: str) -> bool:
    processed = _processed_dir(settings, dataset)
    return processed is not None and (processed / "tensors.npz").is_file()


def list_datasets(settings: Settings) -> list[DatasetSummary]:
    """Every dataset directory under `data/raw/`."""
    raw_root = settings.data_raw_dir
    if not raw_root.is_dir():
        return []
    summaries = []
    for entry in sorted(raw_root.iterdir()):
        if not entry.is_dir() or not is_valid_dataset_id(entry.name):
            continue
        summaries.append(
            DatasetSummary(
                id=entry.name,
                n_frames=_count_frames(entry),
                processed=_is_processed(settings, entry.name),
            )
        )
    return summaries


def get_dataset(settings: Settings, dataset: str) -> DatasetDetail | None:
    """Detail for one dataset, or None if the id is invalid or the dir is absent."""
    raw_dir = _raw_dir(settings, dataset)
    if raw_dir is None or not raw_dir.is_dir():
        return None
    exp = compose_cfg(dataset).experiment
    conditions = OperatingConditions(
        fluid=exp.fluid,
        T_sat_C=exp.T_sat_C,
        q_wall_W_cm2=exp.q_wall_W_cm2,
        flow_rate_mL_hr=exp.flow_rate_mL_hr,
        channel_width_um=exp.channel_width_um,
        channel_height_um=exp.channel_height_um,
        dt_frame_ms=exp.dt_frame_ms,
        flow_direction=exp.flow_direction,
        n_frames_raw=exp.n_frames_raw,
        n_frames_usable=exp.n_frames_usable,
        n_frames_event=exp.n_frames_event,
    )
    return DatasetDetail(
        id=dataset,
        n_frames=_count_frames(raw_dir),
        processed=_is_processed(settings, dataset),
        has_qc=qc_path(settings, dataset) is not None,
        conditions=conditions,
    )


def get_dataset_summary(settings: Settings, dataset: str) -> DatasetSummary | None:
    """Summary for one dataset, built directly (no directory rescan)."""
    raw_dir = _raw_dir(settings, dataset)
    if raw_dir is None or not raw_dir.is_dir():
        return None
    return DatasetSummary(
        id=dataset,
        n_frames=_count_frames(raw_dir),
        processed=_is_processed(settings, dataset),
    )


def qc_path(settings: Settings, dataset: str) -> Path | None:
    """The preprocessing QC figure for a dataset, or None if absent."""
    processed = _processed_dir(settings, dataset)
    if processed is None:
        return None
    qc = processed / "qc_preprocess.png"
    return qc if qc.is_file() else None


def frame_preview_png(
    settings: Settings, dataset: str, n: int, max_width: int = 640
) -> bytes | None:
    """A raw TIFF frame rendered to a downscaled PNG for the browser."""
    raw_dir = _raw_dir(settings, dataset)
    if raw_dir is None:
        return None
    frame = (raw_dir / f"{n}.tif").resolve()
    if not frame.is_relative_to(raw_dir.resolve()) or not frame.is_file():
        return None
    from PIL import Image

    try:
        with Image.open(frame) as img:
            img = img.convert("L")
            if img.width > max_width:
                height = round(img.height * max_width / img.width)
                img = img.resize((max_width, height))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
    except OSError as exc:  # unreadable / corrupt image
        log.warning("could not render frame %s of %s: %s", n, dataset, exc)
        return None
    return buffer.getvalue()


def _verify_tiff(data: bytes, index: int) -> None:
    """Confirm bytes are a decodable TIFF (magic bytes alone aren't enough)."""
    if not data.startswith(TIFF_MAGIC):
        raise UploadError(f"frame {index} is not a TIFF image")
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()  # structural check; also raises on a decompression bomb
    except Exception as exc:  # noqa: BLE001 — any decode failure is a bad upload
        raise UploadError(f"frame {index} is not a decodable TIFF image") from exc


def save_frames(settings: Settings, dataset: str, frames: list[bytes]) -> int:
    """Validate and save uploaded frames as 1.tif, 2.tif, … Returns the count.

    Server-generated names; TIFF magic + real decode + size + count checks
    (SECURITY.md §2). Replaces any existing sequence — never splices onto it.
    """
    raw_dir = _raw_dir(settings, dataset)
    if raw_dir is None:
        raise UploadError(f"invalid dataset id {dataset!r}")
    if not frames:
        raise UploadError("no files uploaded")
    if len(frames) > MAX_FRAMES:
        raise UploadError(f"too many frames ({len(frames)} > {MAX_FRAMES})")
    for i, data in enumerate(frames, start=1):
        if len(data) > MAX_FRAME_BYTES:
            raise UploadError(f"frame {i} exceeds {MAX_FRAME_BYTES} bytes")
        _verify_tiff(data, i)

    raw_dir.mkdir(parents=True, exist_ok=True)
    for stale in raw_dir.glob("*.tif"):  # replace, don't splice
        stale.unlink()
    for i, data in enumerate(frames, start=1):
        (raw_dir / f"{i}.tif").write_bytes(data)
    log.info("saved %d frame(s) to %s", len(frames), raw_dir)
    return len(frames)
