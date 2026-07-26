"""Reading and writing datasets under `data/raw/<id>/`.

Read: list datasets, their operating conditions (from the composed config), QC
figure, and per-frame previews. Write: validated image-sequence upload. Driving
preprocess lives in :mod:`naviernet_api.services.jobs`.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import tempfile
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from naviernet.data.preprocess import BAKED_CONDITION_FIELDS
from naviernet.physics.registry import REGISTRY
from naviernet.utils.logging import get_logger
from naviernet_api.models import DatasetDetail, DatasetSummary, OperatingConditions
from naviernet_api.services.config_service import compose_cfg
from naviernet_api.services.fluids import available_fluid_ids, is_known_fluid
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


class ConditionsError(ValueError):
    """A rejected conditions edit (unknown field, non-physical value)."""


class ExclusionError(ValueError):
    """A rejected frame-exclusion edit (out of range, holdout, too few left)."""


# Editable per-series conditions: the Hydra override path each field drives,
# plus its accepted [min, max] (server-side bounds per SECURITY.md §4; the
# ranges are generous physical sanity limits, not experiment tuning). Saved
# values apply at every compose site for the dataset (detail, groups,
# preprocess, run launches), so the whole pipeline sees them.
CONDITION_FIELDS: dict[str, tuple[str, float, float]] = {
    "dt_frame_ms": ("experiment.dt_frame_ms", 1e-6, 1e4),
    "channel_width_um": ("experiment.channel_width_um", 1.0, 1e6),
    "channel_height_um": ("experiment.channel_height_um", 1.0, 1e6),
    "flow_rate_mL_hr": ("experiment.flow_rate_mL_hr", 1e-3, 1e6),
    "q_wall_W_cm2": ("experiment.q_wall_W_cm2", 1e-3, 1e4),
    "U_ref": ("scales.U_ref", 1e-6, 1e3),
}
# T_sat is not here: it is a property of the selected fluid (derived, read-only),
# not a typed operating condition. The working fluid is chosen by group name and
# validated against the fluids allow-list; it is stored alongside the scalars
# under this key in conditions.json.
FLUID_KEY = "fluid"

# BAKED_CONDITION_FIELDS (imported at the top from the pipeline) names the
# condition fields written into the tensors; editing one makes them stale until a
# re-preprocess. Everything else (fluid, wall heat flux, flow rate, channel
# height) only feeds the dimensionless groups / Stage-B physics and applies live.


def is_valid_dataset_id(dataset: str) -> bool:
    # Exclude "." / ".." explicitly: they match the character class but "." would
    # resolve to the data root itself (collapsing per-dataset scoping); SECURITY.md §3.
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


def tensors_path(settings: Settings, dataset: str) -> Path | None:
    """The dataset's preprocessed tensors archive, or None if absent/invalid."""
    processed = _processed_dir(settings, dataset)
    if processed is None:
        return None
    tensors = processed / "tensors.npz"
    return tensors if tensors.is_file() else None


def _is_processed(settings: Settings, dataset: str) -> bool:
    return tensors_path(settings, dataset) is not None


# ── Per-series operating conditions ──────────────────────────────────────────


def _conditions_path(settings: Settings, dataset: str) -> Path | None:
    raw_dir = _raw_dir(settings, dataset)
    return None if raw_dir is None else raw_dir / "conditions.json"


def _read_raw_conditions(settings: Settings, dataset: str) -> dict:
    """The series' raw saved conditions object ({} when absent/corrupt). Carries
    both the scalar fields and the fluid choice; callers project out what they
    need so an edit to one never drops the other."""
    path = _conditions_path(settings, dataset)
    if path is None or not path.is_file():
        return {}
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("ignoring unreadable conditions for %s: %s", dataset, exc)
        return {}
    if not isinstance(saved, dict):
        # A wrong-shaped file must degrade like a corrupt one; not take the
        # whole dataset listing down with an AttributeError.
        log.warning("ignoring non-object conditions file for %s", dataset)
        return {}
    return saved


def read_conditions(settings: Settings, dataset: str) -> dict[str, float]:
    """The series' saved scalar condition values ({} when none saved)."""
    saved = _read_raw_conditions(settings, dataset)
    return {k: v for k, v in saved.items() if k in CONDITION_FIELDS}


def read_series_fluid(settings: Settings, dataset: str) -> str | None:
    """The series' selected fluid group id, or None when none saved (or the
    saved id is no longer a known fluid)."""
    fluid = _read_raw_conditions(settings, dataset).get(FLUID_KEY)
    if fluid is None:
        return None
    if isinstance(fluid, str) and is_known_fluid(fluid):
        return fluid
    # A fluid config that a series once selected was renamed or removed. Fall
    # back to the default fluid group, but say so rather than silently reverting.
    log.warning("ignoring unknown saved fluid %r for %s", fluid, dataset)
    return None


def conditions_overrides(settings: Settings, dataset: str) -> list[str]:
    """Saved conditions as Hydra overrides, for compose sites of this dataset.
    Includes the fluid *group* override (`fluid=<id>`) when one is selected."""
    saved = read_conditions(settings, dataset)
    overrides = [f"{CONDITION_FIELDS[key][0]}={value}" for key, value in sorted(saved.items())]
    fluid = read_series_fluid(settings, dataset)
    if fluid is not None:
        overrides.append(f"{FLUID_KEY}={fluid}")
    return overrides


def save_conditions(settings: Settings, dataset: str, updates: dict[str, float | str]) -> dict:
    """Merge validated condition edits into the series' file; returns the set.
    Accepts the scalar operating conditions and an allow-listed ``fluid`` id."""
    path = _conditions_path(settings, dataset)
    if path is None or not path.parent.is_dir():
        raise ConditionsError(f"dataset {dataset!r} not found")

    fluid = updates.get(FLUID_KEY)
    if fluid is not None and (not isinstance(fluid, str) or not is_known_fluid(fluid)):
        raise ConditionsError(
            f"unknown fluid {fluid!r}; choose one of {', '.join(available_fluid_ids())}"
        )
    for key, value in updates.items():
        if key == FLUID_KEY:
            continue
        if key not in CONDITION_FIELDS:
            raise ConditionsError(f"unknown condition field {key!r}")
        _, lo, hi = CONDITION_FIELDS[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConditionsError(f"{key} must be a finite number, got {value!r}")
        if not lo <= value <= hi:
            raise ConditionsError(f"{key} must be within [{lo}, {hi}], got {value!r}")

    merged = {**_read_raw_conditions(settings, dataset), **updates}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, indent=2))
    tmp.replace(path)
    log.info("saved conditions for %s: %s", dataset, sorted(updates))
    return merged


# ── Per-series frame exclusions ──────────────────────────────────────────────


def _exclusions_path(settings: Settings, dataset: str) -> Path | None:
    raw_dir = _raw_dir(settings, dataset)
    return None if raw_dir is None else raw_dir / "excluded_frames.json"


def read_excluded_frames(settings: Settings, dataset: str) -> list[int]:
    """The series' excluded camera frames, sorted ([] when none are saved)."""
    path = _exclusions_path(settings, dataset)
    if path is None or not path.is_file():
        return []
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("ignoring unreadable exclusions for %s: %s", dataset, exc)
        return []
    if not isinstance(saved, list):
        # Degrade like a corrupt file rather than taking the listing down.
        log.warning("ignoring non-list exclusions file for %s", dataset)
        return []
    return sorted({int(n) for n in saved if isinstance(n, int) and not isinstance(n, bool)})


def save_excluded_frames(settings: Settings, dataset: str, frames: list[int]) -> list[int]:
    """Replace the series' exclusion set with a validated one; returns it.

    The set is a full replacement, not a merge: the client sends the frames it
    wants excluded, so unchecking one is expressible.
    """
    path = _exclusions_path(settings, dataset)
    if path is None or not path.parent.is_dir():
        raise ExclusionError(f"dataset {dataset!r} not found")

    n_raw = _count_frames(path.parent)
    wanted = sorted({int(n) for n in frames})
    for frame in wanted:
        if not 1 <= frame <= n_raw:
            raise ExclusionError(f"frame {frame} is outside the sequence (1-{n_raw})")

    # The same config preprocessing will compose (minus the exclusions being
    # judged), so the API cannot accept a set the pipeline would then refuse.
    cfg = compose_cfg(dataset, overrides=base_overrides(settings, dataset))
    holdout = int(cfg.training.holdout_frame) + 1  # config is 0-based; frames are 1-based
    if holdout > 0 and holdout in wanted:
        raise ExclusionError(
            f"frame {holdout} is the holdout frame, the only unsupervised check on "
            "the model. Move the holdout before excluding this frame."
        )
    # The floor lives with the stage that enforces it, so the API cannot accept
    # a set preprocessing would then reject.
    from naviernet.data.preprocess import MIN_USABLE_FRAMES

    n_usable = int(cfg.experiment.n_frames_usable)
    kept = [n for n in range(1, n_usable + 1) if n not in set(wanted)]
    if len(kept) < MIN_USABLE_FRAMES:
        raise ExclusionError(
            f"at least {MIN_USABLE_FRAMES} usable frames must remain; this would leave "
            f"{len(kept)}"
        )

    if wanted:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(wanted, indent=2))
        tmp.replace(path)
    else:
        path.unlink(missing_ok=True)  # no file == no exclusions
    log.info("saved excluded frames for %s: %s", dataset, wanted)
    return wanted


# ── Per-series model & physics config ────────────────────────────────────────
#
# A series' model architecture and which governing equations it trains are saved
# in `data/raw/<id>/model.json` and applied as Hydra overrides at every compose
# site (mirroring conditions), so a launched run trains exactly what the Physics
# & Model page shows. The equation set is data-driven from the physics registry,
# so the API never hardcodes physics.

# Model-architecture globals a series may override, each with its Hydra path and
# server-side [min, max] bounds (SECURITY.md §4: generous sanity limits).
MODEL_ARCH_FIELDS: dict[str, tuple[str, float, float]] = {
    "hidden": ("model.hidden", 8, 1024),
    "layers": ("model.layers", 1, 32),
    "fourier_feats": ("model.fourier_feats", 4, 512),
    "fourier_scale": ("model.fourier_scale", 0.1, 100.0),
    "alpha_eps": ("model.alpha_eps", 1e-4, 1.0),
}
# The Stage-A field set the base architecture always trains (mirrors
# `configs/model/stage_a.yaml`; a small, stable fact not worth a compose to read).
_STAGE_A_FIELDS = ("phi", "u", "v", "s")
FIELD_NAMES = ("phi", "u", "v", "s", "p", "T")
# The Stage-B equations a series can toggle, keyed by id -> the fields they add.
_TOGGLEABLE = {e.id: e.fields_added for e in REGISTRY if not e.core and e.fields_added}
# Only the Stage-B equations' loss weights are owned here. The Stage-A weights
# (data/vof/div/src/bc) belong to the run-launch form (LossWeightsInput); keeping
# them out of the saved physics config avoids a silent precedence conflict when a
# run composes both.
_STAGE_B_WEIGHT_KEYS = {e.weight_key for e in REGISTRY if e.stage == "B"}
_WEIGHT_MAX = 1e4  # server-side weight bound, matching LossWeightsInput (SECURITY.md §4)


class ModelConfigError(ValueError):
    """A rejected model or physics edit (unknown field/equation, bad value)."""


def _require_finite(value: object, label: str, lo: float, hi: float) -> None:
    """Reject a value that is not a finite, in-range, non-bool number."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ModelConfigError(f"{label} must be a finite number, got {value!r}")
    if not lo <= value <= hi:
        raise ModelConfigError(f"{label} must be within [{lo}, {hi}], got {value!r}")


def _model_config_path(settings: Settings, dataset: str) -> Path | None:
    raw_dir = _raw_dir(settings, dataset)
    return None if raw_dir is None else raw_dir / "model.json"


def read_model_config(settings: Settings, dataset: str) -> dict:
    """The series' saved model + physics config ({} when none is saved)."""
    path = _model_config_path(settings, dataset)
    if path is None or not path.is_file():
        return {}
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("ignoring unreadable model config for %s: %s", dataset, exc)
        return {}
    return saved if isinstance(saved, dict) else {}


def _validate_arch(arch: dict) -> None:
    for key, value in ((k, v) for k, v in arch.items() if k != "per_field"):
        if key not in MODEL_ARCH_FIELDS:
            raise ModelConfigError(f"unknown model field {key!r}")
        _, lo, hi = MODEL_ARCH_FIELDS[key]
        _require_finite(value, key, lo, hi)
    for name, sub in arch.get("per_field", {}).items():
        if name not in FIELD_NAMES:
            raise ModelConfigError(f"unknown field {name!r} in per_field")
        for key, value in sub.items():
            if key not in ("hidden", "layers"):
                raise ModelConfigError(f"per_field.{name}.{key} is not overridable")
            _, lo, hi = MODEL_ARCH_FIELDS[key]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ModelConfigError(f"per_field.{name}.{key} must be an integer")
            _require_finite(value, f"per_field.{name}.{key}", lo, hi)


def save_model_arch(settings: Settings, dataset: str, arch: dict) -> dict:
    """Replace the series' model-architecture config (globals + per_field) with a
    validated edit; returns the full saved config.

    Full replacement, not a deep merge: the client sends the complete architecture
    it wants (mirroring ``save_excluded_frames``), so ``per_field`` here is the
    whole map, and a field the client omits is deliberately no longer overridden.
    """
    _validate_arch(arch)
    merged = {**read_model_config(settings, dataset), **arch}
    return _write_model_config(settings, dataset, merged)


def save_physics(settings: Settings, dataset: str, physics: dict) -> dict:
    """Replace the series' physics config -- which Stage-B equations are on and
    their loss weights -- with a validated edit; returns the full saved config.

    ``physics`` carries ``enabled`` (equation ids) and ``weights`` (weight_key ->
    value). Full replacement of both (the client sends the complete set each
    time). Only Stage-B weights are accepted; Stage-A weights are owned by the
    run-launch form.
    """
    enabled = list(physics.get("enabled", []))
    weights = dict(physics.get("weights", {}))
    for eid in enabled:
        if eid not in _TOGGLEABLE:
            raise ModelConfigError(f"{eid!r} is not a toggleable equation")
    for key, value in weights.items():
        if key not in _STAGE_B_WEIGHT_KEYS:
            raise ModelConfigError(
                f"loss weight {key!r} is not editable here (Stage-A weights are set at run launch)"
            )
        _require_finite(value, f"weight {key}", 0.0, _WEIGHT_MAX)
    merged = {
        **read_model_config(settings, dataset),
        "enabled": sorted(set(enabled)),
        "weights": weights,
    }
    return _write_model_config(settings, dataset, merged)


def _write_model_config(settings: Settings, dataset: str, config: dict) -> dict:
    path = _model_config_path(settings, dataset)
    if path is None or not path.parent.is_dir():
        raise ModelConfigError(f"dataset {dataset!r} not found")
    # A unique temp file per write: the model and physics saves land concurrently
    # (the web sends them together), and a shared ".json.tmp" would let one write's
    # replace() remove the file the other is about to replace.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix="model.", suffix=".json.tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(config, indent=2))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    log.info("saved model config for %s: %s", dataset, sorted(config))
    return config


def series_fields(settings: Settings, dataset: str, cfg: dict | None = None) -> list[str]:
    """The model fields this series trains: the Stage-A base plus whatever the
    enabled Stage-B equations add (pressure for momentum, temperature for energy).

    ``cfg`` may be a pre-read model config, to avoid re-reading model.json.
    """
    enabled = (cfg if cfg is not None else read_model_config(settings, dataset)).get(
        "enabled", []
    )
    added = [f for eid in enabled if eid in _TOGGLEABLE for f in _TOGGLEABLE[eid]]
    return list(_STAGE_A_FIELDS) + added


def _inline_per_field(per_field: dict) -> str:
    """Serialise per_field as a Hydra inline dict: `{p:{hidden:64,layers:3}}`."""
    parts = []
    for name, sub in per_field.items():
        inner = ",".join(f"{k}:{v}" for k, v in sub.items())
        parts.append(f"{name}:{{{inner}}}")
    return "{" + ",".join(parts) + "}"


def model_overrides(settings: Settings, dataset: str) -> list[str]:
    """The series' saved model + physics config as Hydra overrides.

    Applied at every compose site for the dataset, so the model view, the physics
    view, and a launched training run all train exactly this architecture and
    equation set. Only keys the pipeline reads are emitted; absent keys keep the
    pipeline defaults (Stage A).
    """
    cfg = read_model_config(settings, dataset)
    overrides: list[str] = []
    for key, (path, _, _) in MODEL_ARCH_FIELDS.items():
        if key in cfg:
            overrides.append(f"{path}={cfg[key]}")
    if cfg.get("enabled") is not None:
        overrides.append(f"model.fields=[{','.join(series_fields(settings, dataset, cfg))}]")
    if cfg.get("per_field"):
        overrides.append(f"+model.per_field={_inline_per_field(cfg['per_field'])}")
    for key, value in sorted((cfg.get("weights") or {}).items()):
        overrides.append(f"training.weights.{key}={value}")
    return overrides


def has_own_experiment_config(dataset: str) -> bool:
    """Whether `configs/experiment/<dataset>.yaml` describes this series.

    `configs/config.yaml` pins a single experiment group, so a series without
    one of these composes another series' experiment block. What is safe to
    inherit (fluid, wall heat flux) and what is not (how many frames there are)
    is decided against this.
    """
    from naviernet.config import config_dir

    return (config_dir() / "experiment" / f"{dataset}.yaml").is_file()


def _frame_count_overrides(settings: Settings, dataset: str) -> list[str]:
    """Frame counts taken from the frames actually on disk.

    Without this, an uploaded series inherits the pinned experiment's counts and
    preprocessing walks off the end of the sequence looking for frames that were
    never uploaded. A series with its own config keeps the counts it declares:
    those encode which frames are usable and which belong to one growth event,
    which no file listing can infer.
    """
    if has_own_experiment_config(dataset):
        return []
    raw_dir = _raw_dir(settings, dataset)
    n_frames = _count_frames(raw_dir) if raw_dir is not None else 0
    if n_frames == 0:
        return []
    # Every uploaded frame is assumed usable and part of the event until the
    # user says otherwise (by excluding frames in the sequence panel).
    return [
        f"experiment.n_frames_raw={n_frames}",
        f"experiment.n_frames_usable={n_frames}",
        f"experiment.n_frames_event={n_frames}",
    ]


def base_overrides(settings: Settings, dataset: str) -> list[str]:
    """The series' conditions and frame counts, without its exclusions.

    What an exclusion edit has to be validated against: folding in the current
    exclusions would judge a proposed set against itself.
    """
    return [
        *conditions_overrides(settings, dataset),
        *_frame_count_overrides(settings, dataset),
    ]


def series_overrides(settings: Settings, dataset: str) -> list[str]:
    """Everything known about this series as Hydra overrides: its saved
    conditions, the frame counts read off disk, and its excluded frames. Every
    compose site for the dataset uses this, so the detail view, the groups,
    preprocessing, and run launches all see one series."""
    overrides = base_overrides(settings, dataset)
    excluded = read_excluded_frames(settings, dataset)
    if excluded:
        overrides.append(f"experiment.excluded_frames=[{','.join(map(str, excluded))}]")
    # Model/physics overrides fold into the one series override list too. They
    # only touch cfg.model / cfg.training.weights, which preprocessing and groups
    # ignore, so a single list stays cheapest to keep in sync across compose sites.
    overrides.extend(model_overrides(settings, dataset))
    return overrides


def exclusions_applied(settings: Settings, dataset: str) -> bool:
    """Whether the preprocessed tensors were built with the saved exclusions.

    False while an edit is pending a re-run; the UI says so rather than
    implying the model already ignores the frame.
    """
    meta = tensors_meta(settings, dataset)
    if not meta:
        return False
    applied = meta.get("excluded_frames", [])
    return sorted(int(n) for n in applied) == read_excluded_frames(settings, dataset)


def _current_baked_conditions(settings: Settings, dataset: str) -> dict[str, float]:
    """The composed values of the tensor-baked condition fields for this series
    (with its saved overrides applied)."""
    cfg = compose_cfg(dataset, overrides=series_overrides(settings, dataset))
    baked: dict[str, float] = {}
    for key in BAKED_CONDITION_FIELDS:
        path = CONDITION_FIELDS[key][0]  # e.g. "experiment.dt_frame_ms"
        baked[key] = float(OmegaConf.select(cfg, path))
    return baked


def conditions_applied(settings: Settings, dataset: str) -> bool:
    """Whether the preprocessed tensors were built with the current tensor-baked
    conditions. False only when a baked edit is pending a re-preprocess.

    True for an unprocessed series (nothing to be stale about) and for tensors
    written before baked-condition snapshots existed (no false alarms).
    """
    snapshot = tensors_meta(settings, dataset).get("baked_conditions")
    if not snapshot:
        return True
    current = _current_baked_conditions(settings, dataset)
    return all(
        key in snapshot and math.isclose(float(snapshot[key]), value, rel_tol=1e-9)
        for key, value in current.items()
    )


def tensors_meta(settings: Settings, dataset: str) -> dict:
    """The meta record inside tensors.npz ({} when not preprocessed)."""
    path = tensors_path(settings, dataset)
    if path is None:
        return {}
    import numpy as np

    try:
        with np.load(path) as data:
            return json.loads(str(data["meta"]))
    except (OSError, ValueError, KeyError) as exc:
        log.warning("could not read tensors meta for %s: %s", dataset, exc)
        return {}


def _dt_frame_ms(settings: Settings, dataset: str) -> float | None:
    """The series' frame interval (with its saved overrides); None on failure."""
    try:
        cfg = compose_cfg(dataset, overrides=series_overrides(settings, dataset))
        return float(cfg.experiment.dt_frame_ms)
    except Exception as exc:  # noqa: BLE001 (a bad config must not hide the series)
        log.warning("could not compose config for %s: %s", dataset, exc)
        return None


def _frame_dimensions(raw_dir: Path) -> tuple[int, int] | None:
    """(width, height) of the first raw frame, or None if unreadable."""
    first = raw_dir / "1.tif"
    if not first.is_file():
        return None
    from PIL import Image

    try:
        with Image.open(first) as img:
            return (img.width, img.height)
    except OSError:
        return None


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
                conditions_set=bool(read_conditions(settings, entry.name)),
                frame_px=_frame_dimensions(entry),
                dt_frame_ms=_dt_frame_ms(settings, entry.name),
            )
        )
    return summaries


def _notes_for(cfg: DictConfig, dataset: str) -> str | None:
    """The experiment's frame-usage story, but only if it is about *this* series.

    `configs/config.yaml` pins one experiment group, so a series with no config
    of its own composes another experiment's block. Its prose describes that
    other series' frames; reporting it here would state confident falsehoods
    about frames this dataset does not have.
    """
    if not has_own_experiment_config(dataset):
        return None
    return cfg.experiment.notes or None


def get_dataset(settings: Settings, dataset: str) -> DatasetDetail | None:
    """Detail for one dataset, or None if the id is invalid or the dir is absent."""
    raw_dir = _raw_dir(settings, dataset)
    if raw_dir is None or not raw_dir.is_dir():
        return None
    cfg = compose_cfg(dataset, overrides=series_overrides(settings, dataset))
    meta = tensors_meta(settings, dataset)
    return DatasetDetail(
        id=dataset,
        n_frames=_count_frames(raw_dir),
        processed=_is_processed(settings, dataset),
        has_qc=qc_path(settings, dataset) is not None,
        conditions=conditions_from_cfg(cfg),
        conditions_set=bool(read_conditions(settings, dataset)),
        frame_px=_frame_dimensions(raw_dir),
        # Config stores the 0-based tensor index; report the 1-based camera
        # frame (f06), matching evaluation's metrics.json convention. -1 means
        # "train on all frames"; no holdout to mark.
        holdout_frame=(
            None if int(cfg.training.holdout_frame) < 0 else int(cfg.training.holdout_frame) + 1
        ),
        um_per_px=meta.get("um_per_px"),
        notes=_notes_for(cfg, dataset),
        excluded_frames=read_excluded_frames(settings, dataset),
        exclusions_applied=exclusions_applied(settings, dataset),
        conditions_applied=conditions_applied(settings, dataset),
    )


def conditions_from_cfg(cfg: DictConfig) -> OperatingConditions:
    """The response model's view of a composed config's operating conditions."""
    exp = cfg.experiment
    return OperatingConditions(
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
        U_ref_m_s=cfg.scales.U_ref,
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
        conditions_set=bool(read_conditions(settings, dataset)),
        frame_px=_frame_dimensions(raw_dir),
        dt_frame_ms=_dt_frame_ms(settings, dataset),
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
    except Exception as exc:  # noqa: BLE001 (any decode failure is a bad upload)
        raise UploadError(f"frame {index} is not a decodable TIFF image") from exc


def save_frames(settings: Settings, dataset: str, frames: list[bytes]) -> int:
    """Validate and save uploaded frames as 1.tif, 2.tif, … Returns the count.

    Server-generated names; TIFF magic + real decode + size + count checks
    (SECURITY.md §2). Replaces any existing sequence; never splices onto it.
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
