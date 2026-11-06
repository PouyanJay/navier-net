"""A loaded run: its config, model, dataset, groups and scales, held once per
checkpoint and shared by every endpoint."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from naviernet.physics import registry
from naviernet.physics.groups import compute_groups
from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths
from naviernet_studio import ids
from naviernet_studio.scales import Scales
from naviernet_studio.settings import Settings

log = get_logger(__name__)

SCENE_CACHE_SIZE = 4
_scenes: dict[str, Scene] = {}
_cache_lock = threading.Lock()

ROLE_TRAIN = "train"
ROLE_HOLDOUT = "holdout"
ROLE_TRUNCATED = "truncated"
ROLE_EXCLUDED = "excluded"


@dataclass(frozen=True)
class FrameInfo:
    frame: int
    t_ms: float | None
    t_star: float | None
    role: str
    row: int | None


@dataclass
class Scene:
    run_id: str
    cfg: object
    model: object
    data: object
    groups: dict[str, float]
    scales: Scales
    paths: RunPaths
    checkpoint_mtime: float
    # Model evaluation is serialised per scene: the model caches shared
    # intermediates per call and the exact-continuity potentials per instant.
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def fields(self) -> list[str]:
        return list(self.model.fields)

    @property
    def has_front(self) -> bool:
        return bool(getattr(self.model, "front_geometry", False)) and bool(
            getattr(self.model, "sharp_interface", False)
        )

    @property
    def has_footprint(self) -> bool:
        return bool(getattr(self.model, "footprint_evaporation", False))

    @property
    def sharp_thermal(self) -> bool:
        return getattr(self.model, "thermal_model", "legacy") == "sharp"

    @property
    def equation_ids(self) -> list[str]:
        equations = registry.collocation_equations(
            registry.enabled_equations(
                self.cfg.model.fields, **registry.flags_from(self.cfg.model)
            )
        )
        return [eq.id for eq in equations]

    @property
    def inertia(self) -> bool:
        return bool(getattr(self.cfg.model, "momentum_inertia", False))

    def frames(self) -> list[FrameInfo]:
        data = self.data
        held = set(data.validation_frames)
        out: list[FrameInfo] = []
        for row, number in enumerate(data.frame_numbers):
            if row < data.n_event:
                role = ROLE_HOLDOUT if number in held else ROLE_TRAIN
            else:
                role = ROLE_TRUNCATED
            t_star = float(data.t[row])
            out.append(FrameInfo(number, t_star * self.scales.time_ms, t_star, role, row))
        for number in data.meta.get("excluded_frames", []):
            out.append(FrameInfo(int(number), None, None, ROLE_EXCLUDED, None))
        out.sort(key=lambda f: f.frame)
        return out

    def row_of(self, frame: int) -> int | None:
        numbers = self.data.frame_numbers
        return numbers.index(frame) if frame in numbers else None

    def t_star_of(self, frame: int | None, t_ms: float | None) -> float:
        """The instant a request names, clamped to the trained span."""
        if frame is not None:
            row = self.row_of(frame)
            if row is None:
                raise LookupError(f"frame {frame} is not in this run's tensors")
            return float(self.data.t[row])
        if t_ms is None:
            raise LookupError("name an instant with frame= or t_ms=")
        t = t_ms / self.scales.time_ms
        return float(min(max(t, self.data.domain.t_min), self.data.domain.t_max))

    def frame_of(self, t_star: float) -> int | None:
        """The camera frame at exactly this instant, if there is one."""
        for row, value in enumerate(self.data.t):
            if abs(float(value) - t_star) < 1e-6:
                return int(self.data.frame_numbers[row])
        return None


def _load_config(settings: Settings, run_id: str, directory: Path):
    """The run's own config snapshot over the current schema, pinned to this
    checkout and the CPU; None when the run kept no snapshot."""
    snapshot = directory / ".hydra" / "config.yaml"
    if not snapshot.is_file():
        return None
    from omegaconf import OmegaConf

    from naviernet.config.schema import Config

    try:
        cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(snapshot))
    except Exception as exc:  # noqa: BLE001 (any parse or merge failure means unreadable)
        log.warning("could not load the config snapshot of %s: %s", run_id, exc)
        return None
    cfg.paths.root = str(settings.repo_root)
    cfg.run_name = run_id
    cfg.training.device = "cpu"
    OmegaConf.set_readonly(cfg, True)
    return cfg


def _load(settings: Settings, run_id: str) -> Scene | None:
    directory = ids.run_dir(settings, run_id)
    paths = ids.run_paths(settings, run_id)
    if directory is None or paths is None or not paths.checkpoint.is_file():
        return None
    cfg = _load_config(settings, run_id, directory)
    if cfg is None:
        return None
    from naviernet.training import load_model

    try:
        model, data, _ = load_model(cfg, paths)
    except FileNotFoundError as exc:
        log.warning("run %s cannot be loaded: %s", run_id, exc)
        return None
    groups = compute_groups(cfg)
    return Scene(
        run_id=run_id,
        cfg=cfg,
        model=model,
        data=data,
        groups=groups,
        scales=Scales.from_run(cfg, groups, data),
        paths=paths,
        checkpoint_mtime=paths.checkpoint.stat().st_mtime,
    )


def get_scene(settings: Settings, run_id: str) -> Scene | None:
    """The scene for a run, loaded once per checkpoint; None when unavailable."""
    paths = ids.run_paths(settings, run_id)
    if paths is None or not paths.checkpoint.is_file():
        return None
    mtime = paths.checkpoint.stat().st_mtime
    with _cache_lock:
        cached = _scenes.get(run_id)
        if cached is not None and cached.checkpoint_mtime == mtime:
            return cached
    scene = _load(settings, run_id)
    if scene is None:
        return None
    with _cache_lock:
        _scenes[run_id] = scene
        while len(_scenes) > SCENE_CACHE_SIZE:
            del _scenes[next(iter(_scenes))]
    return scene
