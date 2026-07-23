"""Runtime settings for the API.

The API is a thin layer over the naviernet package and its on-disk artifacts, so
its only real configuration is *where the repository lives* — the directory that
holds `outputs/`, `data/`, and `configs/`. Everything else derives from there via
the reused `RunPaths` layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import naviernet


def _default_repo_root() -> Path:
    """Locate the naviernet repo root.

    naviernet is installed editable from the repo, so its package file is
    `<root>/src/naviernet/__init__.py`; the root is two parents up. An explicit
    `NAVIERNET_ROOT` env var overrides this (e.g. in a container).
    """
    if env := os.environ.get("NAVIERNET_ROOT"):
        return Path(env).resolve()
    return Path(naviernet.__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    # Origins allowed to call the API. The Vite dev server defaults to :5173.
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )

    @property
    def outputs_dir(self) -> Path:
        return self.repo_root / "outputs"

    @property
    def data_raw_dir(self) -> Path:
        return self.repo_root / "data" / "raw"


@lru_cache
def get_settings() -> Settings:
    origins = os.environ.get("NAVIERNET_CORS_ORIGINS")
    if origins:
        return Settings(
            repo_root=_default_repo_root(),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        )
    return Settings(repo_root=_default_repo_root())
