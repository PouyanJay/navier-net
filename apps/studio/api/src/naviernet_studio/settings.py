"""Where the repository lives; everything else derives from it."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import naviernet

# The Vite dev server of apps/studio/web.
DEFAULT_WEB_PORT = 5174


def _default_repo_root() -> Path:
    """naviernet is installed editable from the repository, so its package file
    is ``<root>/src/naviernet/__init__.py``; ``NAVIERNET_ROOT`` overrides."""
    if env := os.environ.get("NAVIERNET_ROOT"):
        return Path(env).resolve()
    return Path(naviernet.__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    cors_origins: tuple[str, ...] = (
        f"http://localhost:{DEFAULT_WEB_PORT}",
        f"http://127.0.0.1:{DEFAULT_WEB_PORT}",
    )

    @property
    def outputs_dir(self) -> Path:
        return self.repo_root / "outputs"

    @property
    def data_raw_dir(self) -> Path:
        return self.repo_root / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.repo_root / "data" / "processed"

    @property
    def web_dist_dir(self) -> Path:
        return self.repo_root / "apps" / "studio" / "web" / "dist"


@lru_cache
def get_settings() -> Settings:
    origins = os.environ.get("NAVIERNET_STUDIO_CORS_ORIGINS")
    if origins:
        return Settings(
            repo_root=_default_repo_root(),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        )
    return Settings(repo_root=_default_repo_root())
