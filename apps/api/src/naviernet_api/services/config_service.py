"""Composing a Hydra config for a dataset.

Reuses the pipeline's own config schema and groups so the API never re-implements
any physics. Hydra keeps global state, so composition is serialized behind a lock
and the global instance is cleared each time — safe for the API's occasional,
low-frequency use (operating conditions, live groups, driving preprocess).
"""

from __future__ import annotations

import threading

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from naviernet.config import config_dir, register_configs

# NOTE: both this cache and the job registry key on dataset name alone, assuming
# one Settings.repo_root per process (true for the API). The cached configs are
# marked read-only below so a shared instance can't be silently mutated.
_lock = threading.Lock()
_registered = False
_cache: dict[tuple[str, tuple[str, ...]], DictConfig] = {}


def compose_cfg(
    dataset: str, overrides: list[str] | None = None, cache: bool = True
) -> DictConfig:
    """Compose the config for ``dataset`` (+ optional Hydra overrides).

    Composition is serialized and memoized: Hydra keeps global state, so doing it
    once per (dataset, overrides) and caching the result both avoids redundant
    work on every request and keeps repeated `initialize_config_dir` calls (which
    do not compose cleanly under concurrency) to a minimum.

    ``cache=False`` skips memoization for one-off compositions whose overrides
    are unique per call (a run launch carries a freshly minted ``run_name``, so
    caching those would grow the cache without ever hitting it).
    """
    global _registered
    key = (dataset, tuple(overrides or ()))
    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
        if not _registered:
            register_configs()
            _registered = True
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(config_dir()), version_base="1.3"):
            cfg = compose(
                config_name="config",
                overrides=[f"dataset={dataset}", *(overrides or [])],
            )
        OmegaConf.set_readonly(cfg, True)  # a cached instance is shared; fail on mutation
        if cache:
            _cache[key] = cfg
        return cfg


def compute_groups_for(dataset: str) -> dict[str, float]:
    """Live dimensionless groups for a dataset, computed from its config."""
    from naviernet.physics.groups import compute_groups

    return compute_groups(compose_cfg(dataset))
