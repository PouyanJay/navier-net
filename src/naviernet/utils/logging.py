"""Logging helpers.

Stages log through the standard :mod:`logging` module rather than printing, so
Hydra captures a complete transcript of every run to
``outputs/<run_name>/<job>.log`` alongside its results.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_basic_logging(level: int = logging.INFO) -> None:
    """Set up console logging for use outside a Hydra job (notebooks, tests)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )


def format_mapping(d: dict, fmt: str = "{:.4g}") -> str:
    """Compact one-line rendering of a name -> number mapping for logs."""
    return "  ".join(
        f"{k}={fmt.format(v)}" if isinstance(v, (int, float)) else f"{k}={v}"
        for k, v in d.items()
    )
