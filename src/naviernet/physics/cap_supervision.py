"""Shared machinery for the per-cap window supervisions (root + nose).

The root and nose supervisions are one design: a frozen, deterministic plan of
per-frame measured targets (built once, so resume supervises against exactly
the measurement it trained on), a signed-distance term sampled AT the
predicted cap contour, and a per-frame bluntness-ratio regression. The
root/nose difference is which end's fit measures the targets and which cap's
contour is sampled -- so the plan's shape, the held-out exclusion, and the
differentiable SDF sampler live here, once, and the two supervision modules
keep only what genuinely differs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from naviernet.data.dataset import BubbleDataset


@dataclass(frozen=True)
class SupervisionContext:
    """Per-call inputs: the model to evaluate the front with, and the training
    config carrying the weights and the cap sampling budget."""

    model: object
    tcfg: object


@dataclass(frozen=True)
class WindowPlan:
    """One dataset's measured window targets for one end cap, frozen before
    the training loop. One slice per usable supervised frame: its time, its
    signed-distance raster, its measured bluntness, and its window depth (the
    model is read at the SAME depths to be comparable). The nose's targets
    legitimately exceed 1 (the overhang); nothing reading this may assume
    ratios <= 1."""

    times: torch.Tensor  # (F, 1)
    sdf: torch.Tensor  # (F, H, W) signed distance, positive outside the vapour
    origin: tuple[float, float]  # (x*, y*) of pixel (0, 0)
    spacing: tuple[float, float]  # (dx*, dy*) per pixel
    bluntness: torch.Tensor  # (F, 1) measured half-width ratios
    depth: torch.Tensor  # (F, 1) measured window depths
    rows: tuple[int, ...]  # the dataset rows each slice came from


def window_plan_fields(
    data: BubbleDataset, device, fit_of: Callable, flag: str, end: str
) -> dict:
    """The :class:`WindowPlan` fields, measured over every supervised frame --
    the shared core of both plan builders.

    Held-out rows are excluded -- supervising a cap against a validation frame
    would leak it. Frames whose window is too thin to fit (NaN) are skipped;
    raising when none survive is what keeps a supervision flag from reporting
    success while contributing nothing.
    """
    held_out = set(data.validation_rows)
    rows, targets, depths = [], [], []
    for row in range(data.n_event):
        if row in held_out:
            continue
        fit = fit_of(data, row)
        if not np.isfinite(fit.bluntness):
            continue
        rows.append(row)
        targets.append([fit.bluntness])
        depths.append([fit.window_depth])
    if not rows:
        raise ValueError(
            f"training.{flag} needs at least one training-visible frame whose "
            f"{end} window fits, and dataset {data.cfg.dataset!r} has none. "
            f"Its event frames are {data.event_frames} and "
            f"{data.validation_frames} are held out of supervision."
        )
    return {
        "times": torch.tensor(
            [[float(data.t[r])] for r in rows], dtype=torch.float32, device=device
        ),
        "sdf": torch.tensor(
            np.stack([data.sdf[r] for r in rows]), dtype=torch.float32, device=device
        ),
        "origin": (float(data.x[0]), float(data.y[0])),
        "spacing": (float(data.x[1] - data.x[0]), float(data.y[1] - data.y[0])),
        "bluntness": torch.tensor(targets, dtype=torch.float32, device=device),
        "depth": torch.tensor(depths, dtype=torch.float32, device=device),
        "rows": tuple(rows),
    }


def sample_sdf(plan: WindowPlan, points: torch.Tensor, frame_of: torch.Tensor) -> torch.Tensor:
    """Bilinear samples of each frame's signed-distance raster at ``points``,
    DIFFERENTIABLE in the point positions: the gradient of a distance field is
    its unit direction toward the measured interface, which is exactly the
    direction the loss must push the contour. Clamped to the grid -- an end
    cap sits well inside the imaged domain.

    One of three bilinear samplers, each with different gradient semantics:
    :meth:`naviernet.physics.root.Raster.sample` (numpy, diagnostics) and
    :func:`naviernet.physics.front_velocity.sample_measured` (torch, DETACHED
    -- a velocity raster's bilinear gradient is staircase noise, where a
    distance field's is the true direction). Do not merge them blindly."""
    _, height, width = plan.sdf.shape
    col = ((points[:, 0] - plan.origin[0]) / plan.spacing[0]).clamp(0.0, width - 1.0)
    row = ((points[:, 1] - plan.origin[1]) / plan.spacing[1]).clamp(0.0, height - 1.0)
    col_lo = col.detach().floor().clamp(0, width - 2).long()
    row_lo = row.detach().floor().clamp(0, height - 2).long()
    fx = (col - col_lo).clamp(0.0, 1.0)
    fy = (row - row_lo).clamp(0.0, 1.0)

    flat = plan.sdf.reshape(-1)
    base = frame_of * (height * width) + row_lo * width + col_lo
    return (
        flat[base] * (1 - fx) * (1 - fy)
        + flat[base + 1] * fx * (1 - fy)
        + flat[base + width] * (1 - fx) * fy
        + flat[base + width + 1] * fx * fy
    )
