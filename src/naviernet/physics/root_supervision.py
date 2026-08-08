"""Root-window supervision (nucleation-root R1, ``training.root_supervision``).

The bluntness DOF ``w(t)`` ships WITH its supervision channel, in the same
task, because freedom without a driving signal does nothing: ``cap_freedom``
was benched UNDRIVEN twice (1.06% of a 20% budget, then 1-2% of 30% under the
film recipe), and the diagnosis was structural -- the root's mismatch band is
thinner than the interface blur, and cap pixels are a sliver of the data
batch, so pixel losses scale with mismatch AREA and starve the knob. Both
terms here are built to survive that:

- ``root_sdf``: the measured mask's signed distance transform, sampled AT the
  predicted root-cap contour (Kervadec's boundary loss, MIDL 2019). Scales
  with mismatch DISTANCE, not area, so it stays informative inside the blur
  band. The distance rasters already exist (``data.sdf``); sampling is
  bilinear and differentiable in the sample position, whose gradient is the
  distance field's own unit direction -- a real direction, unlike the
  staircase a velocity raster gives (which is why front_velocity detaches its
  sampling and this term must not).
- ``root_bluntness``: the per-frame SCALAR measured by
  :func:`naviernet.physics.root.measured_root_fit` -- the half-width ratio at
  quarter- vs full-window depth -- regressed against the model's own ratio at
  the same absolute depths. A 1-D regression the loss cannot dilute away.

Like the kinematic and front-velocity terms, both sit OUTSIDE causal
weighting, RBA, and the gradient-norm rebalancer: they are supervised
functionals on the front, not pointwise residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from naviernet.data.dataset import BubbleDataset
from naviernet.models.geometry import ANCHOR_FLOOR
from naviernet.physics.front_velocity import time_index
from naviernet.physics.root import BLUNTNESS_DEPTH_FRACTION, measured_root_fit


@dataclass(frozen=True)
class RootContext:
    """Per-call inputs: the model to evaluate the front with, and the training
    config carrying the weights and the root sampling budget."""

    model: object
    tcfg: object


@dataclass(frozen=True)
class RootPlan:
    """One dataset's measured root targets, frozen before the training loop.

    Deterministic and built once (the front-velocity pattern), so a resumed run
    supervises against exactly the same measurement it was trained on. One
    slice per usable supervised frame: its time, its signed-distance raster,
    its measured bluntness, and its window depth (the absolute length the
    bluntness ratio is evaluated over -- the model must be read at the SAME
    depths to be comparable).
    """

    times: torch.Tensor  # (F, 1)
    sdf: torch.Tensor  # (F, H, W) signed distance, positive outside the vapour
    origin: tuple[float, float]  # (x*, y*) of pixel (0, 0)
    spacing: tuple[float, float]  # (dx*, dy*) per pixel
    bluntness: torch.Tensor  # (F, 1) measured half-width ratios
    depth: torch.Tensor  # (F, 1) measured root-window depths
    x_root: float  # the measured pin anchor the depths are anchored at
    rows: tuple[int, ...]  # the dataset rows each slice came from


def build_plan(data: BubbleDataset, cfg, device) -> RootPlan:
    """Measure this dataset's root targets over every supervised frame.

    Held-out rows are excluded -- supervising the root against a validation
    frame would leak it. Frames whose window is too thin to fit (NaN) are
    skipped; raising when none survive is what keeps a supervision flag from
    reporting success while contributing nothing.
    """
    held_out = set(data.validation_rows)
    rows, targets, depths = [], [], []
    for row in range(data.n_event):
        if row in held_out:
            continue
        fit = measured_root_fit(data, row)
        if not np.isfinite(fit.bluntness):
            continue
        rows.append(row)
        targets.append([fit.bluntness])
        depths.append([fit.window_depth])
    if not rows:
        raise ValueError(
            f"training.root_supervision needs at least one training-visible frame "
            f"whose root window fits, and dataset {data.cfg.dataset!r} has none. "
            f"Its event frames are {data.event_frames} and "
            f"{data.validation_frames} are held out of supervision."
        )
    x_root, _ = data.pin_anchor
    return RootPlan(
        times=torch.tensor(
            [[float(data.t[r])] for r in rows], dtype=torch.float32, device=device
        ),
        sdf=torch.tensor(
            np.stack([data.sdf[r] for r in rows]), dtype=torch.float32, device=device
        ),
        origin=(float(data.x[0]), float(data.y[0])),
        spacing=(float(data.x[1] - data.x[0]), float(data.y[1] - data.y[0])),
        bluntness=torch.tensor(targets, dtype=torch.float32, device=device),
        depth=torch.tensor(depths, dtype=torch.float32, device=device),
        x_root=float(x_root),
        rows=tuple(rows),
    )


def root_losses(ctx: RootContext, plan: RootPlan) -> tuple[torch.Tensor, dict[str, float]]:
    """The weighted root-supervision total for one step, plus per-term floats
    for the log. Recorded values are the raw (unweighted) magnitudes."""
    tcfg = ctx.tcfg
    sdf = _signed_distance_loss(ctx, plan)
    blunt = _bluntness_loss(ctx, plan)
    total = float(tcfg.root_sdf_weight) * sdf + float(tcfg.root_bluntness_weight) * blunt
    return total, {"root_sdf": float(sdf.detach()), "root_blunt": float(blunt.detach())}


def _signed_distance_loss(ctx: RootContext, plan: RootPlan) -> torch.Tensor:
    """Mean squared measured-distance at the predicted root-cap samples, as a
    fraction of the cap radius -- the same normalisation the diagnostics'
    ``model_root_rms`` uses, so the trained quantity IS the reported one."""
    geometry = ctx.model.nets["phi"] if hasattr(ctx.model, "nets") else ctx.model
    front = ctx.model.front(plan.times, n_body=1, n_cap=int(ctx.tcfg.root_cap_samples))
    on_root = (front.on_cap.squeeze(1) > 0) & (front.u.squeeze(1) < 0.5)
    points = front.points[on_root]
    frame_of = time_index(plan.times, points[:, 2:3])

    distance = _sample_sdf(plan, points, frame_of)
    radius = geometry.frame(plan.times).r_root.detach().abs().clamp(min=ANCHOR_FLOOR)
    return ((distance / radius[frame_of].squeeze(1)) ** 2).mean()


def _sample_sdf(plan: RootPlan, points: torch.Tensor, frame_of: torch.Tensor) -> torch.Tensor:
    """Bilinear samples of each frame's signed-distance raster at ``points``,
    DIFFERENTIABLE in the point positions: the gradient of a distance field is
    its unit direction toward the measured interface, which is exactly the
    direction the loss must push the contour. Clamped to the grid -- a root cap
    sits well inside the imaged domain.

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


def _bluntness_loss(ctx: RootContext, plan: RootPlan) -> torch.Tensor:
    """Squared error between the model's half-width ratio and the measured one,
    both at the measured window's own quarter- and full-depth stations."""
    geometry = ctx.model.nets["phi"] if hasattr(ctx.model, "nets") else ctx.model
    shallow = geometry.root_half_width(
        plan.x_root + BLUNTNESS_DEPTH_FRACTION * plan.depth, plan.times
    )
    deep = geometry.root_half_width(plan.x_root + plan.depth, plan.times)
    model_ratio = shallow / deep.clamp(min=ANCHOR_FLOOR)
    return ((model_ratio - plan.bluntness) ** 2).mean()
