"""Nose-window supervision (``training.nose_supervision``).

The bluntness DOF ``w_nose(t)`` ships WITH its supervision channel, in the
same task, because freedom without a driving signal does nothing: this exact
nose was ``cap_freedom``'s target, and cap_freedom was benched UNDRIVEN twice
(1.06% of a 20% budget, then 1-2% of 30% under the film recipe) -- the cap's
mismatch band is thinner than the interface blur and cap pixels are a sliver
of the batch, so pixel losses scale with mismatch AREA and starve the knob.
The nucleation-root journey fixed the root by shipping the DOF and its signal
together (root RMS 0.127 -> 0.040), and this module is that channel's mirror
for the head:

- ``nose_sdf``: the measured mask's signed distance transform, sampled AT the
  predicted nose-cap contour (Kervadec's boundary loss, MIDL 2019). Scales
  with mismatch DISTANCE, not area, so it stays informative inside the blur
  band. The sampler is :func:`naviernet.physics.root_supervision.sample_sdf`
  itself -- differentiable in the sample position, whose gradient is the
  distance field's own unit direction.
- ``nose_bluntness``: the per-frame SCALAR measured by
  :func:`naviernet.physics.nose.measured_nose_fit`, regressed against the
  model's own half-width ratio at the same absolute stations. The nose
  targets legitimately EXCEED 1 (the overhang -- the measured drift is
  0.61 -> 1.83); nothing here assumes ratios <= 1, and nothing may.

One real difference from the root: the root cap is PINNED at the measured
anchor, so its bluntness stations can be absolute positions. The nose is not
pinned -- the model's apex floats near the measured edge -- and anchoring the
stations at the MEASURED edge lets any axial offset leak into the shape ratio
(measured: on a circular-nose control the loss bought the offset down with a
spurious w_nose of -0.58). So the model's ratio is read at the measured
window's DEPTHS but anchored at the model's own (detached) apex: bluntness is
a shape property, and the position stays owned by the sdf and data terms.

Like the kinematic, front-velocity and root terms, both sit OUTSIDE causal
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
from naviernet.physics.nose import measured_nose_fit
from naviernet.physics.root import BLUNTNESS_DEPTH_FRACTION
from naviernet.physics.root_supervision import sample_sdf


@dataclass(frozen=True)
class NoseContext:
    """Per-call inputs: the model to evaluate the front with, and the training
    config carrying the weights and the nose sampling budget."""

    model: object
    tcfg: object


@dataclass(frozen=True)
class NosePlan:
    """One dataset's measured nose targets, frozen before the training loop.

    Deterministic and built once (the front-velocity pattern), so a resumed
    run supervises against exactly the same measurement it was trained on.
    One slice per usable supervised frame: its time, its signed-distance
    raster, its measured bluntness, and its window depth (the model is read at
    the SAME depths -- from its own apex -- to be comparable).
    """

    times: torch.Tensor  # (F, 1)
    sdf: torch.Tensor  # (F, H, W) signed distance, positive outside the vapour
    origin: tuple[float, float]  # (x*, y*) of pixel (0, 0)
    spacing: tuple[float, float]  # (dx*, dy*) per pixel
    bluntness: torch.Tensor  # (F, 1) measured ratios -- above 1 late, by design
    depth: torch.Tensor  # (F, 1) measured nose-window depths
    rows: tuple[int, ...]  # the dataset rows each slice came from


def build_plan(data: BubbleDataset, cfg, device) -> NosePlan:
    """Measure this dataset's nose targets over every supervised frame.

    Held-out rows are excluded -- supervising the nose against a validation
    frame would leak it. Frames whose window is too thin to fit (NaN) are
    skipped; raising when none survive is what keeps a supervision flag from
    reporting success while contributing nothing.
    """
    held_out = set(data.validation_rows)
    rows, targets, depths = [], [], []
    for row in range(data.n_event):
        if row in held_out:
            continue
        fit = measured_nose_fit(data, row)
        if not np.isfinite(fit.bluntness):
            continue
        rows.append(row)
        targets.append([fit.bluntness])
        depths.append([fit.window_depth])
    if not rows:
        raise ValueError(
            f"training.nose_supervision needs at least one training-visible frame "
            f"whose nose window fits, and dataset {data.cfg.dataset!r} has none. "
            f"Its event frames are {data.event_frames} and "
            f"{data.validation_frames} are held out of supervision."
        )
    return NosePlan(
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
        rows=tuple(rows),
    )


def nose_losses(ctx: NoseContext, plan: NosePlan) -> tuple[torch.Tensor, dict[str, float]]:
    """The weighted nose-supervision total for one step, plus per-term floats
    for the log. Recorded values are the raw (unweighted) magnitudes."""
    tcfg = ctx.tcfg
    sdf = _signed_distance_loss(ctx, plan)
    blunt = _bluntness_loss(ctx, plan)
    total = float(tcfg.nose_sdf_weight) * sdf + float(tcfg.nose_bluntness_weight) * blunt
    return total, {"nose_sdf": float(sdf.detach()), "nose_blunt": float(blunt.detach())}


def _signed_distance_loss(ctx: NoseContext, plan: NosePlan) -> torch.Tensor:
    """Mean squared measured-distance at the predicted NOSE-cap samples, as a
    fraction of the nose-cap radius -- the same normalisation the diagnostics'
    ``model_nose_rms`` uses, so the trained quantity IS the reported one."""
    geometry = ctx.model.nets["phi"] if hasattr(ctx.model, "nets") else ctx.model
    front = ctx.model.front(plan.times, n_body=1, n_cap=int(ctx.tcfg.nose_cap_samples))
    on_nose = (front.on_cap.squeeze(1) > 0) & (front.u.squeeze(1) > 0.5)
    points = front.points[on_nose]
    frame_of = time_index(plan.times, points[:, 2:3])

    distance = sample_sdf(plan, points, frame_of)
    radius = geometry.frame(plan.times).r_nose.detach().abs().clamp(min=ANCHOR_FLOOR)
    return ((distance / radius[frame_of].squeeze(1)) ** 2).mean()


def _bluntness_loss(ctx: NoseContext, plan: NosePlan) -> torch.Tensor:
    """Squared error between the model's half-width ratio and the measured
    one, both at the measured window's own quarter- and full-depth stations,
    measured inward from the model's OWN apex -- detached, so this loss
    shapes the head rather than dragging the nose (the module docstring's
    measured coordinate trap)."""
    geometry = ctx.model.nets["phi"] if hasattr(ctx.model, "nets") else ctx.model
    anchor = geometry.nose(plan.times).detach()
    shallow = geometry.nose_half_width(
        anchor - BLUNTNESS_DEPTH_FRACTION * plan.depth, plan.times
    )
    deep = geometry.nose_half_width(anchor - plan.depth, plan.times)
    model_ratio = shallow / deep.clamp(min=ANCHOR_FLOOR)
    return ((model_ratio - plan.bluntness) ** 2).mean()
