"""Root-window supervision (nucleation-root R1, ``training.root_supervision``).

The bluntness DOF ``w(t)`` ships WITH its supervision channel, in the same
task, because freedom without a driving signal does nothing: ``cap_freedom``
was benched UNDRIVEN twice (1.06% of a 20% budget, then 1-2% of 30% under the
film recipe), and the diagnosis was structural -- the root's mismatch band is
thinner than the interface blur, and cap pixels are a sliver of the data
batch, so pixel losses scale with mismatch AREA and starve the knob. Both
terms here are built to survive that -- the plan, the SDF sampler and the
loss shape live in :mod:`naviernet.physics.cap_supervision`, shared with the
nose:

- ``root_sdf``: the measured mask's signed distance transform, sampled AT the
  predicted root-cap contour (Kervadec's boundary loss, MIDL 2019). Scales
  with mismatch DISTANCE, not area, so it stays informative inside the blur
  band.
- ``root_bluntness``: the per-frame SCALAR measured by
  :func:`naviernet.physics.root.measured_root_fit` -- the half-width ratio at
  quarter- vs full-window depth -- regressed against the model's own ratio at
  the same absolute depths (the root cap is PINNED at the measured anchor, so
  absolute stations are exact; the unpinned nose anchors at its own apex
  instead, see :mod:`naviernet.physics.nose_supervision`).

Like the kinematic and front-velocity terms, both sit OUTSIDE causal
weighting, RBA, and the gradient-norm rebalancer: they are supervised
functionals on the front, not pointwise residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from naviernet.data.dataset import BubbleDataset
from naviernet.models.geometry import ANCHOR_FLOOR
from naviernet.physics.cap_supervision import (
    SupervisionContext,
    WindowPlan,
    sample_sdf,
    window_plan_fields,
)
from naviernet.physics.front_velocity import time_index
from naviernet.physics.root import BLUNTNESS_DEPTH_FRACTION, measured_root_fit


@dataclass(frozen=True)
class RootPlan(WindowPlan):
    """The shared window plan plus the root's one extra fact: the measured pin
    anchor the absolute bluntness stations are measured inward from."""

    x_root: float


def build_plan(data: BubbleDataset, cfg, device) -> RootPlan:
    """Measure this dataset's root targets over every supervised frame."""
    x_root, _ = data.pin_anchor
    return RootPlan(
        **window_plan_fields(data, device, measured_root_fit, "root_supervision", "root"),
        x_root=float(x_root),
    )


def root_losses(
    ctx: SupervisionContext, plan: RootPlan
) -> tuple[torch.Tensor, dict[str, float]]:
    """The weighted root-supervision total for one step, plus per-term floats
    for the log. Recorded values are the raw (unweighted) magnitudes."""
    tcfg = ctx.tcfg
    sdf = _signed_distance_loss(ctx, plan)
    blunt = _bluntness_loss(ctx, plan)
    total = float(tcfg.root_sdf_weight) * sdf + float(tcfg.root_bluntness_weight) * blunt
    return total, {"root_sdf": float(sdf.detach()), "root_blunt": float(blunt.detach())}


def _signed_distance_loss(ctx: SupervisionContext, plan: RootPlan) -> torch.Tensor:
    """Mean squared measured-distance at the predicted root-cap samples, as a
    fraction of the cap radius -- the same normalisation the diagnostics'
    ``model_root_rms`` uses, so the trained quantity IS the reported one."""
    geometry = ctx.model.nets["phi"] if hasattr(ctx.model, "nets") else ctx.model
    front = ctx.model.front(plan.times, n_body=1, n_cap=int(ctx.tcfg.root_cap_samples))
    on_root = (front.on_cap.squeeze(1) > 0) & (front.u.squeeze(1) < 0.5)
    points = front.points[on_root]
    frame_of = time_index(plan.times, points[:, 2:3])

    distance = sample_sdf(plan, points, frame_of)
    radius = geometry.frame(plan.times).r_root.detach().abs().clamp(min=ANCHOR_FLOOR)
    return ((distance / radius[frame_of].squeeze(1)) ** 2).mean()


def _bluntness_loss(ctx: SupervisionContext, plan: RootPlan) -> torch.Tensor:
    """Squared error between the model's half-width ratio and the measured one,
    both at the measured window's own quarter- and full-depth stations."""
    geometry = ctx.model.nets["phi"] if hasattr(ctx.model, "nets") else ctx.model
    shallow = geometry.root_half_width(
        plan.x_root + BLUNTNESS_DEPTH_FRACTION * plan.depth, plan.times
    )
    deep = geometry.root_half_width(plan.x_root + plan.depth, plan.times)
    model_ratio = shallow / deep.clamp(min=ANCHOR_FLOOR)
    return ((model_ratio - plan.bluntness) ** 2).mean()
