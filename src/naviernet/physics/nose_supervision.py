"""Nose-window supervision (``training.nose_supervision``).

The bluntness DOF ``w_nose(t)`` ships WITH its supervision channel, in the
same task, because freedom without a driving signal does nothing: this exact
nose was ``cap_freedom``'s target, and cap_freedom was benched UNDRIVEN twice
(1.06% of a 20% budget, then 1-2% of 30% under the film recipe) -- the cap's
mismatch band is thinner than the interface blur and cap pixels are a sliver
of the batch, so pixel losses scale with mismatch AREA and starve the knob.
The nucleation-root journey fixed the root by shipping the DOF and its signal
together (root RMS 0.127 -> 0.040), and this module is that channel's mirror
for the head -- the plan, the SDF sampler and the loss shape live in
:mod:`naviernet.physics.cap_supervision`, shared with the root:

- ``nose_sdf``: the measured mask's signed distance transform, sampled AT the
  predicted nose-cap contour (Kervadec's boundary loss, MIDL 2019). Scales
  with mismatch DISTANCE, not area, so it stays informative inside the blur
  band.
- ``nose_bluntness``: the per-frame SCALAR measured by
  :func:`naviernet.physics.nose.measured_nose_fit`, regressed against the
  model's own half-width ratio at the same depths. The nose targets
  legitimately EXCEED 1 (the overhang -- the measured drift is 0.61 -> 1.83);
  nothing here assumes ratios <= 1, and nothing may.

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
from naviernet.physics.nose import measured_nose_fit
from naviernet.physics.root import BLUNTNESS_DEPTH_FRACTION


def build_plan(data: BubbleDataset, cfg, device) -> WindowPlan:
    """Measure this dataset's nose targets over every supervised frame -- the
    shared window plan, with no extra anchor: the nose's stations follow the
    model's own apex (see the module docstring)."""
    return WindowPlan(
        **window_plan_fields(data, device, measured_nose_fit, "nose_supervision", "nose")
    )


def nose_losses(
    ctx: SupervisionContext, plan: WindowPlan
) -> tuple[torch.Tensor, dict[str, float]]:
    """The weighted nose-supervision total for one step, plus per-term floats
    for the log. Recorded values are the raw (unweighted) magnitudes."""
    tcfg = ctx.tcfg
    sdf = _signed_distance_loss(ctx, plan)
    blunt = _bluntness_loss(ctx, plan)
    total = float(tcfg.nose_sdf_weight) * sdf + float(tcfg.nose_bluntness_weight) * blunt
    return total, {"nose_sdf": float(sdf.detach()), "nose_blunt": float(blunt.detach())}


def _signed_distance_loss(ctx: SupervisionContext, plan: WindowPlan) -> torch.Tensor:
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


def _bluntness_loss(ctx: SupervisionContext, plan: WindowPlan) -> torch.Tensor:
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
