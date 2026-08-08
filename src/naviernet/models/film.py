"""The liquid-film thickness net: ``delta(x, t)`` along the channel.

A 1-D auxiliary field evaluated on the SAME front samples the sharp-interface
conditions already use -- one small net, no new sampler. The coordinate is the
LAB-FRAME axial position, not the front's own spine parameter, for a physical
reason: the film is attached to the wall, and walls do not move. Measured on a
fully-trained run, the bubble's body RECEDES at essentially every station (the
capsule narrows as it elongates; only the nose advances), so the film beside a
flank was deposited by the nose PASSING that position, not by the flank's own
motion. In (x, t) that history is a boundary condition at the advancing front
plus a pure time derivative behind it; in front coordinates it would need an
advection term to undo the relabeling of stations as the bubble stretches.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from naviernet.models.layers import logit, mlp


class LiquidFilm(nn.Module):
    """Film thickness at axial position ``x`` and time ``t``, channel-bounded.

    ``delta = (H*/2) * sigmoid(net(x, t))``: strictly positive (a film exists
    until it dries out), bounded by the half-gap (the film cannot hold more
    liquid than the gap does), and able to approach zero -- which is what makes
    dryout reachable once depletion exists.

    Initialized at the deposition law's own value at the reference speed
    (``v_n* = 1``), the same data-anchored-start treatment the geometry nets
    get -- and doubly right here: stations the deposition condition does not
    reach (the receding body, until depletion exists) start near the value the
    advancing nose would have left there, rather than at half the gap, ~20x too
    thick.
    """

    def __init__(self, half_gap: float, delta_ref: float, n_cond: int = 0):
        super().__init__()
        if half_gap <= 0.0:
            raise ValueError(f"the half-gap must be positive, got {half_gap}")
        self.half_gap = float(half_gap)
        self.net = mlp(2 + int(n_cond), out_bias=logit(delta_ref / half_gap))

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = torch.cat([x, t], dim=1)
        if c is not None:
            features = torch.cat([features, c], dim=1)
        return self.half_gap * torch.sigmoid(self.net(features))
