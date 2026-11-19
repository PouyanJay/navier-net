"""The inlet's flow over time (``model.unsteady_inflow``).

A growing bubble pushes back on the liquid feeding the channel, and what is
held back upstream returns once the growth eases, so the flow through the
inlet swings about the rate the pump is set to. A steady inlet makes the liquid
beside the bubble's head absorb all of that: on Series-1-w300 at 15 mL/h it is
forced backward past the nose at up to 0.4 m/s in the last frames, and the
momentum law turns that into about 200 Pa at the nose tip that the measured
outline does not show (ghost8, ghost10).

The curve here is the inflow as a multiple of the set flow rate: one smooth
function of time the run trains with its fields. It is a cubic B-spline in
time passed through a sigmoid, so it changes gradually and never leaves the
band it is given.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# The state-dict prefix of the curve: the trainer recognises a checkpoint saved
# without it (a shared warm-up, or a steady run continued with the curve).
INFLOW_PREFIX = "inflow_curve."

# Adam moves a parameter about one learning rate per step whatever its
# gradient's size, and the physics stage ends at rates near 1e-4. At this gain
# a coefficient crosses most of the band in about a thousand such steps, while
# a step's jitter stays under 0.1% of the band.
COEFFICIENT_GAIN = 8.0

# When the set flow sits on an end of the band (an inflow that may only fall,
# say), the curve starts this share of the band inside it: the sigmoid cannot
# sit on its own limit.
EDGE_SHARE = 0.05

# Instants of the quadrature behind :meth:`InflowCurve.mean` and
# :meth:`InflowCurve.running_mean`.
MEAN_INSTANTS = 129


def _cardinal_cubic(u: torch.Tensor) -> torch.Tensor:
    """The cubic B-spline on unit knots, centred on zero: positive on ``|u| < 2``
    and, summed over the integers, one everywhere."""
    distance = u.abs()
    return (torch.relu(2.0 - distance) ** 3 - 4.0 * torch.relu(1.0 - distance) ** 3) / 6.0


class InflowCurve(nn.Module):
    """The inflow over the run's time window, as a multiple of the set flow.

    ``intervals`` equal spans of the window ``[t_min, t_max]`` carry the
    spline; ``lowest`` and ``highest`` are the band, as multiples of the set
    flow. The curve starts at the set flow (multiple one) at every instant.
    Outside the window it holds the value at the nearer end.
    """

    def __init__(
        self, t_min: float, t_max: float, intervals: int, lowest: float, highest: float
    ):
        super().__init__()
        if not t_max > t_min:
            raise ValueError(f"the inflow curve needs a time window, got [{t_min}, {t_max}]")
        if intervals < 1:
            raise ValueError(f"the inflow curve needs at least one interval, got {intervals}")
        if not lowest < highest:
            raise ValueError(
                f"the inflow's lowest multiple ({lowest}) must be below its highest ({highest})"
            )
        if not lowest <= 1.0 <= highest:
            raise ValueError(
                f"the inflow's band [{lowest}, {highest}] must hold the set flow (multiple 1)"
            )
        self.t_min, self.t_max = float(t_min), float(t_max)
        self.lowest, self.highest = float(lowest), float(highest)
        self.spacing = (self.t_max - self.t_min) / intervals
        # One knot beyond each end, so every instant of the window has its four
        # splines and they sum to one.
        knots = self.t_min + self.spacing * torch.arange(-1, intervals + 2, dtype=torch.float32)
        self.register_buffer("knots", knots, persistent=False)
        share = (1.0 - self.lowest) / (self.highest - self.lowest)
        share = min(max(share, EDGE_SHARE), 1.0 - EDGE_SHARE)
        self.start = math.log(share / (1.0 - share))
        self.coefficients = nn.Parameter(torch.zeros(len(knots)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """The multiple of the set flow at each instant ``t`` ``(N, 1)``, ``(N, 1)``."""
        inside = t.clamp(self.t_min, self.t_max)
        basis = _cardinal_cubic((inside - self.knots) / self.spacing)
        level = self.start + COEFFICIENT_GAIN * basis @ self.coefficients.reshape(-1, 1)
        return self.lowest + (self.highest - self.lowest) * torch.sigmoid(level)

    def mean(self) -> torch.Tensor:
        """The curve's average over the window, a scalar."""
        instants = self._quadrature_instants()
        return torch.trapezoid(self(instants.reshape(-1, 1))[:, 0], instants) / (
            self.t_max - self.t_min
        )

    def running_mean(self, span: float) -> torch.Tensor:
        """The curve's average over every stretch of length ``span`` inside the
        window, ``(M,)``: one per quadrature instant a stretch can start at. A
        span as long as the window gives the one average of :meth:`mean`."""
        if not span > 0.0:
            raise ValueError(f"the inflow's averaging span must be positive, got {span}")
        if span >= self.t_max - self.t_min:
            return self.mean().reshape(1)
        instants = self._quadrature_instants()
        values = self(instants.reshape(-1, 1))[:, 0]
        step = (self.t_max - self.t_min) / (MEAN_INSTANTS - 1)
        integral = torch.cat([values.new_zeros(1), torch.cumulative_trapezoid(values, dx=step)])
        starts = int(math.floor((self.t_max - self.t_min - span) / step + 1e-6)) + 1
        # A stretch ends between two instants: the integral there is read off
        # the line between them.
        position = torch.arange(starts, device=values.device) + span / step
        below = position.floor().clamp(max=MEAN_INSTANTS - 2).long()
        share = position - below
        at_end = integral[below] + share * (integral[below + 1] - integral[below])
        return (at_end - integral[:starts]) / span

    def _quadrature_instants(self) -> torch.Tensor:
        return torch.linspace(
            self.t_min, self.t_max, MEAN_INSTANTS, device=self.coefficients.device
        )
