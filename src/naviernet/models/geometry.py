"""The interface as the parameterized object (R3, ``model.front_geometry``).

Why this exists: with a free level-set net, extrapolated frames keep the bubble's
volume and the pinned root but lose its SHAPE -- the elongated capsule rounds
off (perimeter ratio decaying), goes wavy, and sheds spurious components,
because nothing off-data constrains the shape function. Here the shape is not
learned as a preference but imposed as the representation: phi is built from a
monotone nose position, a bounded width envelope, and a bounded centerline, so
a single connected capsule rooted at the measured cavity is the ONLY thing the
model can express, at any time.

Structural guarantees (each regression-tested):
- the interface passes exactly through the root point at every t;
- the nose never retreats, and extrapolates at its last learned rate;
- the vapour region is one connected capsule closed at both ends;
- the width is bounded by the channel and phi is C-infinity (the ``|y - c|``
  kink is smoothed -- Stage-B curvature differentiates phi twice, the same
  lesson the hard-pin gate learned).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Hidden layout of the three small geometry nets. Deliberately module constants,
# not config: the nets parameterize low-frequency curves (a rate, a width
# profile, a centerline), and capacity beyond this re-admits the wiggles the
# representation exists to forbid.
GEO_HIDDEN = 32
GEO_DEPTH = 2

# Nodes of the fixed time grid the nose rate is integrated on. Dense enough that
# the piecewise-linear nose resolves the training window; deterministic so
# resume rebuilds it exactly.
NOSE_GRID_NODES = 128

# The grid extends this far past the training window (as a fraction of it), so
# extrapolated queries still ride the learned rate before the constant-rate
# linear extension takes over.
NOSE_GRID_SLACK = 0.5

# Smoothing half-width of |y - c| (the y* analogue of the KAPPA_EPS lesson):
# far smaller than the interface half-thickness, so the interface location is
# unchanged, but phi stays twice-differentiable on the centerline.
ABS_SMOOTH = 1e-3


@dataclass(frozen=True)
class GeometryPriors:
    """Data-derived anchors the construction is built around: the measured root
    point, the measured first-training-frame front (the nose's initial value),
    and the domain bounds."""

    x_root: float
    y_root: float
    s0: float
    y_min: float
    y_max: float
    t_min: float
    t_max: float


def _mlp(in_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = [in_dim] + [GEO_HIDDEN] * GEO_DEPTH
    for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
        layers += [nn.Linear(d_in, d_out), nn.Tanh()]
    layers.append(nn.Linear(dims[-1], 1))
    return nn.Sequential(*layers)


class GeometricInterface(nn.Module):
    """``phi(x, y, t) = W(xi, t) - smoothabs(y - c(xi, t))`` with
    ``xi = (x - x_root) / (s(t) - x_root)``.

    ``s(t)`` is the nose: ``s0 + cumulative-trapezoid of softplus(rate(t))`` on
    the fixed grid (exactly monotone; linear extension beyond the grid).
    ``W`` is the half-width: channel-bounded sigmoid profile times the
    ``4 xi (1 - xi)`` envelope -- zero at root and nose, negative outside, so
    the capsule closes and nothing exists beyond it. ``c`` is the centerline,
    tanh-bounded inside the channel around the measured root height.

    Drop-in for the phi FieldNet: ``forward(x, c=None) -> (N, 1)``. Conditioned
    (joint) calls are rejected by the trainer before construction.
    """

    def __init__(self, priors: GeometryPriors):
        super().__init__()
        self.priors = priors
        self.rate_net = _mlp(1)
        self.width_net = _mlp(2)
        self.center_net = _mlp(2)
        # s(t_min) = x_root + softplus(_s0_raw): initialized so the nose starts
        # at the measured first-training-frame front.
        gap = max(priors.s0 - priors.x_root, 1e-3)
        self._s0_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(gap)))))

        span = priors.t_max - priors.t_min
        grid = torch.linspace(
            priors.t_min, priors.t_max + NOSE_GRID_SLACK * span, NOSE_GRID_NODES
        )
        self.register_buffer("t_grid", grid, persistent=False)

        half = 0.5 * (priors.y_max - priors.y_min)
        self._y_half = float(half)
        # Centerline amplitude: the root height must stay strictly inside the
        # channel, so the tanh swing is the smaller margin to either wall.
        self._c_amp = float(min(priors.y_root - priors.y_min, priors.y_max - priors.y_root))

    def nose(self, t: torch.Tensor) -> torch.Tensor:
        """Nose position s(t), shape-preserving for ``t`` of shape (N, 1).

        Exactly monotone: rates are non-negative and the cumulative trapezoid
        only accumulates; queries beyond the grid extend at the final rate.
        """
        grid = self.t_grid
        rates = torch.nn.functional.softplus(self.rate_net(grid.unsqueeze(1))).squeeze(1)
        steps = grid[1:] - grid[:-1]
        cum = torch.cat(
            [
                torch.zeros(1, device=grid.device),
                torch.cumsum(0.5 * (rates[1:] + rates[:-1]) * steps, dim=0),
            ]
        )

        tq = t.clamp(min=float(grid[0]))
        idx = (torch.searchsorted(grid, tq.reshape(-1), right=True) - 1).clamp(
            0, grid.numel() - 2
        )
        s = cum[idx] + (tq.reshape(-1) - grid[idx]) * rates[idx]
        s0 = self.priors.x_root + torch.nn.functional.softplus(self._s0_raw)
        return (s0 + s).reshape(t.shape)

    def centerline(self, xi: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raw = self.center_net(torch.cat([xi, t], dim=1))
        return self.priors.y_root + self._c_amp * torch.tanh(raw)

    def root_point(self, t: float) -> torch.Tensor:
        """The (x, y, t) point the interface passes through at the root -- the
        exact pin, for tests and diagnostics."""
        tt = torch.tensor([[float(t)]])
        with torch.no_grad():
            y = self.centerline(torch.zeros_like(tt), tt)
        return torch.tensor([self.priors.x_root, float(y), float(t)])

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if c is not None:
            raise NotImplementedError(
                "model.front_geometry does not support conditioned (joint) calls yet."
            )
        t = x[:, 2:3]
        span = (self.nose(t) - self.priors.x_root).clamp(min=1e-6)
        xi = (x[:, 0:1] - self.priors.x_root) / span

        width = self._y_half * torch.sigmoid(self.width_net(torch.cat([xi, t], dim=1)))
        envelope = 4.0 * xi * (1.0 - xi)  # 1 mid-capsule, 0 at the ends, negative outside
        offset = x[:, 1:2] - self.centerline(xi, t)
        smoothabs = torch.sqrt(offset**2 + ABS_SMOOTH**2) - ABS_SMOOTH
        return width * envelope - smoothabs
