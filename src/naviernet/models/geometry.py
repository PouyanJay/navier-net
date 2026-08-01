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
- the caps are CIRCULAR (constant curvature -- the Young-Laplace cap shape),
  the width is bounded by the channel, and phi is smooth on the spine (the
  matched-floor form; Stage-B curvature differentiates phi twice, the same
  lesson the hard-pin gate learned). At the cap-body seams the u-clamp leaves
  phi C0-but-not-C1: a measured, accepted trade-off (kappa*grad(alpha) stays
  O(10-50) there, far under harmful scale; a C1 blend would require tying the
  boundary radius slope to zero and cost expressivity) -- regression-tested at
  the seam points.
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
    """Data-derived anchors the construction is built around AND initializes at:
    the measured root point, first-frame front and half-width, and the measured
    front speed. A saturating interface starves gradients when the initial
    capsule sits far from the true one (alpha is 0/1 more than ~eps away), so
    the construction starts as a data-shaped moving capsule, not noise."""

    x_root: float
    y_root: float
    s0: float
    w0: float
    rate0: float
    y_min: float
    y_max: float
    t_min: float
    t_max: float


def _mlp(in_dim: int, out_bias: float = 0.0) -> nn.Sequential:
    """A small tanh MLP whose LAST layer starts at zero weights and the given
    bias: the net begins as the constant ``out_bias`` and learns deviations --
    the data-anchored initialization the priors provide."""
    layers: list[nn.Module] = []
    dims = [in_dim] + [GEO_HIDDEN] * GEO_DEPTH
    for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
        layers += [nn.Linear(d_in, d_out), nn.Tanh()]
    last = nn.Linear(dims[-1], 1)
    # Small (not zero: exact zeros would block gradient into the hidden layers)
    # so the net starts within a hair of the constant and can still learn.
    nn.init.normal_(last.weight, std=0.01)
    nn.init.constant_(last.bias, out_bias)
    layers.append(last)
    return nn.Sequential(*layers)


def _inverse_softplus(value: float) -> float:
    value = max(value, 1e-6)
    return float(torch.log(torch.expm1(torch.tensor(value))))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(p)))


class GeometricInterface(nn.Module):
    """A varying-radius capsule: a spine from the root apex to the nose apex,
    inflated by a channel-bounded radius profile, closed by CIRCULAR caps.

    ``s(t)`` is the nose: ``s0 + cumulative-trapezoid of softplus(rate(t))`` on
    the fixed grid (exactly monotone; linear extension beyond the grid). The
    radius ``R(u, t)`` and centerline ``c(u, t)`` are small nets over the spine
    parameter; cap centers sit one radius inside each apex so the interface
    passes exactly through the pinned root and the nose (see ``forward``).

    Drop-in for the phi FieldNet: ``forward(x, c=None) -> (N, 1)``. Conditioned
    (joint) calls are rejected by the trainer before construction.
    """

    def __init__(self, priors: GeometryPriors):
        super().__init__()
        self.priors = priors
        half = 0.5 * (priors.y_max - priors.y_min)
        self._y_half = float(half)
        # Centerline amplitude: the root height must stay strictly inside the
        # channel, so the tanh swing is the smaller margin to either wall.
        self._c_amp = float(min(priors.y_root - priors.y_min, priors.y_max - priors.y_root))

        # Data-anchored start: rate = measured front speed, width = measured
        # first-frame half-width (at the envelope's midpoint), centerline flat
        # at the measured root height.
        self.rate_net = _mlp(1, out_bias=_inverse_softplus(priors.rate0))
        self.width_net = _mlp(2, out_bias=_logit(min(priors.w0 / self._y_half, 1.0)))
        self.center_net = _mlp(2, out_bias=0.0)
        # s(t_min) = x_root + softplus(_s0_raw): initialized so the nose starts
        # at the measured first-training-frame front.
        gap = max(priors.s0 - priors.x_root, 1e-3)
        self._s0_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(gap)))))

        span = priors.t_max - priors.t_min
        grid = torch.linspace(
            priors.t_min, priors.t_max + NOSE_GRID_SLACK * span, NOSE_GRID_NODES
        )
        self.register_buffer("t_grid", grid, persistent=False)

    def nose(self, t: torch.Tensor) -> torch.Tensor:
        """Nose position s(t), shape-preserving for ``t`` of shape (N, 1).

        Exactly monotone: rates are non-negative and the cumulative trapezoid
        only accumulates; queries beyond the grid extend at the final rate.
        """
        grid = self.t_grid
        rates = torch.nn.functional.softplus(self.rate_net(grid.unsqueeze(1))).squeeze(1)
        steps = grid[1:] - grid[:-1]
        # Segment slopes ARE the trapezoid averages: interpolating the cumulative
        # array with them is continuous at every node and exactly monotone
        # (slopes >= 0). Interpolating with the raw nodal rate instead creates
        # node discontinuities that go NEGATIVE once the rate net trains away
        # from its flat init -- the reviewed, reproduced failure of the very
        # guarantee this class exists for.
        slopes = 0.5 * (rates[1:] + rates[:-1])
        cum = torch.cat(
            [torch.zeros(1, device=grid.device), torch.cumsum(slopes * steps, dim=0)]
        )

        tq = t.clamp(min=float(grid[0]))
        idx = (torch.searchsorted(grid, tq.reshape(-1), right=True) - 1).clamp(
            0, grid.numel() - 2
        )
        s = cum[idx] + (tq.reshape(-1) - grid[idx]) * slopes[idx]
        s0 = self.priors.x_root + torch.nn.functional.softplus(self._s0_raw)
        return (s0 + s).reshape(t.shape)

    def centerline(self, xi: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raw = self.center_net(torch.cat([xi, t], dim=1))
        return self.priors.y_root + self._c_amp * torch.tanh(raw)

    def root_point(self, t: float) -> torch.Tensor:
        """The (x, y, t) point the interface passes through at the root -- the
        exact pin, for tests and diagnostics."""
        device = self.t_grid.device
        tt = torch.tensor([[float(t)]], device=device)
        with torch.no_grad():
            y = self.centerline(torch.zeros_like(tt), tt)
        return torch.tensor([self.priors.x_root, float(y), float(t)], device=device)

    def nose_point(self, t: float) -> torch.Tensor:
        """The (x, y, t) point the interface passes through at the nose -- the
        capsule's far closure, ``root_point``'s mirror."""
        device = self.t_grid.device
        tt = torch.tensor([[float(t)]], device=device)
        with torch.no_grad():
            s = self.nose(tt)
            y = self.centerline(torch.ones_like(tt), tt)
        return torch.tensor([float(s), float(y), float(t)], device=device)

    def _radius(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Half-width (inflation radius) profile along the spine, channel-bounded."""
        return self._y_half * torch.sigmoid(self.width_net(torch.cat([u, t], dim=1)))

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """Varying-radius capsule: the spine from the root apex to the nose apex,
        inflated by the radius profile, with CIRCULAR end caps.

        The caps are the physics: a bubble cap is constant-curvature
        (Young-Laplace), i.e. a circular arc -- the earlier ``4 xi (1 - xi)``
        envelope closed the ends linearly and produced wedge-shaped tips the
        real bubble never shows. Cap centers sit one radius inside each apex, so
        the interface passes EXACTLY through the pinned root and the monotone
        nose (phi compares ``sqrt(R^2 + d^2_floor)`` against the smoothed
        distance, which cancels the floor identically on the interface).
        """
        if c is not None:
            raise NotImplementedError(
                "model.front_geometry does not support conditioned (joint) calls yet."
            )
        t = x[:, 2:3]
        s = self.nose(t)
        r_root = self._radius(torch.zeros_like(t), t)
        r_nose = self._radius(torch.ones_like(t), t)

        # Cap centers sit one radius inside each apex. When the bubble is
        # shorter than the two cap radii (a just-nucleated bubble -- realistic,
        # review-reproduced), the raw centers would CROSS the opposite apex and
        # break exact nose closure; rescaling both radii jointly keeps the caps
        # reaching exactly x_root and s while preserving a minimum spine
        # segment of ABS_SMOOTH, which also bounds du/dx (and with it alpha_x
        # in the VOF residual) in the degenerate regime.
        length = (s - self.priors.x_root).clamp(min=2.0 * ABS_SMOOTH)
        scale = ((length - ABS_SMOOTH) / (r_root + r_nose)).clamp(max=1.0)
        r_root = r_root * scale
        r_nose = r_nose * scale
        ax = self.priors.x_root + r_root
        bx = s - r_nose
        u = ((x[:, 0:1] - ax) / (bx - ax)).clamp(0.0, 1.0)

        # The same scale applies along the whole spine, so the cap radii the
        # centers were placed with are exactly the radii the field compares
        # against -- apex exactness survives the degenerate rescale.
        radius = self._radius(u, t) * scale
        spine_x = ax + u * (bx - ax)
        spine_y = self.centerline(u, t)
        d_sq = (x[:, 0:1] - spine_x) ** 2 + (x[:, 1:2] - spine_y) ** 2
        # Matched floors: phi = 0 exactly where d = R (the interface, incl. both
        # apexes), while staying C-infinity on the spine (the KAPPA lesson).
        return torch.sqrt(radius**2 + ABS_SMOOTH**2) - torch.sqrt(d_sq + ABS_SMOOTH**2)
