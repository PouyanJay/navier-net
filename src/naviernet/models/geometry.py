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

import math
from dataclasses import dataclass
from typing import NamedTuple

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


class CapsuleFrame(NamedTuple):
    """The capsule's per-time scalars: the nose, the two cap centres and radii,
    and the degenerate-case rescale. The one place the construction's geometry is
    derived, shared by ``forward`` (which builds phi) and ``front`` (which samples
    the interface), so the field and its explicit front can never drift apart."""

    s: torch.Tensor  # nose position
    ax: torch.Tensor  # root cap centre, x
    bx: torch.Tensor  # nose cap centre, x
    r_root: torch.Tensor  # root cap radius (already rescaled)
    r_nose: torch.Tensor  # nose cap radius (already rescaled)
    scale: torch.Tensor  # joint radius rescale for a shorter-than-its-caps bubble


class FrontSamples(NamedTuple):
    """Points sampled exactly ON the interface, with the parameters they came
    from. ``points`` is ``(N, 3)`` ordered ``(x, y, t)``; the rest are ``(N, 1)``.

    ``side`` is +1 on the upper profile and -1 on the lower one; ``on_cap`` marks
    the circular end caps, where the profile parameterization ``y = c +/- R`` does
    not apply and the curvature is the cap's own constant instead.
    """

    points: torch.Tensor
    u: torch.Tensor
    side: torch.Tensor
    on_cap: torch.Tensor


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

    def frame(self, t: torch.Tensor) -> CapsuleFrame:
        """The capsule's scalars at times ``t`` of shape ``(N, 1)``.

        Cap centers sit one radius inside each apex. When the bubble is shorter
        than the two cap radii (a just-nucleated bubble -- realistic,
        review-reproduced), the raw centers would CROSS the opposite apex and
        break exact nose closure; rescaling both radii jointly keeps the caps
        reaching exactly ``x_root`` and ``s`` while preserving a minimum spine
        segment of ``ABS_SMOOTH``, which also bounds ``du/dx`` (and with it
        ``alpha_x`` in the VOF residual) in the degenerate regime.
        """
        s = self.nose(t)
        r_root = self._radius(torch.zeros_like(t), t)
        r_nose = self._radius(torch.ones_like(t), t)
        length = (s - self.priors.x_root).clamp(min=2.0 * ABS_SMOOTH)
        scale = ((length - ABS_SMOOTH) / (r_root + r_nose)).clamp(max=1.0)
        r_root = r_root * scale
        r_nose = r_nose * scale
        return CapsuleFrame(s, self.priors.x_root + r_root, s - r_nose, r_root, r_nose, scale)

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
        f = self.frame(t)
        u = ((x[:, 0:1] - f.ax) / (f.bx - f.ax)).clamp(0.0, 1.0)

        # The same scale applies along the whole spine, so the cap radii the
        # centers were placed with are exactly the radii the field compares
        # against -- apex exactness survives the degenerate rescale.
        radius = self._radius(u, t) * f.scale
        spine_x = f.ax + u * (f.bx - f.ax)
        spine_y = self.centerline(u, t)
        d_sq = (x[:, 0:1] - spine_x) ** 2 + (x[:, 1:2] - spine_y) ** 2
        # Matched floors: phi = 0 exactly where d = R (the interface, incl. both
        # apexes), while staying C-infinity on the spine (the KAPPA lesson).
        return torch.sqrt(radius**2 + ABS_SMOOTH**2) - torch.sqrt(d_sq + ABS_SMOOTH**2)

    def front_curvature(self, front: FrontSamples) -> torch.Tensor:
        """In-plane curvature at each front sample ``(N, 1)``, positive where the
        vapour region is convex -- the sign convention the diffuse
        :func:`~naviernet.physics.residuals.curvature` uses.

        Taken from the parameterization in closed form rather than by
        differentiating a smeared alpha twice: the diffuse route has to floor
        ``|grad alpha|`` (``KAPPA_EPS``) and, on this construction, spikes to
        |kappa| ~ 1e4 on the spine, where a handful of points then carry most of
        the momentum loss. There is nothing to floor here.

        On the caps the curvature is exactly the cap's own ``1/r`` -- a circular
        arc is constant-curvature, which is why the caps are circles at all.
        """
        cap = 1.0 / self._cap_radius(front)
        return front.on_cap * cap

    def _cap_radius(self, front: FrontSamples) -> torch.Tensor:
        """The cap radius each sample belongs to: the root cap's at ``u = 0``, the
        nose cap's at ``u = 1``. Off the caps the value is unused but must stay
        finite and non-zero, so the frame's radii are selected, never divided into
        by zero."""
        f = self.frame(front.points[:, 2:3])
        return torch.where(front.u < 0.5, f.r_root, f.r_nose)

    def front(self, t: torch.Tensor, n_body: int, n_cap: int) -> FrontSamples:
        """Points sampled exactly ON the interface at each time in ``t``.

        This is what makes the sharp-interface conditions possible: the front is
        an explicit object, so the Young-Laplace jump and the kinematic condition
        can be imposed AT the interface instead of being diffused into bulk
        collocation residuals -- where, away from the interface, they are
        trivially satisfied and constrain nothing.

        Four segments per time: the upper and lower body profiles
        ``y = c(u,t) +/- R(u,t)``, and the two circular caps swept in angle. The
        segments meet exactly at the seams (the cap radius IS the profile's end
        radius), and every point satisfies ``phi = 0`` identically -- see
        :func:`forward`, whose matched floors cancel on the interface.

        Deliberately NOT detached: the loss must be able to move the front.
        """
        if n_body < 1 or n_cap < 2:
            raise ValueError(
                f"front() needs n_body >= 1 and n_cap >= 2 (a cap is an arc, not a "
                f"point), got n_body={n_body}, n_cap={n_cap}"
            )
        t = t.reshape(-1, 1)
        f = self.frame(t)

        # Body: the two profiles, off the seams so each u belongs to one segment.
        u = torch.linspace(0.0, 1.0, n_body + 2, device=t.device)[1:-1].reshape(1, -1)
        u_body = u.expand(t.shape[0], -1).reshape(-1, 1)
        t_body = t.expand(-1, n_body).reshape(-1, 1)
        radius = self._radius(u_body, t_body) * f.scale.repeat_interleave(n_body, dim=0)
        spine_x = (f.ax + u_body.reshape(t.shape[0], -1) * (f.bx - f.ax)).reshape(-1, 1)
        spine_y = self.centerline(u_body, t_body)

        # Caps: the root sweeps the far half-circle, the nose the near one, so
        # together with the two profiles they close the contour exactly once.
        angle = torch.linspace(-0.5 * math.pi, 0.5 * math.pi, n_cap, device=t.device)
        angle = angle.reshape(1, -1).expand(t.shape[0], -1).reshape(-1, 1)
        t_cap = t.expand(-1, n_cap).reshape(-1, 1)
        zeros, ones = torch.zeros_like(t_cap), torch.ones_like(t_cap)

        def cap(centre_x, radius_cap, u_end, sign):
            r = radius_cap.repeat_interleave(n_cap, dim=0)
            cx = centre_x.repeat_interleave(n_cap, dim=0)
            cy = self.centerline(u_end, t_cap)
            return torch.cat(
                [cx + sign * r * torch.cos(angle), cy + r * torch.sin(angle)], dim=1
            )

        root_cap = cap(f.ax, f.r_root, zeros, -1.0)
        nose_cap = cap(f.bx, f.r_nose, ones, +1.0)

        points = torch.cat(
            [
                torch.cat([spine_x, spine_y + radius, t_body], dim=1),
                torch.cat([spine_x, spine_y - radius, t_body], dim=1),
                torch.cat([root_cap, t_cap], dim=1),
                torch.cat([nose_cap, t_cap], dim=1),
            ],
            dim=0,
        )
        # On the caps ``side`` continues its meaning as the half of the contour the
        # point sits on (the apex itself, angle = 0, is genuinely on neither); it is
        # read only for the body profiles, where it selects ``c + R`` vs ``c - R``.
        half = torch.sign(angle)
        u_all = torch.cat([u_body, u_body, zeros, ones], dim=0)
        side = torch.cat([torch.ones_like(u_body), -torch.ones_like(u_body), half, half], dim=0)
        on_cap = torch.cat(
            [torch.zeros_like(u_body), torch.zeros_like(u_body), ones, ones], dim=0
        )
        return FrontSamples(points, u_all, side, on_cap)
