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

# Floor on the measured anchors the per-dataset rescalings divide by, so a
# dataset whose front never moved (rate0 = 0) cannot produce a division by zero.
ANCHOR_FLOOR = 1e-3


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

    ``kappa_par`` is the IN-PLANE curvature, positive where the vapour region is
    convex (the sign the diffuse
    :func:`~naviernet.physics.residuals.curvature` uses, so a bubble's Laplace
    jump is positive). It is the in-plane half of the total curvature; the
    gap-direction half is a property of the channel, not of this curve.

    ``normal`` is the outward unit normal ``(N, 2)`` -- out of the vapour --
    and ``normal_speed`` is how fast the front advances along it. The speed is
    the front's OWN motion, taken from the parameterization; equating it to
    ``u.n`` is the kinematic condition, and it is also the local capillary
    number the Bretherton film correction reads.
    """

    points: torch.Tensor
    u: torch.Tensor
    side: torch.Tensor
    on_cap: torch.Tensor
    kappa_par: torch.Tensor
    normal: torch.Tensor
    normal_speed: torch.Tensor


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


def _half_height(priors: GeometryPriors) -> float:
    """Half the channel height for this dataset -- the bound the width envelope
    saturates at."""
    return 0.5 * (priors.y_max - priors.y_min)


def _with_context(features: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
    """Append the dataset's conditioning row to a geometry net's inputs.

    The row is a per-dataset CONSTANT, so it is reduced to its first row and
    re-broadcast rather than assumed to match the batch. Callers legitimately
    arrive with either shape -- the trainer's bound view pre-expands ``c`` to its
    point batch, while the geometry evaluates its own internal batches (the nose
    grid, the front samples) whose length has nothing to do with that one.

    ``None`` passes through unchanged, which is what makes the unconditioned
    construction byte-for-byte itself.
    """
    if c is None:
        return features
    return torch.cat([features, c[:1].expand(features.shape[0], -1)], dim=-1)


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

    def __init__(
        self,
        priors: GeometryPriors,
        allow_pinch: bool = False,
        n_cond: int = 0,
    ):
        super().__init__()
        # The REFERENCE dataset's anchors. A single-dataset run has only these; a
        # joint run passes each dataset's own per call (see `priors=` below) and
        # uses these solely to set the nets' initial scales.
        self.priors = priors
        self.n_cond = int(n_cond)
        # Topology and monotonicity become learnable rather than guaranteed. Off
        # by default, so the construction is byte-for-byte what it was: the
        # radius is strictly positive and the nose strictly non-retreating.
        self.allow_pinch = bool(allow_pinch)
        self._y_half = _half_height(priors)
        # The reference dataset's measured start and speed. Every per-dataset
        # quantity below is expressed RELATIVE to these, so one shared set of
        # weights lands on each dataset's own measured anchors -- and so a
        # single-dataset run, where every ratio is exactly 1, is unchanged.
        self._ref_gap = max(priors.s0 - priors.x_root, ANCHOR_FLOOR)
        self._ref_rate = max(priors.rate0, ANCHOR_FLOOR)

        # Data-anchored start: rate = measured front speed, width = measured
        # first-frame half-width (at the envelope's midpoint), centerline flat
        # at the measured root height.
        # Under `allow_pinch` the rate is used raw, so the measured front speed IS
        # the bias; with the softplus it has to be pre-inverted. Getting this wrong
        # starts the nose retreating instead of advancing at the measured rate.
        self.rate_net = _mlp(
            1 + self.n_cond,
            out_bias=priors.rate0 if allow_pinch else _inverse_softplus(priors.rate0),
        )
        # The signed radius spans (-y_half, y_half), so the same measured w0 sits
        # at a different point of the sigmoid; without this the construction would
        # start with a NEGATIVE radius and no bubble at all.
        # The width starts at the REFERENCE dataset's measured fraction of the
        # channel for every dataset; unlike the nose and the root there is no
        # dataset-relative rescaling that keeps a shared sigmoid bias honest, so
        # a joint run's other conditions start here and the conditioning vector
        # -- which carries the regime -- has to move them.
        fraction = min(priors.w0 / self._y_half, 1.0)
        self.width_net = _mlp(
            2 + self.n_cond,
            out_bias=_logit(0.5 * (1.0 + fraction) if allow_pinch else fraction),
        )
        self.center_net = _mlp(2 + self.n_cond, out_bias=0.0)
        # s(t_min) = x_root + softplus(_s0_raw): initialized so the nose starts
        # at the measured first-training-frame front.
        self._s0_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(self._ref_gap))))

    def _anchors(self, priors: GeometryPriors | None) -> GeometryPriors:
        """This call's dataset anchors: the bound ones for a joint run, the
        model's own otherwise."""
        return self.priors if priors is None else priors

    def time_grid(self, priors: GeometryPriors | None = None, device=None) -> torch.Tensor:
        """The fixed grid the nose rate is integrated on, for THIS dataset's time
        window. Built per call rather than buffered: a joint run spans datasets
        whose windows differ, and the grid is a deterministic linspace, so
        rebuilding it costs nothing and cannot go stale."""
        anchors = self._anchors(priors)
        span = anchors.t_max - anchors.t_min
        return torch.linspace(
            anchors.t_min,
            anchors.t_max + NOSE_GRID_SLACK * span,
            NOSE_GRID_NODES,
            device=device if device is not None else self._s0_raw.device,
        )

    def nose(
        self,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        priors: GeometryPriors | None = None,
    ) -> torch.Tensor:
        """Nose position s(t), shape-preserving for ``t`` of shape (N, 1).

        Exactly monotone: rates are non-negative and the cumulative trapezoid
        only accumulates; queries beyond the grid extend at the final rate.

        The rate and the starting gap are scaled to THIS dataset's measured
        values, so one shared set of weights starts every condition at its own
        front. For a single-dataset run both ratios are exactly 1.
        """
        anchors = self._anchors(priors)
        grid = self.time_grid(anchors, t.device)
        raw = self.rate_net(_with_context(grid.unsqueeze(1), c)).squeeze(1)
        # softplus is what makes the nose exactly monotone. Under `allow_pinch`
        # the raw rate is used instead: once a bubble can detach, the front that
        # remains is no longer the daughter's advancing nose, so requiring it
        # never to retreat asserts something that is not true of it.
        rates = raw if self.allow_pinch else torch.nn.functional.softplus(raw)
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
        rate_scale = max(anchors.rate0, ANCHOR_FLOOR) / self._ref_rate
        gap = max(anchors.s0 - anchors.x_root, ANCHOR_FLOOR) / self._ref_gap
        s0 = anchors.x_root + torch.nn.functional.softplus(self._s0_raw) * gap
        return (s0 + s * rate_scale).reshape(t.shape)

    def centerline(
        self,
        xi: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        priors: GeometryPriors | None = None,
    ) -> torch.Tensor:
        """Interface centreline, anchored at THIS dataset's measured root height
        and swinging no further than the nearer channel wall."""
        anchors = self._anchors(priors)
        raw = self.center_net(_with_context(torch.cat([xi, t], dim=1), c))
        amplitude = min(anchors.y_root - anchors.y_min, anchors.y_max - anchors.y_root)
        return anchors.y_root + amplitude * torch.tanh(raw)

    def root_point(
        self,
        t: float,
        c: torch.Tensor | None = None,
        priors: GeometryPriors | None = None,
    ) -> torch.Tensor:
        """The (x, y, t) point the interface passes through at the root -- the
        exact pin, for tests and diagnostics."""
        anchors = self._anchors(priors)
        device = self._s0_raw.device
        tt = torch.tensor([[float(t)]], device=device)
        with torch.no_grad():
            y = self.centerline(torch.zeros_like(tt), tt, c, anchors)
        return torch.tensor([anchors.x_root, float(y), float(t)], device=device)

    def nose_point(
        self,
        t: float,
        c: torch.Tensor | None = None,
        priors: GeometryPriors | None = None,
    ) -> torch.Tensor:
        """The (x, y, t) point the interface passes through at the nose -- the
        capsule's far closure, ``root_point``'s mirror."""
        anchors = self._anchors(priors)
        device = self._s0_raw.device
        tt = torch.tensor([[float(t)]], device=device)
        with torch.no_grad():
            s = self.nose(tt, c, anchors)
            y = self.centerline(torch.ones_like(tt), tt, c, anchors)
        return torch.tensor([float(s), float(y), float(t)], device=device)

    def _radius(
        self,
        u: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        priors: GeometryPriors | None = None,
    ) -> torch.Tensor:
        """Half-width (inflation radius) profile along the spine, channel-bounded.

        Under ``allow_pinch`` the radius is SIGNED, spanning
        ``(-y_half, y_half)`` instead of ``(0, y_half)``. A negative radius means
        the bubble is simply not present at that station -- which is what lets
        the vapour region separate into two. With a strictly positive radius,
        ``phi`` on the spine is ``sqrt(R^2 + e^2) - e >= 0`` at every station, so
        alpha never drops below 0.5 there and detachment is unreachable however
        hard the physics pushes for it.
        """
        raw = torch.sigmoid(self.width_net(_with_context(torch.cat([u, t], dim=1), c)))
        half = _half_height(self._anchors(priors))
        return half * (2.0 * raw - 1.0 if self.allow_pinch else raw)

    def half_width(
        self,
        u: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        priors: GeometryPriors | None = None,
    ) -> torch.Tensor:
        """The bubble's half-width at spine parameter ``u`` and time ``t`` -- the
        radius the interface actually sits at, degenerate-case rescale included.

        This is the shape as a diagnostic reads it, and the quantity the measured
        masks are compared against station by station."""
        return self._radius(u, t, c, priors) * self.frame(t, c, priors).scale

    def frame(
        self,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        priors: GeometryPriors | None = None,
    ) -> CapsuleFrame:
        """The capsule's scalars at times ``t`` of shape ``(N, 1)``.

        Cap centers sit one radius inside each apex. When the bubble is shorter
        than the two cap radii (a just-nucleated bubble -- realistic,
        review-reproduced), the raw centers would CROSS the opposite apex and
        break exact nose closure; rescaling both radii jointly keeps the caps
        reaching exactly ``x_root`` and ``s`` while preserving a minimum spine
        segment of ``ABS_SMOOTH``, which also bounds ``du/dx`` (and with it
        ``alpha_x`` in the VOF residual) in the degenerate regime.
        """
        anchors = self._anchors(priors)
        s = self.nose(t, c, anchors)
        r_root = self._radius(torch.zeros_like(t), t, c, anchors)
        r_nose = self._radius(torch.ones_like(t), t, c, anchors)
        # The length the caps have to share is the one that ACTUALLY exists, not a
        # clamped stand-in: clamping it up let the caps consume more spine than the
        # bubble had, and a just-nucleated bubble then closed to bx == ax -- a
        # zero-length spine that the front's own curvature divides by. Measured:
        # kappa went non-finite. `forward` never saw it because its u is clamped
        # to [0, 1] straight afterwards, which turns the same infinity into a
        # plausible number.
        available = s - anchors.x_root
        # Only a cap that EXISTS consumes spine length, and under `allow_pinch` a
        # radius may be negative or vanish -- so the positive parts set the
        # rescale, floored so a bubble with no caps at all cannot divide by zero.
        caps = (r_root.clamp(min=0.0) + r_nose.clamp(min=0.0)).clamp(min=ABS_SMOOTH)
        scale = ((available - ABS_SMOOTH) / caps).clamp(min=0.0, max=1.0)
        r_root = r_root * scale
        r_nose = r_nose * scale
        ax = anchors.x_root + r_root
        # The documented invariant, now enforced rather than implied: at least
        # ABS_SMOOTH of spine survives. It binds only when the bubble is shorter
        # than that -- there is no bubble left to be exact about -- and everywhere
        # else `available - caps >= ABS_SMOOTH` already holds by the rescale above,
        # so both apexes stay exact.
        bx = torch.maximum(s - r_nose, ax + ABS_SMOOTH)
        return CapsuleFrame(s, ax, bx, r_root, r_nose, scale)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        priors: GeometryPriors | None = None,
    ) -> torch.Tensor:
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
        t = x[:, 2:3]
        f = self.frame(t, c, priors)
        u = ((x[:, 0:1] - f.ax) / (f.bx - f.ax)).clamp(0.0, 1.0)

        # The same scale applies along the whole spine, so the cap radii the
        # centers were placed with are exactly the radii the field compares
        # against -- apex exactness survives the degenerate rescale.
        radius = self._radius(u, t, c, priors) * f.scale
        spine_x = f.ax + u * (f.bx - f.ax)
        spine_y = self.centerline(u, t, c, priors)
        d_sq = (x[:, 0:1] - spine_x) ** 2 + (x[:, 1:2] - spine_y) ** 2
        # Matched floors: phi = 0 exactly where d = R (the interface, incl. both
        # apexes), while staying C-infinity on the spine (the KAPPA lesson).
        # `copysign` carries a negative radius through as a negative phi, so a
        # vanished station is empty rather than a hair of vapour; for a positive
        # radius -- always, unless `allow_pinch` -- it is the identity.
        inflated = torch.copysign(torch.sqrt(radius**2 + ABS_SMOOTH**2), radius)
        return inflated - torch.sqrt(d_sq + ABS_SMOOTH**2)

    def front(
        self,
        t: torch.Tensor,
        n_body: int,
        n_cap: int,
        c: torch.Tensor | None = None,
        priors: GeometryPriors | None = None,
    ) -> FrontSamples:
        """The interface at each time in ``t``, as an explicit sampled curve.

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

        Every sample carries its own outward normal, in-plane curvature, and
        normal speed, all taken from the parameterization rather than from a
        smeared field. Deliberately NOT detached: the loss must be able to move
        the front.
        """
        if n_body < 1 or n_cap < 2:
            raise ValueError(
                f"front() needs n_body >= 1 and n_cap >= 2 (a cap is an arc, not a "
                f"point), got n_body={n_body}, n_cap={n_cap}"
            )
        t = t.reshape(-1, 1)
        n_times = t.shape[0]
        device = t.device

        def tiled(values: torch.Tensor) -> torch.Tensor:
            """One row of parameter values, repeated for every time (t-major)."""
            return values.reshape(1, -1).expand(n_times, -1).reshape(-1, 1)

        # The body's u excludes the seams, so each u belongs to exactly one
        # segment and the contour is closed exactly once.
        u_grid = torch.linspace(0.0, 1.0, n_body + 2, device=device)[1:-1]
        angles = torch.linspace(-0.5 * math.pi, 0.5 * math.pi, n_cap, device=device)
        body_zeros = torch.zeros(n_times * n_body, 1, device=device)
        cap_zeros = torch.zeros(n_times * n_cap, 1, device=device)
        cap_ones = torch.ones(n_times * n_cap, 1, device=device)

        u = torch.cat([tiled(u_grid), tiled(u_grid), cap_zeros, cap_ones], dim=0)
        angle = torch.cat([body_zeros, body_zeros, tiled(angles), tiled(angles)], dim=0)
        on_cap = torch.cat([body_zeros, body_zeros, cap_ones, cap_ones], dim=0)
        # On the caps ``side`` continues its meaning as the half of the contour the
        # point sits on (the apex itself, angle = 0, is on neither); it selects
        # ``c + R`` vs ``c - R`` only on the body.
        side = torch.cat(
            [
                torch.ones_like(body_zeros),
                -torch.ones_like(body_zeros),
                torch.sign(tiled(angles)),
                torch.sign(tiled(angles)),
            ],
            dim=0,
        )
        times = torch.cat(
            [
                t.repeat_interleave(n_body, dim=0),
                t.repeat_interleave(n_body, dim=0),
                t.repeat_interleave(n_cap, dim=0),
                t.repeat_interleave(n_cap, dim=0),
            ],
            dim=0,
        )
        # A leaf, so the front's own motion dP/dt is available by autograd: it is
        # both the Bretherton correction's local capillary number and the left
        # side of the kinematic condition.
        times = times.detach().requires_grad_(True)

        position, normal, kappa = self._front_frame(u, side, on_cap, angle, times, c, priors)
        speed = self._normal_speed(position, normal, times)
        points = torch.cat([position, times], dim=1)
        return FrontSamples(points, u.detach(), side, on_cap, kappa, normal, speed)

    def _front_frame(
        self,
        u: torch.Tensor,
        side: torch.Tensor,
        on_cap: torch.Tensor,
        angle: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        priors: GeometryPriors | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(position, outward_normal, kappa_par)`` per sample.

        Both branches -- body profile and circular cap -- are evaluated for every
        sample and selected by ``on_cap``. Blending rather than indexing keeps one
        vectorised, twice-differentiable expression, which is what the jump
        condition needs to be able to move the shape.
        """
        f = self.frame(t, c, priors)
        length = f.bx - f.ax

        # Body: y = c(u,t) + side R(u,t), with x affine in u -- so y' = y_u/L and
        # y'' = y_uu/L^2 exactly, with no second term.
        u_leaf = u.detach().requires_grad_(True)
        centre = self.centerline(u_leaf, t, c, priors)
        radius = self._radius(u_leaf, t, c, priors) * f.scale

        def d_du(field: torch.Tensor) -> torch.Tensor:
            return torch.autograd.grad(
                field, u_leaf, torch.ones_like(field), create_graph=True
            )[0]

        centre_u, radius_u = d_du(centre), d_du(radius)
        y_u = centre_u + side * radius_u
        y_uu = d_du(centre_u) + side * d_du(radius_u)
        y_x = y_u / length
        slope = torch.sqrt(1.0 + y_x**2)

        body_xy = torch.cat([f.ax + u * length, centre + side * radius], dim=1)
        # Outward normal of y = c + side R: (-y', 1)/|.| points up, which is out
        # of the vapour on the upper profile and into it on the lower -- hence the
        # side factor.
        body_normal = side * torch.cat([-y_x, torch.ones_like(y_x)], dim=1) / slope
        # kappa = -side y''/(1 + y'^2)^{3/2}: the sign makes a bubble that bulges
        # outward read POSITIVE, matching the diffuse convention.
        body_kappa = -side * y_uu / length**2 / slope**3

        # Caps: circular arcs about centres one radius inside each apex. A
        # circular arc is constant-curvature -- which is why the caps are circles
        # at all -- so each reads exactly its own 1/r, and its outward normal is
        # simply the radial direction.
        at_root = u < 0.5
        cap_r = torch.where(at_root, f.r_root, f.r_nose)
        cap_x = torch.where(at_root, f.ax, f.bx)
        cap_sign = torch.where(at_root, -torch.ones_like(u), torch.ones_like(u))
        cos, sin = torch.cos(angle), torch.sin(angle)
        cap_xy = torch.cat(
            [cap_x + cap_sign * cap_r * cos, self.centerline(u, t, c, priors) + cap_r * sin],
            dim=1,
        )
        cap_normal = torch.cat([cap_sign * cos, sin], dim=1)
        # A cap that has vanished has no curvature to report; floor the radius so
        # the reciprocal stays finite instead of diverging at the pinch.
        cap_kappa = 1.0 / cap_r.clamp(min=ABS_SMOOTH)

        return (
            torch.where(on_cap > 0, cap_xy, body_xy),
            torch.where(on_cap > 0, cap_normal, body_normal),
            torch.where(on_cap > 0, cap_kappa, body_kappa),
        )

    def _normal_speed(
        self, position: torch.Tensor, normal: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """How fast the front advances along its own outward normal, ``(N, 1)``.

        ``dP/dt`` is taken at fixed parameter, which slides along the curve as
        well as across it; projecting onto the normal discards that tangential
        part, so the result is the parameterization-independent front speed --
        the same quantity the kinematic condition equates to ``u.n``, and the one
        the Bretherton film correction is a function of.
        """
        velocity = [
            torch.autograd.grad(position[:, i].sum(), t, create_graph=True)[0] for i in (0, 1)
        ]
        return velocity[0] * normal[:, 0:1] + velocity[1] * normal[:, 1:2]
