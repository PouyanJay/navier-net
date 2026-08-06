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
- the width is bounded by the channel, and phi is smooth on the spine (the
  matched-floor form; Stage-B curvature differentiates phi twice, the same
  lesson the hard-pin gate learned). At the cap-body seams the u-clamp leaves
  phi C0-but-not-C1: a measured, accepted trade-off (kappa*grad(alpha) stays
  O(10-50) there, far under harmful scale; a C1 blend would require tying the
  boundary radius slope to zero and cost expressivity) -- regression-tested at
  the seam points.

The caps are circular by default -- constant curvature, the STATIC Young-Laplace
cap shape. ``model.cap_freedom`` drops that assumption (see
:meth:`GeometricInterface._cap_modulation`): an advancing bubble in a Hele-Shaw
gap has a flattened nose, and asserting a circle there denies the shape to the
data (a circle cannot comply) and the curvature to the physics (it is handed
1/r rather than reading it off the curve). Everything above still holds with the
freedom on; the gate is built to vanish exactly where those guarantees live.
"""

from __future__ import annotations

import math
from collections.abc import Callable
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
    ``u.n`` is the kinematic condition, it is the local capillary number the
    Bretherton film correction reads, and it is what the measured front velocity
    supervises.

    ``angle`` is the cap sweep angle in ``[-pi/2, pi/2]``, and 0 on the body. It
    is what locates a sample ALONG a cap: ``u`` is pinned at 0 or 1 there, so
    without it the two caps are each a single point as far as a consumer can
    tell -- which is not enough to bin a profile over them.
    """

    points: torch.Tensor
    u: torch.Tensor
    side: torch.Tensor
    on_cap: torch.Tensor
    angle: torch.Tensor
    kappa_par: torch.Tensor
    normal: torch.Tensor
    normal_speed: torch.Tensor


class GeometryContext(NamedTuple):
    """Which dataset a geometry call is about: its conditioning row and its
    measured anchors.

    The two always travel together -- a joint run binds both per dataset, a
    single-dataset run binds neither -- so they are one object rather than two
    optional parameters threaded through the whole construction. ``None``
    anywhere means "the model's own", which is what a single-dataset run has.
    """

    c: torch.Tensor | None = None
    priors: GeometryPriors | None = None


class FrontQuery(NamedTuple):
    """The per-sample parameters that locate one point on the contour.

    Built together in :meth:`GeometricInterface.front` and consumed together in
    :meth:`GeometricInterface._front_frame`; carrying them as one object is what
    keeps that evaluation to three arguments.
    """

    u: torch.Tensor  # spine parameter, 0 at the root apex and 1 at the nose
    side: torch.Tensor  # +1 upper profile, -1 lower
    on_cap: torch.Tensor  # 1 on an end cap, where the profile form does not apply
    angle: torch.Tensor  # sweep angle on a cap; 0 off them
    t: torch.Tensor  # the sample's own time, a leaf so dP/dt is available


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


def _d_wrt(leaf: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Derivative operator with respect to ``leaf``, graph kept.

    ``create_graph`` is what lets a SECOND derivative be taken, and what lets the
    loss reach the nets through the curvature -- the shape has to be movable by
    the term that reads it.
    """

    def derivative(field: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(field, leaf, torch.ones_like(field), create_graph=True)[0]

    return derivative


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
        evolving_width: bool = False,
        cap_freedom: bool = False,
        cap_delta: float = 0.2,
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
        self.evolving_width = bool(evolving_width)
        # The end caps stop being circles. Validated here rather than at the call
        # site because this is the object that owns the bound: at delta >= 1 the
        # modulation can drive the cap radius to zero or through it, and the cap
        # folds through itself -- which is not a shape the physics can be scored
        # on. Checked whatever `cap_freedom` says, so a config that would break
        # the moment the flag is flipped fails now instead of then.
        self.cap_freedom = bool(cap_freedom)
        self.cap_delta = float(cap_delta)
        if not 0.0 <= self.cap_delta < 1.0:
            raise ValueError(
                f"model.cap_delta must be in [0, 1) -- it is the cap's departure from "
                f"its circle as a FRACTION of the local radius, and at 1 the radius "
                f"reaches zero and the cap self-intersects. Got {self.cap_delta}."
            )
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
        start = _logit(0.5 * (1.0 + fraction) if allow_pinch else fraction)
        if self.evolving_width:
            # Two nets of u alone: where the profile STARTS, and which way it
            # travels. Time enters only through the scalar elongation below, so
            # the profile cannot decay back to its bias off-data -- it can only
            # keep going the way it was going. The rate starts at zero, so the
            # construction still opens as the measured first-frame capsule.
            self.width_base = _mlp(1 + self.n_cond, out_bias=start)
            self.width_rate = _mlp(1 + self.n_cond, out_bias=0.0)
        else:
            self.width_net = _mlp(
                2 + self.n_cond,
                out_bias=start,
            )
        self.center_net = _mlp(2 + self.n_cond, out_bias=0.0)
        # The cap's departure from its circle, over (cos psi, sin psi, u, t): the
        # angle enters through its sine and cosine rather than through psi itself,
        # so there is no atan2 branch in the field and no wrap to reason about.
        # `u` (0 at the root cap, 1 at the nose) is what lets one net give the two
        # caps different shapes. Bias 0 -> tanh(0) = 0 -> the construction OPENS as
        # the circle it replaces and learns its way off it.
        if self.cap_freedom:
            self.cap_net = _mlp(4 + self.n_cond, out_bias=0.0)
        # s(t_min) = x_root + softplus(_s0_raw): initialized so the nose starts
        # at the measured first-training-frame front.
        self._s0_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(self._ref_gap))))

    def _anchors(self, ctx: GeometryContext) -> GeometryPriors:
        """This call's dataset anchors: the bound ones for a joint run, the
        model's own otherwise."""
        return self.priors if ctx.priors is None else ctx.priors

    def time_grid(self, ctx: GeometryContext | None = None, device=None) -> torch.Tensor:
        """The fixed grid the nose rate is integrated on, for THIS dataset's time
        window. Built per call rather than buffered: a joint run spans datasets
        whose windows differ, and the grid is a deterministic linspace, so
        rebuilding it costs nothing and cannot go stale."""
        anchors = self._anchors(ctx or GeometryContext())
        span = anchors.t_max - anchors.t_min
        return torch.linspace(
            anchors.t_min,
            anchors.t_max + NOSE_GRID_SLACK * span,
            NOSE_GRID_NODES,
            device=device if device is not None else self._s0_raw.device,
        )

    def nose(self, t: torch.Tensor, ctx: GeometryContext | None = None) -> torch.Tensor:
        """Nose position s(t), shape-preserving for ``t`` of shape (N, 1).

        Exactly monotone: rates are non-negative and the cumulative trapezoid
        only accumulates; queries beyond the grid extend at the final rate.

        The rate and the starting gap are scaled to THIS dataset's measured
        values, so one shared set of weights starts every condition at its own
        front. For a single-dataset run both ratios are exactly 1.
        """
        ctx = ctx or GeometryContext()
        anchors = self._anchors(ctx)
        grid = self.time_grid(ctx, t.device)
        raw = self.rate_net(_with_context(grid.unsqueeze(1), ctx.c)).squeeze(1)
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
        self, xi: torch.Tensor, t: torch.Tensor, ctx: GeometryContext | None = None
    ) -> torch.Tensor:
        """Interface centreline, anchored at THIS dataset's measured root height
        and swinging no further than the nearer channel wall."""
        ctx = ctx or GeometryContext()
        anchors = self._anchors(ctx)
        raw = self.center_net(_with_context(torch.cat([xi, t], dim=1), ctx.c))
        amplitude = min(anchors.y_root - anchors.y_min, anchors.y_max - anchors.y_root)
        return anchors.y_root + amplitude * torch.tanh(raw)

    def root_point(self, t: float, ctx: GeometryContext | None = None) -> torch.Tensor:
        """The (x, y, t) point the interface passes through at the root -- the
        exact pin, for tests and diagnostics."""
        ctx = ctx or GeometryContext()
        anchors = self._anchors(ctx)
        device = self._s0_raw.device
        tt = torch.tensor([[float(t)]], device=device)
        with torch.no_grad():
            y = self.centerline(torch.zeros_like(tt), tt, ctx)
        return torch.tensor([anchors.x_root, float(y), float(t)], device=device)

    def apex(self, t: torch.Tensor, ctx: GeometryContext | None = None) -> torch.Tensor:
        """The nose apex ``(x, y)`` at times ``t`` of shape ``(N, 1)``, as
        ``(N, 2)``. Batched and differentiable -- ``nose_point``'s trainable
        counterpart.

        This is the one point on the interface with an honest frame-to-frame
        CORRESPONDENCE. Everywhere else a curve sliding along itself looks
        identical between frames, so only the normal component of the motion is
        observable; the apex is geometrically distinguishable, so its full 2-D
        displacement is measurable and can be supervised directly.
        """
        ctx = ctx or GeometryContext()
        return torch.cat(
            [self.nose(t, ctx), self.centerline(torch.ones_like(t), t, ctx)], dim=1
        )

    def nose_point(self, t: float, ctx: GeometryContext | None = None) -> torch.Tensor:
        """The (x, y, t) point the interface passes through at the nose -- the
        capsule's far closure, ``root_point``'s mirror."""
        device = self._s0_raw.device
        tt = torch.tensor([[float(t)]], device=device)
        with torch.no_grad():
            apex = self.apex(tt, ctx).reshape(-1)
        return torch.tensor([float(apex[0]), float(apex[1]), float(t)], device=device)

    def _radius(
        self, u: torch.Tensor, t: torch.Tensor, ctx: GeometryContext | None = None
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
        ctx = ctx or GeometryContext()
        raw = torch.sigmoid(self._width_logit(u, t, ctx))
        half = _half_height(self._anchors(ctx))
        return half * (2.0 * raw - 1.0 if self.allow_pinch else raw)

    def _cap_modulation(
        self,
        nx: torch.Tensor,
        ny: torch.Tensor,
        u: torch.Tensor,
        t: torch.Tensor,
        ctx: GeometryContext,
    ) -> torch.Tensor:
        """The cap's departure from its circle, as a factor on the local radius.

        ``(nx, ny)`` is the unit radial direction from the spine point out to the
        query point -- the circle's own outward normal. The gate is
        ``4 nx^2 ny^2``, which is ``sin^2(2 psi)`` in the cap angle, and every
        property this construction needs is a property of that gate:

        - **Zero at the apex** (``ny = 0``). The tip does not move, so the root
          stays pinned exactly on the measured anchor and the nose stays exactly
          the monotone ``s(t)``. Those two are what the R3 win was built on, and
          a gate that was non-zero here would quietly spend them.
        - **Zero at the seam, and across the whole body** (``nx = 0``: on the body
          the spine point shares the query point's x). So the cap still meets the
          profile, and -- the invariant everything rests on -- ``forward`` and
          ``front`` keep describing ONE shape. If they diverged, the data term and
          the sharp-interface residuals would be pulling on two different bubbles.
        - **Quadratic about the apex** (``b''(0) = 2``), which is the whole point:
          for a polar curve with ``r'(0) = 0``, ``kappa = (r - r'')/r^2``, so a
          non-zero ``r''`` at the tip moves the tip curvature off ``1/r``. A
          flattened nose -- what an advancing bubble in a Hele-Shaw gap actually
          has -- becomes reachable for the first time.

        Even powers throughout, so it is smooth across ``nx = 0`` with no
        ``atan2`` and no ``abs``: a kink at the seam would have fed straight into
        the jump condition as a curvature spike (the KAPPA lesson).

        **How exact the two routes agree.** ``forward`` builds ``(nx, ny)`` by
        normalising with the ABS_SMOOTH-floored distance; ``_cap_frame`` has the
        angle itself and uses exact ``cos``/``sin``. The two therefore differ by
        the factor ``1/sqrt(1 + ABS_SMOOTH^2/d^2)``, and the modulation's VALUE
        between apex and seam differs with it -- measured, a front sample sits
        within 2.5e-5 of ``phi = 0`` rather than exactly on it, five orders under
        ``alpha_eps``. The floor is deliberate: without it the direction's
        gradient diverges like ``1/|d|`` at the cap centre, straight into the VOF
        residual, which is the failure the floor exists to prevent.

        What the floor does NOT weaken is the part the guarantees rest on. The
        gate's ZEROS are structural, not numerical: at the apex ``dy = 0``
        exactly and on the body/seam ``dx = 0`` exactly (there the spine point
        shares the query point's x), so the gate is exactly zero at all three
        whatever the floor is. The root pin, the monotone nose and the untouched
        body are exact; only the modulation's magnitude in between is approximate.

        ``tanh`` bounds the departure to ``cap_delta`` of the local radius, which
        is what stops a cap folding through itself; the net's zero bias means the
        construction OPENS as the circle it replaces and learns its way off it.
        """
        if not self.cap_freedom:
            return torch.ones_like(nx)
        gate = 4.0 * nx**2 * ny**2
        features = _with_context(torch.cat([nx, ny, u, t], dim=1), ctx.c)
        return 1.0 + self.cap_delta * gate * torch.tanh(self.cap_net(features))

    def _width_logit(
        self, u: torch.Tensor, t: torch.Tensor, ctx: GeometryContext
    ) -> torch.Tensor:
        """The pre-sigmoid width profile.

        With ``evolving_width`` it is ``base(u) + rate(u) * elongation(t)``: one
        learned direction per station, travelled at the bubble's own elongation.
        The saturating sigmoid then turns "keep going" into a bounded limit --
        and where the rate is negative that limit is ZERO WIDTH, which is how a
        neck is able to pinch rather than merely deepen.
        """
        if not self.evolving_width:
            return self.width_net(_with_context(torch.cat([u, t], dim=1), ctx.c))
        features = _with_context(u, ctx.c)
        return self.width_base(features) + self.width_rate(features) * self._elongation(t, ctx)

    def _elongation(self, t: torch.Tensor, ctx: GeometryContext) -> torch.Tensor:
        """How much longer the bubble is than when it was first measured, as a
        fraction of that first length. Zero at ``t_min`` and monotone after,
        because the nose is -- so it carries the nose's own extrapolation
        guarantee into the shape instead of leaving time to a saturating net.
        """
        anchors = self._anchors(ctx)
        gap = max(anchors.s0 - anchors.x_root, ANCHOR_FLOOR)
        return (self.nose(t, ctx) - anchors.x_root) / gap - 1.0

    def half_width(
        self, u: torch.Tensor, t: torch.Tensor, ctx: GeometryContext | None = None
    ) -> torch.Tensor:
        """The bubble's half-width at spine parameter ``u`` and time ``t`` -- the
        radius the interface actually sits at, degenerate-case rescale included.

        This is the shape as a diagnostic reads it, and the quantity the measured
        masks are compared against station by station."""
        return self._radius(u, t, ctx) * self.frame(t, ctx).scale

    def frame(self, t: torch.Tensor, ctx: GeometryContext | None = None) -> CapsuleFrame:
        """The capsule's scalars at times ``t`` of shape ``(N, 1)``.

        Cap centers sit one radius inside each apex. When the bubble is shorter
        than the two cap radii (a just-nucleated bubble -- realistic,
        review-reproduced), the raw centers would CROSS the opposite apex and
        break exact nose closure; rescaling both radii jointly keeps the caps
        reaching exactly ``x_root`` and ``s`` while preserving a minimum spine
        segment of ``ABS_SMOOTH``, which also bounds ``du/dx`` (and with it
        ``alpha_x`` in the VOF residual) in the degenerate regime.
        """
        ctx = ctx or GeometryContext()
        anchors = self._anchors(ctx)
        s = self.nose(t, ctx)
        r_root = self._radius(torch.zeros_like(t), t, ctx)
        r_nose = self._radius(torch.ones_like(t), t, ctx)
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
        ctx = GeometryContext(c, priors)
        t = x[:, 2:3]
        f = self.frame(t, ctx)
        u = ((x[:, 0:1] - f.ax) / (f.bx - f.ax)).clamp(0.0, 1.0)

        # The same scale applies along the whole spine, so the cap radii the
        # centers were placed with are exactly the radii the field compares
        # against -- apex exactness survives the degenerate rescale.
        radius = self._radius(u, t, ctx) * f.scale
        spine_x = f.ax + u * (f.bx - f.ax)
        spine_y = self.centerline(u, t, ctx)
        dx, dy = x[:, 0:1] - spine_x, x[:, 1:2] - spine_y
        d_sq = dx**2 + dy**2
        # Free caps: the radius the field compares against varies with the
        # direction the query point lies in. The gate is zero on the body (where
        # `dx` is zero because the spine point shares the point's x), so this is
        # the same field as before everywhere except inside the two caps.
        if self.cap_freedom:
            d = torch.sqrt(d_sq + ABS_SMOOTH**2)
            radius = radius * self._cap_modulation(dx / d, dy / d, u, t, ctx)
        # Matched floors: phi = 0 exactly where d = R (the interface, incl. both
        # apexes), while staying C-infinity on the spine (the KAPPA lesson).
        # `copysign` carries a negative radius through as a negative phi, so a
        # vanished station is empty rather than a hair of vapour; for a positive
        # radius -- always, unless `allow_pinch` -- it is the identity.
        inflated = torch.copysign(torch.sqrt(radius**2 + ABS_SMOOTH**2), radius)
        return inflated - torch.sqrt(d_sq + ABS_SMOOTH**2)

    def front(
        self, t: torch.Tensor, n_body: int, n_cap: int, ctx: GeometryContext | None = None
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
        query = self._sample_grid(t, n_body, n_cap)
        position, normal, kappa = self._front_frame(query, ctx or GeometryContext())
        speed = self._normal_speed(position, normal, query.t)
        points = torch.cat([position, query.t], dim=1)
        return FrontSamples(
            points,
            query.u.detach(),
            query.side,
            query.on_cap,
            query.angle,
            kappa,
            normal,
            speed,
        )

    @staticmethod
    def _sample_grid(t: torch.Tensor, n_body: int, n_cap: int) -> FrontQuery:
        """Where on the contour to sample: the four segments' parameters, laid out
        t-major and concatenated in the order the profiles and caps are assembled.

        Pure bookkeeping -- no geometry is evaluated here.
        """
        if n_body < 1 or n_cap < 2:
            raise ValueError(
                f"front() needs n_body >= 1 and n_cap >= 2 (a cap is an arc, not a "
                f"point), got n_body={n_body}, n_cap={n_cap}"
            )
        t = t.reshape(-1, 1)
        n_times, device = t.shape[0], t.device

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
        body_t = t.repeat_interleave(n_body, dim=0)
        cap_t = t.repeat_interleave(n_cap, dim=0)

        # On the caps ``side`` continues its meaning as the half of the contour the
        # point sits on (the apex itself, angle = 0, is on neither); it selects
        # ``c + R`` vs ``c - R`` only on the body.
        halves = torch.sign(tiled(angles))
        return FrontQuery(
            u=torch.cat([tiled(u_grid), tiled(u_grid), cap_zeros, cap_ones], dim=0),
            side=torch.cat(
                [torch.ones_like(body_zeros), -torch.ones_like(body_zeros), halves, halves],
                dim=0,
            ),
            on_cap=torch.cat([body_zeros, body_zeros, cap_ones, cap_ones], dim=0),
            angle=torch.cat([body_zeros, body_zeros, tiled(angles), tiled(angles)], dim=0),
            # A leaf, so the front's own motion dP/dt is available by autograd: it
            # is both the Bretherton correction's local capillary number and the
            # left side of the kinematic condition.
            t=torch.cat([body_t, body_t, cap_t, cap_t], dim=0).detach().requires_grad_(True),
        )

    def _front_frame(
        self, query: FrontQuery, ctx: GeometryContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(position, outward_normal, kappa_par)`` per sample.

        Both branches -- body profile and circular cap -- are evaluated for every
        sample and selected by ``on_cap``. Blending rather than indexing keeps one
        vectorised, twice-differentiable expression, which is what the jump
        condition needs to be able to move the shape.
        """
        frame = self.frame(query.t, ctx)
        body = self._body_frame(query, frame, ctx)
        cap = self._cap_frame(query, frame, ctx)
        on_cap = query.on_cap > 0
        return tuple(torch.where(on_cap, c, b) for c, b in zip(cap, body, strict=True))

    def _body_frame(
        self, query: FrontQuery, frame: CapsuleFrame, ctx: GeometryContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The profile branch: ``y = c(u,t) + side R(u,t)``.

        ``x`` is affine in ``u`` along the body, so ``y' = y_u/L`` and
        ``y'' = y_uu/L^2`` exactly, with no second term.
        """
        side, length = query.side, frame.bx - frame.ax
        u_leaf = query.u.detach().requires_grad_(True)
        centre = self.centerline(u_leaf, query.t, ctx)
        radius = self._radius(u_leaf, query.t, ctx) * frame.scale

        def d_du(field: torch.Tensor) -> torch.Tensor:
            return torch.autograd.grad(
                field, u_leaf, torch.ones_like(field), create_graph=True
            )[0]

        centre_u, radius_u = d_du(centre), d_du(radius)
        y_x = (centre_u + side * radius_u) / length
        y_xx = (d_du(centre_u) + side * d_du(radius_u)) / length**2
        slope = torch.sqrt(1.0 + y_x**2)

        position = torch.cat([frame.ax + query.u * length, centre + side * radius], dim=1)
        # Outward normal of y = c + side R: (-y', 1)/|.| points up, which is out
        # of the vapour on the upper profile and into it on the lower -- hence the
        # side factor. And kappa = -side y''/(1 + y'^2)^{3/2}: the sign makes a
        # bubble that bulges outward read POSITIVE, matching the diffuse convention.
        normal = side * torch.cat([-y_x, torch.ones_like(y_x)], dim=1) / slope
        return position, normal, -side * y_xx / slope**3

    def _cap_frame(
        self, query: FrontQuery, frame: CapsuleFrame, ctx: GeometryContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The cap branch: circular arcs about centres one radius inside each apex.

        A circular arc is constant-curvature -- which is why the caps are circles
        at all -- so each reads exactly its own ``1/r``, and its outward normal is
        simply the radial direction.
        """
        at_root = query.u < 0.5
        base = torch.where(at_root, frame.r_root, frame.r_nose)
        centre_x = torch.where(at_root, frame.ax, frame.bx)
        sign = torch.where(at_root, -torch.ones_like(query.u), torch.ones_like(query.u))

        # The cap angle becomes a leaf, so the curve's own derivatives in it are
        # available by autograd (the same device `_body_frame` uses for u). Without
        # free caps nothing depends on it and the circle's closed forms are exact
        # and cheaper, so that path stays byte-for-byte what it was.
        psi = query.angle.detach().requires_grad_(True) if self.cap_freedom else query.angle
        cos, sin = torch.cos(psi), torch.sin(psi)

        # The same modulation `forward` applies, read off the same outward radial
        # direction -- here it is the cap angle itself, since these points ARE the
        # cap. `sign` carries the root cap's outward direction (-x), so one net
        # sees a consistent frame at both ends.
        radius = base * self._cap_modulation(sign * cos, sin, query.u, query.t, ctx)

        if self.cap_freedom:
            d_psi = _d_wrt(psi)
            r_psi = d_psi(radius)
            r_psi2 = d_psi(r_psi)
        else:
            r_psi = r_psi2 = torch.zeros_like(radius)

        position = torch.cat(
            [
                centre_x + sign * radius * cos,
                self.centerline(query.u, query.t, ctx) + radius * sin,
            ],
            dim=1,
        )
        # A polar curve r(psi) about the cap centre. Both reduce to the circle's
        # closed forms when r' = r'' = 0 -- the normal to the radial direction and
        # kappa to 1/r -- which is why freeing the cap changes nothing until the
        # cap net moves off its zero bias.
        norm = torch.sqrt(radius**2 + r_psi**2).clamp(min=ABS_SMOOTH)
        # The normal divides by a SIGNED length, the way `forward` inflates by a
        # signed radius. Dividing by the unsigned one would flip the normal of a
        # VANISHED cap (`allow_pinch`, radius < 0) -- a point this construction
        # says is not there, but whose reported normal the kinematic condition and
        # the Bretherton capillary number both read. With r' = 0 this is exactly
        # the radial direction for either sign, which is what the circular
        # construction returned unconditionally.
        normal = torch.cat(
            [sign * (radius * cos + r_psi * sin), radius * sin - r_psi * cos], dim=1
        ) / torch.copysign(norm, radius)
        # A cap that has VANISHED (a non-positive radius, reachable only under
        # `allow_pinch`) has no curvature to report. Flooring its radius would
        # hand the jump condition ~1/ABS_SMOOTH -- a huge positive curvature for
        # a point that, by this construction's own contract, is not there -- and
        # that number would flow straight into the Laplace residual at exactly
        # the pinch the feature exists to make trainable. Report zero instead:
        # no interface, no capillary pressure.
        # Under `cap_freedom` the test is on the MODULATED radius, so within a
        # band of width cap_delta about the threshold one cap can read "present"
        # at some angles and "gone" at others in the same frame. Accepted: the
        # band is ABS_SMOOTH-scaled (~1e-3 of the channel), and a bubble that
        # close to vanishing has no meaningful capillary pressure to report at
        # any angle.
        kappa = torch.where(
            radius > ABS_SMOOTH,
            (radius**2 + 2.0 * r_psi**2 - radius * r_psi2) / norm**3,
            torch.zeros_like(radius),
        )
        return position, normal, kappa

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
