"""The measured outline as the model's interface (``model.measured_outline``).

The trained capsule fits the images well, but its curvature jumps where its
caps meet its body. Young-Laplace reads that jump as a pressure step the bubble
does not have (about 70 Pa at the nose on Series-1-w300). From the physics stage
on, this module stands in for it with the outline traced from the images
(:mod:`naviernet.data.outline`): one fair curve per frame, with no seams.

Between frames the outline is a tensor-product cubic spline: periodic in the
curve parameter ``q`` (0 at the nose apex, 1/2 at the root apex, once round) and
cubic in time through the frames. Position, normal, curvature and front speed
all come from that one surface, so they agree with each other at every instant,
and the front speed is ``dP/dt . n`` exactly.

``phi`` is the signed distance to the curve, positive in the vapour: the point is
projected onto the curve by Newton's method, and the last step is kept in the
graph, which is all the first and second derivatives of the distance need (at
the projection, moving along the curve does not change the distance).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline

from naviernet.models.geometry import (
    ABS_SMOOTH,
    FrontSamples,
    GeometricInterface,
    GeometryContext,
)

# Newton steps toward each point's nearest point on the curve, taken without the
# graph before the one step that carries it.
PROJECTION_STEPS = 6

# The coarse lookup the projection starts from: instants per frame interval, and
# every n-th station of the half outline.
COARSE_INSTANTS_PER_FRAME = 8
COARSE_STATION_STRIDE = 4

# Newton's curvature term is floored at this fraction of the Gauss-Newton one,
# so a point beyond a centre of curvature still steps toward the curve.
NEWTON_FLOOR = 0.1


class MeasuredOutline(nn.Module):
    """The traced outline, between frames and along the curve.

    ``stations`` is ``(frames, M, 2)`` ``(x*, y*)``: each frame's closed curve at
    ``M`` equally spaced parameter values, starting at the nose apex, on the
    ``y* > axis`` side first. Its buffers are not saved with the model; they are
    rebuilt from the dataset's outline file.
    """

    def __init__(
        self,
        times: np.ndarray,
        stations: np.ndarray,
        axis: float,
        nose_cap_q: float,
        root_cap_q: float,
    ):
        super().__init__()
        times = np.asarray(times, dtype=np.float64)
        stations = np.asarray(stations, dtype=np.float64)
        n_frames, n_stations, _ = stations.shape
        if n_frames < 2 or len(times) != n_frames:
            raise ValueError(
                "the measured outline needs one curve per frame, two or more frames"
            )
        if n_stations % 2:
            raise ValueError("the measured outline's curves need an even number of stations")
        self.axis = float(axis)
        self.nose_cap_q = float(nose_cap_q)
        self.root_cap_q = float(root_cap_q)
        self.n_stations = n_stations
        knots = np.arange(n_stations + 1) / n_stations
        closed = np.concatenate([stations, stations[:, :1]], axis=1)
        along = CubicSpline(knots, closed, axis=1, bc_type="periodic").c  # (4, M, K, 2)
        through = CubicSpline(times, np.moveaxis(along, 2, 0), axis=0).c  # (4, K-1, 4, M, 2)
        # (interval in t, station interval, power of t, power of q, coordinate)
        coefficients = np.ascontiguousarray(np.transpose(through, (1, 3, 0, 2, 4)))
        self.register_buffer(
            "coefficients", torch.tensor(coefficients, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "times", torch.tensor(times, dtype=torch.float32), persistent=False
        )
        self._build_coarse_lookup()

    @classmethod
    def load(cls, path) -> MeasuredOutline:
        """The outline a dataset's ``outline.npz`` stores."""
        with np.load(path) as archive:
            return cls(
                archive["t_star"],
                archive["stations"],
                float(archive["axis_star"]),
                float(archive["nose_cap_q"]),
                float(archive["root_cap_q"]),
            )

    def _build_coarse_lookup(self) -> None:
        """Positions of every few stations of the half outline at instants a fraction
        of a frame apart: where each projection starts."""
        span = self.times[[0, -1]]
        n_instants = COARSE_INSTANTS_PER_FRAME * (len(self.times) - 1) + 1
        instants = torch.linspace(float(span[0]), float(span[1]), n_instants)
        q = torch.arange(0, self.n_stations // 2 + 1, COARSE_STATION_STRIDE) / self.n_stations
        grid_q = q.reshape(1, -1).expand(n_instants, -1).reshape(-1)
        grid_t = instants.repeat_interleave(len(q))
        points = self.surface(grid_q, grid_t)["P"].reshape(n_instants, len(q), 2)
        self.register_buffer("coarse_q", q, persistent=False)
        self.register_buffer("coarse_points", points, persistent=False)

    def surface(self, q: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """``P``, ``P_q``, ``P_qq`` and ``P_t`` at parameters ``q`` and times ``t``
        (both ``(N,)``), each ``(N, 2)``.

        Outside the frames' span the outline moves on at its end rate, holding
        its end shape.
        """
        times = self.times
        held = t.clamp(times[0], times[-1])
        interval = (torch.searchsorted(times, held.detach(), right=True) - 1).clamp(
            0, len(times) - 2
        )
        tau = held - times[interval]
        wrapped = torch.remainder(q, 1.0)
        station = (
            (wrapped.detach() * self.n_stations).floor().long().clamp(0, self.n_stations - 1)
        )
        u = wrapped - station / self.n_stations
        coefficients = self.coefficients[interval, station]  # (N, 4, 4, 2)

        ones, zeros = torch.ones_like(u), torch.zeros_like(u)
        in_t = torch.stack([tau**3, tau**2, tau, torch.ones_like(tau)], dim=1)
        rate_t = torch.stack(
            [3 * tau**2, 2 * tau, torch.ones_like(tau), torch.zeros_like(tau)], 1
        )
        in_q = torch.stack([u**3, u**2, u, ones], dim=1)
        slope_q = torch.stack([3 * u**2, 2 * u, ones, zeros], dim=1)
        bend_q = torch.stack([6 * u, 2 * ones, zeros, zeros], dim=1)

        def combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.einsum("na,nb,nabd->nd", a, b, coefficients)

        rate = combine(rate_t, in_q)
        return {
            "P": combine(in_t, in_q) + (t - held).unsqueeze(1) * rate,
            "P_q": combine(in_t, slope_q),
            "P_qq": combine(in_t, bend_q),
            "P_t": rate,
        }

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """Signed distance to the outline at each point's own time, positive in the
        vapour, ``(N, 1)`` for ``x`` of ``(N, 3)``.

        The outline is mirror-symmetric, so each point is reflected onto the
        ``y* > axis`` side (through the smoothed ``|y - axis|`` the capsule uses)
        and projected onto that half.
        """
        t = x[:, 2]
        point = torch.stack(
            [x[:, 0], self.axis + torch.sqrt((x[:, 1] - self.axis) ** 2 + ABS_SMOOTH**2)], dim=1
        )
        with torch.no_grad():
            q = self._nearest_station(point, t)
            for _ in range(PROJECTION_STEPS):
                q = self._newton_step(q, point, t)
        q = self._newton_step(q, point, t)
        frame = self.surface(q, t)
        return ((frame["P"] - point) * _outward(frame["P_q"])).sum(dim=1, keepdim=True)

    def _nearest_station(self, point: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The coarse station nearest each point, at the coarse instant nearest its time."""
        span = self.times[-1] - self.times[0]
        last = self.coarse_points.shape[0] - 1
        instant = ((t - self.times[0]) / span * last).round().long().clamp(0, last)
        gap = ((self.coarse_points[instant] - point.unsqueeze(1)) ** 2).sum(dim=2)
        return self.coarse_q[gap.argmin(dim=1)]

    def _newton_step(
        self, q: torch.Tensor, point: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """One Newton step on ``|P(q) - point|^2 / 2`` over ``q``."""
        frame = self.surface(q, t)
        offset = frame["P"] - point
        speed_sq = (frame["P_q"] ** 2).sum(dim=1)
        slope = (offset * frame["P_q"]).sum(dim=1)
        bend = speed_sq + (offset * frame["P_qq"]).sum(dim=1)
        return q - slope / torch.maximum(bend, NEWTON_FLOOR * speed_sq)

    def front(self, t: torch.Tensor, n_body: int, n_cap: int) -> FrontSamples:
        """The outline at each time in ``t``, sampled in the capsule's layout: the
        upper and lower body, then the root and nose caps, each time-major.

        The caps are the arcs within ``root_cap_q`` and ``nose_cap_q`` of each apex
        (labels only; the curve is the same one throughout), swept by an angle in
        ``[-pi/2, pi/2]`` whose sign is the side, as the capsule's are.
        """
        if n_body < 1 or n_cap < 2:
            raise ValueError(
                f"front() needs n_body >= 1 and n_cap >= 2, got n_body={n_body}, n_cap={n_cap}"
            )
        t = t.reshape(-1)
        device = t.device
        nose, root = self.nose_cap_q, self.root_cap_q
        along = torch.linspace(nose, 1.0 - root, n_body + 2, device=device)[1:-1]
        angle = torch.linspace(-0.5 * math.pi, 0.5 * math.pi, n_cap, device=device)
        sweep = angle.abs() / (0.5 * math.pi)
        side = torch.sign(angle)
        body_u = 1.0 - (along - nose) / (1.0 - nose - root)
        ones_body, ones_cap = torch.ones_like(along), torch.ones_like(angle)
        # (half-outline position from the nose apex, side, u, on_cap, angle) per segment
        segments = [
            (along, ones_body, body_u, 0 * ones_body, 0 * ones_body),
            (along, -ones_body, body_u, 0 * ones_body, 0 * ones_body),
            (1.0 - root * sweep, side, 0 * ones_cap, ones_cap, angle),
            (nose * sweep, side, ones_cap, ones_cap, angle),
        ]
        columns = [[] for _ in range(6)]
        for position, sides, u, on_cap, angles in segments:
            q = torch.where(sides >= 0, 0.5 * position, 1.0 - 0.5 * position)
            for column, values in zip(columns[:5], (q, sides, u, on_cap, angles), strict=True):
                column.append(values.reshape(1, -1).expand(len(t), -1).reshape(-1))
            columns[5].append(t.repeat_interleave(len(position)))
        q, sides, u, on_cap, angles, times = (torch.cat(c) for c in columns)

        frame = self.surface(q, times)
        normal = _outward(frame["P_q"])
        p_q, p_qq = frame["P_q"], frame["P_qq"]
        speed = p_q.norm(dim=1)
        kappa = (p_q[:, 0] * p_qq[:, 1] - p_q[:, 1] * p_qq[:, 0]) / speed**3
        return FrontSamples(
            torch.cat([frame["P"], times.unsqueeze(1)], dim=1),
            u.unsqueeze(1),
            sides.unsqueeze(1),
            on_cap.unsqueeze(1),
            angles.unsqueeze(1),
            kappa.unsqueeze(1),
            normal,
            (frame["P_t"] * normal).sum(dim=1, keepdim=True),
        )

    def nose(self, t: torch.Tensor) -> torch.Tensor:
        """The nose apex's ``x*`` at times ``t``, in ``t``'s shape."""
        apex = self.surface(torch.zeros_like(t.reshape(-1)), t.reshape(-1))["P"][:, 0]
        return apex.reshape(t.shape)


def _outward(tangent: torch.Tensor) -> torch.Tensor:
    """The unit normal out of the vapour: the curve runs counter-clockwise (nose,
    over the ``y* > axis`` side, root, back), so outward is the tangent turned
    clockwise."""
    normal = torch.stack([tangent[:, 1], -tangent[:, 0]], dim=1)
    return normal / normal.norm(dim=1, keepdim=True)


class MeasuredInterface(GeometricInterface):
    """The trained capsule through the warm-up; the measured outline once engaged.

    Built, initialised and trained exactly as :class:`GeometricInterface` (same
    parameters, same random draws), so the shared warm-up is unchanged. The
    trainer engages the outline where it freezes the geometry, and from then on
    the field, the sampled front and the nose are the measured ones. The
    capsule's other scalars (its cap radii, its centreline) stay the trained
    fit's; they size auxiliary terms, not the interface.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outline: MeasuredOutline | None = None
        self.engaged = False

    def bind_outline(self, outline: MeasuredOutline) -> None:
        """Attach the dataset's traced outline, on this geometry's device."""
        self.outline = outline.to(self._s0_raw.device)

    def engage(self, on: bool) -> None:
        """Switch the interface to the measured outline (or back to the capsule)."""
        if on and self.outline is None:
            raise RuntimeError(
                "model.measured_outline is on but no outline is bound: the dataset's "
                "outline.npz is written by the preprocess stage"
            )
        self.engaged = bool(on)

    def forward(self, x, c=None, priors=None):
        if self.engaged:
            return self.outline.phi(x)
        return super().forward(x, c, priors)

    def front(self, t, n_body, n_cap, ctx: GeometryContext | None = None):
        if self.engaged:
            return self.outline.front(t, n_body, n_cap)
        return super().front(t, n_body, n_cap, ctx)

    def nose(self, t, ctx: GeometryContext | None = None):
        if self.engaged:
            return self.outline.nose(t)
        return super().nose(t, ctx)

    def apex(self, t, ctx: GeometryContext | None = None):
        if self.engaged:
            return torch.cat(
                [self.outline.nose(t), torch.full_like(t, self.outline.axis)], dim=1
            )
        return super().apex(t, ctx)
