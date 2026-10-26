"""Potential flows in the channel for the exact liquid velocity (``model.exact_continuity``).

The liquid's velocity is built as ``U_in(y) e_x + grad Phi + curl(W N)``, every
part divergence-free (see :meth:`BubblePINN._exact_velocity`). This module
holds the two harmonic fields that construction needs at each instant of the
frozen outline:

* ``Phi``, the potential that carries the kinematic condition: its normal
  derivative on the front is ``V_n - n.U_in``, no flux crosses the side walls or
  the inlet plane, so the liquid moves with the interface and all of the
  bubble's growth leaves through the outlet;
* ``C``, zero on the front and one on both walls, so ``0 < C < 1`` in the liquid
  (the maximum principle). The stream function's weight ``W = wall_factor * C``
  vanishes on the front and the walls with no corner at the cap seams, and
  across a gap of any width ``C`` rises as ``d / w``: nothing screens it.

Both are sums of a Green function of the strip over sources placed just inside
the vapour (the method of fundamental solutions), so they are exactly harmonic
in the liquid and smooth to every order. With ``z = x + i (y - y_lo)`` and
``k = pi / (2 H)``, a unit source at ``z_0`` is ``Re F`` with::

    F(z) = (1 / 2 pi) sum_images sign_m log sinh(k (z - z_m))

``Phi`` uses the Neumann images ``x_0 +/- i eta_0`` and ``(2 x_in - x_0) +/- i
eta_0``, all positive (outward flux one, all of it downstream). ``C`` uses the
Dirichlet images ``x_0 + i eta_0`` (positive) and ``x_0 - i eta_0`` (negative),
zero on both walls. Every derivative is analytic, ``d^a/dx^a d^b/dy^b Re F =
Re(i^b F^(a+b))``, so :class:`_HarmonicDerivative` evaluates any of them in
closed form and its backward is the same function one order higher: autograd
differentiates the fields to any order without recording a graph through the
source sum.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from naviernet.utils.logging import get_logger

log = get_logger(__name__)

# The relative Tikhonov term on the least-squares solves: below the conditioning
# the source offsets give on the real outline (1e1 to 1e4), above roundoff.
TIKHONOV = 1e-10
# Points per chunk of the source sum: bounds the (points x images) complex
# matrix at about 50 MB.
CHUNK = 2048
# Halvings of a source's offset before a source that will not sit inside the
# vapour is an error.
OFFSET_HALVINGS = 6
# Distinct instants whose solves are kept. Training draws fresh instants every
# step, so only the fixed quadratures and evaluation frames repeat.
CACHE_SIZE = 64
# Highest derivative order of F supported. The velocity is order one, the
# momentum law's Laplacian adds two, and the backward pass to the collocation
# points one more; the rest is headroom.
MAX_ORDER = 8

# Where the trainer scores the inlet condition (BubbleDataset.sample_boundary
# puts its inlet points at x* = 0): the potentials' image plane. Kept here, not
# in the dataset module, whose source keys the shared warm-up.
INLET_X = 0.0

# A kinematic miss above this share of the front's largest speed is logged: the
# fit degrades where the outline necks and the sources crowd.
KINEMATIC_WARNING = 0.05

NEUMANN = "neumann"
DIRICHLET = "dirichlet"


def _coth_polynomials(max_order: int) -> list[np.ndarray]:
    """``P_n`` with ``d^n coth(w) / dw^n = P_n(coth w)``: ``P_0 = c`` and
    ``P_{n+1} = (1 - c^2) P_n'(c)``. Coefficients lowest power first."""
    poly = np.polynomial.polynomial
    polys = [np.array([0.0, 1.0])]
    for _ in range(max_order):
        polys.append(poly.polymul([1.0, 0.0, -1.0], poly.polyder(polys[-1])))
    return polys


_COTH_POLYS = _coth_polynomials(MAX_ORDER)


@dataclass(frozen=True)
class Channel:
    """The strip ``y_lo < y < y_hi`` with its inlet plane at ``x_in``."""

    y_lo: float
    y_hi: float
    x_in: float

    @property
    def k(self) -> float:
        return math.pi / (2.0 * (self.y_hi - self.y_lo))


class HarmonicField:
    """``offset + sum_j q_j G(x; source_j)`` in the channel, ``G`` the strip's
    Neumann (``boundary="neumann"``) or Dirichlet Green function."""

    def __init__(
        self,
        channel: Channel,
        sources: torch.Tensor,
        strengths: torch.Tensor,
        offset: float = 0.0,
        boundary: str = NEUMANN,
    ):
        self.channel, self.offset, self.boundary = channel, float(offset), boundary
        self.sources, self.raw_strengths = sources, strengths
        x0 = sources[:, 0].double()
        eta0 = sources[:, 1].double() - channel.y_lo
        images = [torch.complex(x0, eta0), torch.complex(x0, -eta0)]
        signs = [1.0, 1.0]
        if boundary == NEUMANN:
            mirrored = 2.0 * channel.x_in - x0
            images += [torch.complex(mirrored, eta0), torch.complex(mirrored, -eta0)]
            signs += [1.0, 1.0]
        elif boundary == DIRICHLET:
            signs = [1.0, -1.0]
        else:
            raise ValueError(f"boundary must be {NEUMANN!r} or {DIRICHLET!r}, got {boundary!r}")
        self.images = torch.cat(images)
        self.signs = torch.tensor(signs, dtype=torch.float64, device=self.images.device)
        weights = strengths.double().to(self.images.device)
        self.strengths = torch.cat([s * weights for s in signs]) / (2.0 * math.pi)

    def to(self, device) -> HarmonicField:
        return HarmonicField(
            self.channel,
            self.sources.to(device),
            self.raw_strengths.to(device),
            self.offset,
            self.boundary,
        )

    def value(self, xy: torch.Tensor, a: int, b: int) -> torch.Tensor:
        """``d^a/dx^a d^b/dy^b`` of the field at ``xy``, ``(N, 1)`` in ``xy``'s dtype."""
        out = self._sum(xy, a, b, lambda terms: terms @ self.strengths).reshape(-1, 1)
        if a + b == 0:
            out = out + self.offset
        return out.to(dtype=xy.dtype, device=xy.device)

    def basis(self, xy: torch.Tensor, a: int, b: int) -> torch.Tensor:
        """The same derivative of each source's own field (its images summed,
        strength one), ``(N, n_sources)`` float64: the solves' matrices."""
        n_images = len(self.signs)
        n_sources = len(self.images) // n_images

        def reduce(terms):
            blocks = terms.reshape(len(terms), n_images, n_sources)
            return (blocks * self.signs.reshape(1, -1, 1)).sum(dim=1) / (2.0 * math.pi)

        return self._sum(xy, a, b, reduce)

    def _sum(self, xy: torch.Tensor, a: int, b: int, reduce) -> torch.Tensor:
        n = a + b
        if n > MAX_ORDER + 1:
            raise ValueError(f"derivative order {n} exceeds the supported {MAX_ORDER + 1}")
        xy = xy.detach().to(device=self.images.device, dtype=torch.float64)
        z = torch.complex(xy[:, 0], xy[:, 1] - self.channel.y_lo)
        return torch.cat(
            [
                reduce(self._terms(z[i : i + CHUNK], n, b))
                for i in range(0, max(len(z), 1), CHUNK)
            ]
        )

    def _terms(self, z: torch.Tensor, n: int, b: int) -> torch.Tensor:
        """Each image's term of the derivative of order ``n`` (``b`` of them in
        y), ``(N, n_images)``, without strengths or signs."""
        k = self.channel.k
        w = k * (z.unsqueeze(1) - self.images.unsqueeze(0))
        if n == 0:
            # Re log sinh w = log |sinh w|, single-valued: no branch cut.
            return 0.5 * torch.log(torch.sinh(w.real) ** 2 + torch.sin(w.imag) ** 2)
        coth = 1.0 / torch.tanh(w)
        poly = torch.zeros_like(coth)
        for coefficient in _COTH_POLYS[n - 1][::-1]:
            poly = poly * coth + coefficient
        return ((1j) ** (b % 4) * (k**n) * poly).real


class _HarmonicDerivative(torch.autograd.Function):
    """A derivative of a :class:`HarmonicField` as a differentiable function of
    the points: its backward is the next two derivatives, so the graph holds
    only the points and higher derivatives are taken in closed form."""

    @staticmethod
    def forward(ctx, xy, field, a, b):
        ctx.save_for_backward(xy)
        ctx.field, ctx.order = field, (a, b)
        return field.value(xy, a, b)

    @staticmethod
    def backward(ctx, grad):
        (xy,) = ctx.saved_tensors
        a, b = ctx.order
        d_x = _HarmonicDerivative.apply(xy, ctx.field, a + 1, b)
        d_y = _HarmonicDerivative.apply(xy, ctx.field, a, b + 1)
        return grad * torch.cat([d_x, d_y], dim=1), None, None, None


def harmonic(field: HarmonicField, xy: torch.Tensor, a: int = 0, b: int = 0) -> torch.Tensor:
    """``d^a/dx^a d^b/dy^b`` of ``field`` at the points ``xy`` ``(N, 2)``, ``(N, 1)``,
    differentiable to any order."""
    return _HarmonicDerivative.apply(xy, field, a, b)


@dataclass(frozen=True)
class FrontSample:
    """One instant's front: points ``(n, 2)``, outward normals (out of the
    vapour), normal speeds and arclength weights ``(n,)``, all float64 on CPU."""

    points: torch.Tensor
    normal: torch.Tensor
    speed: torch.Tensor
    spacing: torch.Tensor


@dataclass(frozen=True)
class FrontProblem:
    """What one instant's solves fit: the front sampled for the sources and,
    about twice as dense, for the conditions, and ``n . U_in`` at the latter."""

    sources: FrontSample
    collocation: FrontSample
    inflow_normal: torch.Tensor


@dataclass(frozen=True)
class SourceLayout:
    """Each source ``offset`` local spacings inside the vapour; under a mirror
    axis the sampled half's sources and their reflections share strengths."""

    offset: float
    mirror_axis: float | None


@dataclass(frozen=True)
class LiquidFields:
    """One instant's two potentials, and the root-mean-square miss of ``Phi``'s
    condition at the collocation points."""

    potential: HarmonicField
    capacity: HarmonicField
    kinematic_rms: float


@dataclass(frozen=True)
class _Unknowns:
    """The physical sources the unknowns stand for: rows ``j`` and ``j + m`` of
    ``points`` share unknown ``j`` when ``multiplicity`` is 2 (a mirror pair)."""

    channel: Channel
    points: torch.Tensor
    multiplicity: int

    @classmethod
    def placed(cls, channel: Channel, points: torch.Tensor, mirror_axis: float | None):
        if mirror_axis is None:
            return cls(channel, points, 1)
        reflected = torch.stack([points[:, 0], 2.0 * mirror_axis - points[:, 1]], dim=1)
        return cls(channel, torch.cat([points, reflected]), 2)

    def field(self, strengths: torch.Tensor, offset: float = 0.0, boundary: str = NEUMANN):
        return HarmonicField(
            self.channel, self.points, strengths.repeat(self.multiplicity), offset, boundary
        )

    def basis(self, xy: torch.Tensor, a: int, b: int, boundary: str) -> torch.Tensor:
        """``(n_points, n_unknowns)``: each unknown's unit field differentiated at ``xy``."""
        unit = HarmonicField(
            self.channel,
            self.points,
            torch.ones(len(self.points), dtype=torch.float64),
            boundary=boundary,
        )
        m = len(self.points) // self.multiplicity
        return unit.basis(xy, a, b).reshape(len(xy), self.multiplicity, m).sum(dim=1)


def _regularised_solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """``argmin |A q - b|^2 + (TIKHONOV sigma_max)^2 |q|^2``."""
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    lam = TIKHONOV * s[0]
    return vh.T @ ((s / (s**2 + lam**2)) * (u.T @ rhs))


def _place_sources(
    channel: Channel, sources: FrontSample, offset: float, inside
) -> torch.Tensor:
    """Each source ``offset`` local spacings into the vapour behind its sample,
    halved until ``inside`` (a callable on ``(n, 2)``) says it is in the vapour.
    A source upstream of the inlet plane would put its image in the liquid."""
    if float(sources.points[:, 0].min()) <= channel.x_in:
        raise ValueError(
            f"the front reaches the inlet plane x = {channel.x_in}: its potential flow "
            "would mirror sources into the liquid"
        )
    depth = offset * sources.spacing.clone()
    for _ in range(OFFSET_HALVINGS):
        placed = sources.points - depth.unsqueeze(1) * sources.normal
        outside = ~inside(placed)
        if not bool(outside.any()):
            return placed
        depth = torch.where(outside, 0.5 * depth, depth)
    raise ValueError(
        f"{int(outside.sum())} potential-flow sources would not sit inside the vapour; "
        "lower model.exact_continuity_offset"
    )


def _solve_potential(unknowns: _Unknowns, problem: FrontProblem) -> tuple[HarmonicField, float]:
    """``Phi`` with ``dPhi/dn = V_n - n.U_in`` at the collocation points, in
    least squares, and the root-mean-square miss there."""
    xy, normal = problem.collocation.points, problem.collocation.normal
    flux = normal[:, :1] * unknowns.basis(xy, 1, 0, NEUMANN)
    flux = flux + normal[:, 1:2] * unknowns.basis(xy, 0, 1, NEUMANN)
    rhs = problem.collocation.speed - problem.inflow_normal
    q = _regularised_solve(flux, rhs)
    miss = float(torch.sqrt(((flux @ q - rhs) ** 2).mean()))
    scale = float(problem.collocation.speed.abs().max())
    if miss > KINEMATIC_WARNING * max(scale, 1e-12):
        log.warning(
            "the liquid's potential flow misses the front's speed by %.1f%% rms of its "
            "largest; the outline may neck too thinly for model.exact_continuity's sources",
            100.0 * miss / scale,
        )
    return unknowns.field(q), miss


def _solve_capacity(unknowns: _Unknowns, problem: FrontProblem) -> HarmonicField:
    """``C = 1 + sum p_j G_D``: one on the walls by construction, zero at the
    collocation points in least squares."""
    xy = problem.collocation.points
    values = unknowns.basis(xy, 0, 0, DIRICHLET)
    p = _regularised_solve(values, -torch.ones(len(xy), dtype=values.dtype))
    return unknowns.field(p, offset=1.0, boundary=DIRICHLET)


def solve_liquid_fields(
    channel: Channel, problem: FrontProblem, layout: SourceLayout, inside
) -> LiquidFields:
    """Both potentials at one instant. The fronts are the upper half only under
    a mirror axis; ``inside`` says which points ``(n, 2)`` lie in the vapour."""
    placed = _place_sources(channel, problem.sources, layout.offset, inside)
    unknowns = _Unknowns.placed(channel, placed, layout.mirror_axis)
    potential, miss = _solve_potential(unknowns, problem)
    return LiquidFields(potential, _solve_capacity(unknowns, problem), miss)


class FieldCache:
    """The solved fields by instant, oldest dropped first."""

    def __init__(self, size: int = CACHE_SIZE):
        self.size = size
        self._fields: OrderedDict[float, LiquidFields] = OrderedDict()

    def get(self, instant: float, solve) -> LiquidFields:
        key = round(float(instant), 7)
        if key not in self._fields:
            self._fields[key] = solve()
            while len(self._fields) > self.size:
                self._fields.popitem(last=False)
        return self._fields[key]

    def clear(self) -> None:
        self._fields.clear()
