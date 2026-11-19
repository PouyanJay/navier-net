"""The depth's share of the edge rule (``model.depth_drain``): a 2.5D edge.

The edge rule ties the pressure step across the bubble's edge to the edge's
shape, with the bubble taken as flat across the channel's depth. What the
trained fields leave of that rule, the mismatch, is the part of the edge's
force balance a 2D solution cannot close. At every imaged instant this module
splits the mismatch into three parts:

- the root's part: one law of the distance from the root tip, the same at
  every instant, fitted once and held
  (:class:`~naviernet.models.depth_curve.DepthRootCurve`);
- the drained part: what would have to be handed to the third dimension so
  that the bubble the kept forces would shape stays within a tolerance of the
  real outline, and no more than that (:func:`smallest_drain`);
- the kept part: the rest.

The split is not a setting: it follows from the shape tolerance. Where the
forces already shape the bubble within it, nothing is drained.

The loss charges the whole mismatch outside the root's law at the edge rule's
full weight, drained part included, and charges the drained part a second
time: every unit of it (RMS along a half) costs the same, whatever its size
(``model.depth_drain_price_Pa``). Nothing in the split is cheaper than the
plain rule, so a drain is never the solver's way out of an error; the part of
the mismatch that bends the bubble's shape is the one it pays most for. The
second charge is by the unit and not by the square for two reasons: the pull
on a small drain does not fade, so the solver has a reason to take it all the
way to none, and the pull on a large one does not grow, so an instant the 2D
model cannot bring within the tolerance does not take the run over.

The bubble the kept forces would shape is read off a linear model of the
rebuild: a mismatch in the pressure step is a mismatch in the edge's bend, the
bend summed along the edge tilts it, and the tilt carried along the edge moves
it sideways. Each half of the outline is walked from the nose tip to the root
tip with both tips held and a uniform pressure shift and a uniform stretch
free.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from naviernet.physics.residuals import IN_PLANE_WEIGHT, laplace_jump_residual

# The price of a sideways move against the price of a drain is found by
# bisection, over this many decades either side of the price at which the two
# costs are of one size, in this many steps.
PRICE_DECADES = 8.0
PRICE_STEPS = 30

# Keeps the shift and the stretch, which the drain's cost does not reach,
# determined: a share of the mean length a sample stands for.
RIDGE = 1e-12

# Two instants this close are one: the front's times are single precision.
TIME_SLACK = 1e-6

# The halves an instant's outline is walked in: above and below the axis.
HALVES = 2


class HalfEdges(NamedTuple):
    """The imaged instants' half outlines, one row per instant and side, each
    running from the nose tip to the root tip with the lower halves mirrored
    onto the upper side. ``order`` is ``(C, n)`` indices into the front's
    samples, on the front's device; the rest is double precision on the CPU,
    where the split is solved."""

    order: torch.Tensor
    points: torch.Tensor  # (C, n, 2)
    normal: torch.Tensor  # (C, n, 2), out of the vapour
    steps: torch.Tensor  # (C, n - 1), the distance between neighbours
    cells: torch.Tensor  # (C, n), the length of edge each sample stands for


class Drain(NamedTuple):
    """The smallest drain per half ``(C, n)``, and the rebuilt edge's RMS
    distance from the real one with nothing drained and with it, ``(C,)``."""

    drain: torch.Tensor
    move_before: torch.Tensor
    move_after: torch.Tensor


class DepthSplit(NamedTuple):
    """The edge rule's mismatch in its three parts, ``(N, 1)`` each. ``kept``
    carries the mismatch's gradient, so ``kept + drained`` is the mismatch
    outside the root's law, gradient and all; ``drained`` and ``root`` are
    plain values. ``lengths`` is the length of edge each sample stands for,
    zero before the first image. ``needed`` is the size of the drain (RMS along
    a half), averaged over the front's halves with an instant before the first
    image counting as none: a scalar that carries the gradient of the least
    drain with respect to the mismatch. ``move_before`` and ``move_after`` are
    :class:`Drain`'s."""

    kept: torch.Tensor
    drained: torch.Tensor
    root: torch.Tensor
    lengths: torch.Tensor
    needed: torch.Tensor
    move_before: torch.Tensor
    move_after: torch.Tensor


def imaged(front, starts_at: float) -> torch.Tensor:
    """``True`` on the samples at or after the first image, ``(N, 1)``."""
    return front.points[:, 2:3].detach() >= starts_at - TIME_SLACK


def half_edges(front, starts_at: float) -> HalfEdges | None:
    """The front's samples at the imaged instants as half outlines, or
    ``None`` when no instant is imaged. Relies on the layout both front models
    share: every instant carries the same samples."""
    times = front.points[:, 2].detach()
    seen = imaged(front, starts_at).squeeze(1)
    if not bool(seen.any()):
        return None
    place = _place_along_half(front)
    rows, signs = [], []
    for sign in (1.0, -1.0):
        index = torch.nonzero(seen & (front.side[:, 0] * sign >= 0)).squeeze(1)
        instants, rank = torch.unique(times[index], return_inverse=True)
        index = index[torch.argsort(4.0 * rank.double() + place[index].double())]
        rows.append(index.reshape(len(instants), -1))
        signs.append(torch.full((len(instants), 1, 1), sign, dtype=torch.float64))
    order = torch.cat(rows)
    side = torch.cat(signs)
    mirror = torch.cat([torch.ones_like(side), side], dim=2)
    at = order.cpu()
    points = front.points[:, :2].detach().double().cpu()[at] * mirror
    normal = front.normal.detach().double().cpu()[at] * mirror
    steps = (points[:, 1:] - points[:, :-1]).norm(dim=2)
    cells = torch.zeros(points.shape[:2], dtype=torch.float64)
    cells[:, :-1] += 0.5 * steps
    cells[:, 1:] += 0.5 * steps
    return HalfEdges(order, points, normal, steps, cells)


def _place_along_half(front) -> torch.Tensor:
    """Where each sample sits along its half, ``(N,)``: 0 to 1 over the nose
    cap from the tip, 1 to 2 along the body, 2 to 3 over the root cap to its
    tip."""
    sweep = front.angle[:, 0].abs() / (0.5 * math.pi)
    spine = front.u[:, 0]
    on_cap = front.on_cap[:, 0] > 0.5
    return torch.where(
        on_cap, torch.where(spine >= 0.5, sweep, 3.0 - sweep), 2.0 - spine
    ).detach()


def root_tail(edges: HalfEdges) -> tuple[torch.Tensor, torch.Tensor]:
    """Each sample's distance from its instant's root tip along the channel
    ``(C, n)``, and whether it lies on the tail, behind the half's widest
    point ``(C, n)``: where the root's law applies."""
    x, height = edges.points.unbind(dim=2)
    distance = x - x.min(dim=1, keepdim=True).values
    widest = height.argmax(dim=1, keepdim=True)
    behind = torch.arange(x.shape[1]).reshape(1, -1) > widest
    return distance, behind


def _running_integral(steps: torch.Tensor) -> torch.Tensor:
    """The matrix that sums a quantity along each half from its nose tip by
    the trapezoid rule, ``(C, n, n)``."""
    n = steps.shape[1] + 1
    none = steps.new_zeros(len(steps), 1)
    ahead = torch.cat([steps, none], dim=1)
    behind = torch.cat([none, steps], dim=1)
    row = torch.arange(n).reshape(-1, 1)
    column = torch.arange(n).reshape(1, -1)
    before = (column < row).to(steps.dtype)
    up_to = ((column >= 1) & (column <= row)).to(steps.dtype)
    return 0.5 * (before * ahead[:, None, :] + up_to * behind[:, None, :])


class _Rebuild(NamedTuple):
    """The linear rebuild of each half. The unknowns are the drain at every
    sample, the uniform pressure shift and the uniform stretch, ``n + 2``."""

    move_of_error: torch.Tensor  # (C, n, n): the sideways move the mismatch alone gives
    move_of_unknowns: torch.Tensor  # (C, n, n + 2)
    tip_of_error: torch.Tensor  # (C, 2, n): where the mismatch alone puts the root tip
    tip_of_unknowns: torch.Tensor  # (C, 2, n + 2)


def _rebuild(edges: HalfEdges, stiffness: float) -> _Rebuild:
    """``stiffness`` is the pressure step per unit of in-plane bend."""
    integral = _running_integral(edges.steps)
    nx, ny = edges.normal.unbind(dim=2)
    # A tilt turns the edge's direction toward its inward normal.
    shift_x = -(integral * nx[:, None, :]) @ integral / stiffness
    shift_y = -(integral * ny[:, None, :]) @ integral / stiffness
    offset = edges.points - edges.points[:, :1]
    unknowns_x = torch.cat(
        [-shift_x, shift_x.sum(dim=2, keepdim=True), offset[..., 0:1]], dim=2
    )
    unknowns_y = torch.cat(
        [-shift_y, shift_y.sum(dim=2, keepdim=True), offset[..., 1:2]], dim=2
    )
    return _Rebuild(
        nx[..., None] * shift_x + ny[..., None] * shift_y,
        nx[..., None] * unknowns_x + ny[..., None] * unknowns_y,
        torch.stack([shift_x[:, -1], shift_y[:, -1]], dim=1),
        torch.stack([unknowns_x[:, -1], unknowns_y[:, -1]], dim=1),
    )


def _barred(rebuild: _Rebuild, open_to_drain: torch.Tensor) -> _Rebuild:
    """``rebuild`` with the drain out of reach where ``open_to_drain``
    ``(C, n)`` is false: a drain there moves nothing, so the least one is none."""
    free = open_to_drain.new_ones(len(open_to_drain), 2)
    gate = torch.cat([open_to_drain, free], dim=1).to(rebuild.move_of_unknowns.dtype)
    return rebuild._replace(
        move_of_unknowns=rebuild.move_of_unknowns * gate.unsqueeze(1),
        tip_of_unknowns=rebuild.tip_of_unknowns * gate.unsqueeze(1),
    )


def _apply(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return (matrix @ vector.unsqueeze(2)).squeeze(2)


def _held_tips_alone(error: torch.Tensor, rebuild: _Rebuild) -> torch.Tensor:
    """The unknowns with nothing drained ``(C, n + 2)``: the shift and stretch
    that hold the root tip."""
    n = error.shape[1]
    free = torch.linalg.solve(
        rebuild.tip_of_unknowns[:, :, n:], -_apply(rebuild.tip_of_error, error).unsqueeze(2)
    ).squeeze(2)
    return torch.cat([torch.zeros_like(error), free], dim=1)


def _cheapest_at(
    price: torch.Tensor, error: torch.Tensor, cells: torch.Tensor, rebuild: _Rebuild
) -> torch.Tensor:
    """The unknowns ``(C, n + 2)`` that cost least when a squared sideways
    move costs ``price`` ``(C,)`` times a squared drain, the root tip held."""
    count, n = error.shape
    moves = rebuild.move_of_unknowns
    weighted = moves * cells.unsqueeze(2)
    cost = price.reshape(-1, 1, 1) * (moves.transpose(1, 2) @ weighted)
    cost = cost + RIDGE * cells.mean() * torch.eye(n + 2, dtype=cost.dtype)
    cost[:, :n, :n] += torch.diag_embed(cells)
    pull = -price.reshape(-1, 1) * _apply(
        weighted.transpose(1, 2), _apply(rebuild.move_of_error, error)
    )
    system = torch.zeros(count, n + 4, n + 4, dtype=cost.dtype)
    system[:, : n + 2, : n + 2] = cost
    system[:, : n + 2, n + 2 :] = rebuild.tip_of_unknowns.transpose(1, 2)
    system[:, n + 2 :, : n + 2] = rebuild.tip_of_unknowns
    held = -_apply(rebuild.tip_of_error, error)
    return torch.linalg.solve(system, torch.cat([pull, held], dim=1).unsqueeze(2)).squeeze(2)[
        :, : n + 2
    ]


def _move(
    unknowns: torch.Tensor, error: torch.Tensor, cells: torch.Tensor, rebuild: _Rebuild
) -> torch.Tensor:
    """The rebuilt edge's RMS distance from the real one, ``(C,)``."""
    sideways = _apply(rebuild.move_of_unknowns, unknowns) + _apply(rebuild.move_of_error, error)
    return torch.sqrt((cells * sideways**2).sum(dim=1) / cells.sum(dim=1))


def drain_size(drain: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """The size of each half's drain ``(C,)``: its RMS along the half, by
    length of edge. A half that drains nothing has size zero and no gradient."""
    mean_square = (cells * drain**2).sum(dim=1) / cells.sum(dim=1)
    some = mean_square > 0.0
    root = torch.sqrt(torch.where(some, mean_square, torch.ones_like(mean_square)))
    return torch.where(some, root, torch.zeros_like(mean_square))


@torch.no_grad()
def smallest_drain(
    error: torch.Tensor,
    edges: HalfEdges,
    stiffness: float,
    tolerance: float,
    open_to_drain: torch.Tensor | None = None,
) -> Drain:
    """The least drain (least sum of its squares along the edge) that leaves
    the bubble rebuilt from the kept mismatch, ``error`` less the drain, within
    ``tolerance`` RMS of the real outline. ``error`` is ``(C, n)`` in pressure
    units, double precision. A half already within the tolerance drains
    nothing. ``open_to_drain`` ``(C, n)`` bars the drain from the samples where
    it is false; none bars it nowhere."""
    rebuild = _rebuild(edges, stiffness)
    if open_to_drain is not None:
        rebuild = _barred(rebuild, open_to_drain)
    cells = edges.cells
    undrained = _held_tips_alone(error, rebuild)
    before = _move(undrained, error, cells, rebuild)
    drain = torch.zeros_like(error)
    after = before.clone()
    needs = torch.nonzero(before > tolerance).squeeze(1)
    if len(needs) == 0:
        return Drain(drain, before, after)
    error, cells = error[needs], cells[needs]
    rebuild = _Rebuild(*(part[needs] for part in rebuild))
    matched = cells.sum(dim=1) / (cells.unsqueeze(2) * rebuild.move_of_error**2).sum(
        dim=(1, 2)
    ).clamp(min=1e-300)
    low = torch.full_like(matched, -PRICE_DECADES)
    high = torch.full_like(matched, PRICE_DECADES)
    for _ in range(PRICE_STEPS):
        middle = 0.5 * (low + high)
        unknowns = _cheapest_at(matched * 10.0**middle, error, cells, rebuild)
        within = _move(unknowns, error, cells, rebuild) <= tolerance
        high = torch.where(within, middle, high)
        low = torch.where(within, low, middle)
    unknowns = _cheapest_at(matched * 10.0**high, error, cells, rebuild)
    drain[needs] = unknowns[:, : error.shape[1]]
    after[needs] = _move(unknowns, error, cells, rebuild)
    return Drain(drain, before, after)


def _spread(
    values: torch.Tensor, order: torch.Tensor, size: int, mean: bool, dtype: torch.dtype
) -> torch.Tensor:
    """Per-half values ``(C, n)`` back on the front's samples, ``(size, 1)``
    of ``dtype`` on the front's device: a sample two halves share (a tip)
    takes their mean, or their sum."""
    flat = order.reshape(-1)
    values = values.reshape(-1).to(dtype).to(flat.device)
    total = torch.zeros(size, dtype=values.dtype, device=flat.device).index_add(0, flat, values)
    if mean:
        count = torch.zeros_like(total).index_add(0, flat, torch.ones_like(values))
        total = total / count.clamp(min=1.0)
    return total.unsqueeze(1)


def root_part(model, front, edges: HalfEdges) -> torch.Tensor:
    """The root law's part of the pressure step at each sample, ``(N, 1)``:
    the law on the tail, zero elsewhere."""
    distance, on_tail = root_tail(edges)
    size = len(front.points)
    dtype = front.points.dtype
    reach = _spread(distance, edges.order, size, mean=True, dtype=dtype)
    tail = _spread(on_tail, edges.order, size, mean=True, dtype=dtype)
    return model.depth_root(reach) * (tail > 0.5).to(dtype)


def depth_split(model, front, groups: dict[str, float], view_x_max: float | None) -> DepthSplit:
    """The edge rule's mismatch on ``front`` in its root, drained and kept
    parts. Before the first image there is no split: all of it is kept. Past
    ``view_x_max`` the outline is a guess: its mismatch is left out of the
    rebuild, as it is left out of the loss, and nothing is drained there."""
    mismatch = laplace_jump_residual(model, front, groups)
    nothing = torch.zeros_like(mismatch)
    edges = half_edges(front, model.depth_root.starts_at)
    if edges is None:
        empty = torch.zeros(0, dtype=torch.float64)
        return DepthSplit(
            mismatch, nothing, nothing, nothing.detach(), mismatch.new_zeros(()), empty, empty
        )
    law = root_part(model, front, edges)
    rest = mismatch - law
    seen = torch.ones_like(rest, dtype=torch.bool)
    if view_x_max is not None:
        seen = front.points[:, 0:1].detach() <= view_x_max
    at = edges.order.cpu()
    solved = smallest_drain(
        (rest.detach() * seen.to(rest.dtype)).double().cpu()[at, 0],
        edges,
        IN_PLANE_WEIGHT / groups["We"],
        float(model.depth_drain_tolerance),
        open_to_drain=seen.cpu()[at, 0],
    )
    size = len(mismatch)
    drain = _spread(solved.drain, edges.order, size, mean=True, dtype=mismatch.dtype)
    lengths = _spread(edges.cells, edges.order, size, mean=False, dtype=mismatch.dtype)
    return DepthSplit(
        rest - drain,
        drain,
        law,
        lengths,
        _needed(solved.drain, rest, edges, front),
        solved.move_before,
        solved.move_after,
    )


def _needed(drain: torch.Tensor, rest: torch.Tensor, edges: HalfEdges, front) -> torch.Tensor:
    """The size of the least drain averaged over the front's halves, a scalar,
    an instant before the first image counting as none. The least drain's sum
    of squares changes with the mismatch at twice the drain itself, a length
    of edge at a time (the multiplier rule of the problem
    :func:`smallest_drain` solves), which is the gradient a drain that moves
    one for one with the mismatch has: so each half's drain ``(C, n)`` is
    carried on ``rest``'s gradient."""
    moving = rest - rest.detach()
    carried = drain.to(rest.dtype).to(rest.device) + moving[edges.order, 0]
    sizes = drain_size(carried, edges.cells.to(rest.dtype).to(rest.device))
    instants = len(torch.unique(front.points[:, 2].detach()))
    return sizes.sum() / (HALVES * instants)


@torch.no_grad()
def fit_root_curve(model, front, groups: dict[str, float], view_x_max: float | None) -> None:
    """Set the root's law to the fit of the edge rule's mismatch on the tails
    of ``front``, each sample counted by the length of edge it stands for: done
    once, when a run takes ``model.depth_drain`` on."""
    law = model.depth_root
    edges = half_edges(front, law.starts_at)
    if edges is None:
        return
    with torch.enable_grad():
        mismatch = laplace_jump_residual(model, front, groups).detach()
    at = edges.order.cpu()
    distance, on_tail = root_tail(edges)
    weights = edges.cells * on_tail
    if view_x_max is not None:
        weights = weights * (edges.points[..., 0] <= view_x_max)
    law.fit(
        distance.reshape(-1, 1),
        mismatch.double().cpu()[at, 0].reshape(-1, 1),
        weights.reshape(-1, 1),
    )


def part_forces(split: DepthSplit) -> dict[str, float]:
    """Each part's force on the edge per unit of depth, in the run's units: its
    size summed along the edge over the imaged samples of ``split``."""
    parts = (("kept", split.kept), ("drained", split.drained), ("root", split.root))
    return {name: float((part.detach().abs() * split.lengths).sum()) for name, part in parts}


def force_shares(split: DepthSplit) -> dict[str, float]:
    """The three parts as shares of the edge's whole mismatch force."""
    forces = part_forces(split)
    whole = max(sum(forces.values()), 1e-30)
    return {name: force / whole for name, force in forces.items()}
