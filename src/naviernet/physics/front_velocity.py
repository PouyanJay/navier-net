"""Measured front velocity as supervision on the explicit interface.

The front-geometry construction already knows how fast it thinks the interface
is moving: :attr:`~naviernet.models.geometry.FrontSamples.normal_speed` is the
front's own motion, taken from the parameterisation. The sharp-interface
kinematic condition ties that speed to the model's own ``u.n`` -- one part of
the model checked against another. This module ties it to the DATA instead,
using the normal velocity measured from consecutive masks (see
:mod:`naviernet.data.velocimetry`).

Why it is worth a term at all when the shape is already supervised at every
training frame: the shape is, the RATE is not. Under this parameterisation
``rate(u)`` is precisely what carries the profile past the last supervised
frame, and today it is inferred indirectly from a handful of snapshots.
Structure carries a trend forward; measurement makes the trend correct.

Two terms, both normalised by the dataset's own measured front speed so they
arrive O(1) against the rest of the objective:

- ``fv_normal``: the binned normal-velocity PROFILE along the front. Per-sample
  matching would inject single-pixel mask noise straight into ``rate(u)``, so
  samples are averaged within bins first -- separately per front segment, since
  the upper and lower profiles at the same ``u`` are different points with
  different normals and collapsing them would confuse a widening bubble with a
  drifting one.
- ``fv_apex``: the nose apex's full 2-D displacement over the same interval.
  The apex is the one interface point with an honest frame-to-frame
  correspondence, so it is the one place a true ``(vx, vy)`` is measurable
  rather than just a normal projection. Compared as a displacement over the
  pair's own ``dt`` -- the same finite difference the measurement is -- so this
  term carries no discretisation mismatch at all.

Like the kinematic constraints, both sit OUTSIDE causal weighting, RBA and the
gradient-norm rebalancer: they are supervised functionals on the front, not
pointwise PDE residuals, and causal weighting would down-weight exactly the late
frames whose rate matters most.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from naviernet.data.dataset import BubbleDataset
from naviernet.data.velocimetry import normal_velocity, survives_blur
from naviernet.models.geometry import FrontSamples

# Floor on the reference speed every term is normalised by. A dataset whose
# front never moved measures a rate of 0, and dividing by it would turn a term
# meant to be O(1) into an infinity.
RATE_FLOOR = 1e-3

# A bilinearly interpolated validity weight must be this close to 1 for the
# sample to count: anything less means one of the four surrounding pixels had no
# usable measurement, and a partially-invalid blend is not a measurement.
VALID_TOL = 1e-6


@dataclass(frozen=True)
class FrontVelocityContext:
    """Per-call inputs: the model to evaluate the front with (a dataset's bound
    view on a joint run, the plain model otherwise) and the training config."""

    model: object
    tcfg: object


@dataclass(frozen=True)
class FrontVelocityPlan:
    """One dataset's measured front kinematics, frozen before the training loop.

    Deterministic and built once, so a resumed run supervises against exactly
    the same measurement it was trained on.

    ``v_n``/``weight`` are ``(P, H, W)`` rasters over the dataset's own pixel
    grid, one slice per usable frame pair; ``weight`` is 1 where the measurement
    is usable and 0 where it is not (an invalid pixel, or a gradient too small
    to define a normal). ``apex_shift`` is the measured apex displacement over
    each pair, the one true 2-D observation.
    """

    times: torch.Tensor  # (P, 1) the frame-k time of each pair
    times_next: torch.Tensor  # (P, 1) the frame-(k+1) time
    v_n: torch.Tensor
    weight: torch.Tensor
    origin: tuple[float, float]  # (x*, y*) of pixel (0, 0)
    spacing: tuple[float, float]  # (dx*, dy*) per pixel
    apex_shift: torch.Tensor  # (P, 2)
    n_body: int
    n_cap: int
    bins_body: int  # profile bins per body segment (upper, lower)
    bins_cap: int  # profile bins per end cap (root, nose)
    v_ref: float
    pairs: tuple[tuple[int, int], ...]  # the dataset rows each slice came from

    @property
    def bins_per_time(self) -> int:
        """Bins one frame pair contributes: two body profiles and two caps."""
        return 2 * (self.bins_body + self.bins_cap)


def build_plan(data: BubbleDataset, cfg, device) -> FrontVelocityPlan:
    """Measure this dataset's front velocity over every usable frame pair.

    Raises when no pair is usable: with the front-velocity term switched on and
    nothing to supervise it with, silently contributing zero would be a feature
    that reports success and does nothing.
    """
    pairs = data.supervised_pairs
    if not pairs:
        raise ValueError(
            f"training.front_velocity needs at least one pair of consecutive, "
            f"training-visible frames to measure a velocity over, and dataset "
            f"{data.cfg.dataset!r} has none. Its event frames are "
            f"{data.event_frames} and {data.validation_frames} are held out of "
            f"supervision (training.holdout_frame / training.val_fraction)."
        )
    return _assemble_plan(data, cfg, device, _measure_pairs(data, cfg.training, pairs))


class _Measured(NamedTuple):
    """What the masks alone say, before any of it becomes a tensor: the pairs it
    came from, the grid it was measured on, one velocity raster and usability
    mask per pair, and each pair's apex displacement."""

    pairs: list[tuple[int, int]]
    spacing: tuple[float, float]  # (dy*, dx*) per pixel, as np.gradient orders it
    v_n: list[np.ndarray]
    weight: list[np.ndarray]
    apex_shift: list[list[float]]


def _measure_pairs(data: BubbleDataset, tcfg, pairs) -> _Measured:
    """Measure every usable frame pair. Pure measurement -- no config beyond the
    blur, no tensors, nothing about the model."""
    smooth_px = float(tcfg.fv_smooth_px)
    spacing = (float(data.y[1] - data.y[0]), float(data.x[1] - data.x[0]))
    measured = _Measured(list(pairs), spacing, [], [], [])
    for row, nxt in pairs:
        dt = float(data.t[nxt]) - float(data.t[row])
        v_n, resolvable = normal_velocity(data.sdf[row], data.sdf[nxt], dt, spacing, smooth_px)
        measured.v_n.append(v_n)
        # Both frames' validity, shrunk by the blur's reach: a pixel the blur
        # mixed with invalid data is not a measurement, however resolvable it looks.
        measured.weight.append(
            resolvable
            & survives_blur(data.valid[row] > 0, smooth_px)
            & survives_blur(data.valid[nxt] > 0, smooth_px)
        )
        here, there = data.front_apex(row), data.front_apex(nxt)
        measured.apex_shift.append([there[0] - here[0], there[1] - here[1]])
    return measured


def _assemble_plan(data: BubbleDataset, cfg, device, measured: _Measured) -> FrontVelocityPlan:
    """The measurement plus the run's fixed sampling parameters, on ``device``."""
    n_body = int(cfg.model.front_body_samples)
    n_cap = int(cfg.model.front_cap_samples)
    per_bin = int(cfg.training.fv_samples_per_bin)

    def tensor(values) -> torch.Tensor:
        return torch.tensor(np.asarray(values), dtype=torch.float32, device=device)

    def times_of(rows) -> torch.Tensor:
        return tensor([[float(data.t[r])] for r in rows])

    return FrontVelocityPlan(
        times=times_of([row for row, _ in measured.pairs]),
        times_next=times_of([nxt for _, nxt in measured.pairs]),
        v_n=tensor(np.stack(measured.v_n)),
        weight=tensor(np.stack(measured.weight).astype(np.float32)),
        origin=(float(data.x[0]), float(data.y[0])),
        # (dx*, dy*): the plan indexes rasters by column then row, the reverse of
        # the (dy*, dx*) order np.gradient took the measurement in.
        spacing=(measured.spacing[1], measured.spacing[0]),
        apex_shift=tensor(measured.apex_shift),
        n_body=n_body,
        n_cap=n_cap,
        bins_body=_bin_count(n_body, per_bin),
        bins_cap=_bin_count(n_cap, per_bin),
        v_ref=max(float(data.nose_rate), RATE_FLOOR),
        pairs=tuple(measured.pairs),
    )


def _bin_count(n_samples: int, per_bin: int) -> int:
    """Bins for a segment of ``n_samples``, at ``per_bin`` samples each. At least
    one: a segment too short to fill a bin is averaged whole rather than being
    given a bin count it cannot populate."""
    return max(1, n_samples // per_bin)


def front_velocity_losses(
    ctx: FrontVelocityContext, plan: FrontVelocityPlan
) -> tuple[torch.Tensor, dict[str, float]]:
    """The weighted front-velocity total for one step, plus per-term floats for
    the log. Recorded values are the raw (unweighted) term magnitudes."""
    tcfg = ctx.tcfg
    normal = _normal_profile_loss(ctx, plan)
    apex = _apex_loss(ctx, plan)
    total = float(tcfg.fv_weight) * normal + float(tcfg.fv_apex_weight) * apex
    return total, {"fv_normal": float(normal.detach()), "fv_apex": float(apex.detach())}


def _normal_profile_loss(ctx: FrontVelocityContext, plan: FrontVelocityPlan) -> torch.Tensor:
    """Squared error between the model's normal-speed profile and the measured
    one, both averaged into the same bins.

    The model side is an instantaneous ``dP/dt . n`` and the measured side a
    one-frame difference; they agree to first order in ``dt``, which is the same
    order the measurement itself is good to -- everywhere the front is not the
    nose, which :func:`_off_nose_cap` explains and excludes.
    """
    front = ctx.model.front(plan.times, plan.n_body, plan.n_cap)
    pair = _pair_index(plan, front.points[:, 2:3])
    measured, usable = _sample_rasters(plan, front.points, pair)
    usable = usable * _off_nose_cap(front)

    bins = _bin_index(plan, front, pair)
    n_bins = plan.times.shape[0] * plan.bins_per_time
    # Masked once, here: an unusable sample contributes nothing to either sum and
    # nothing to the count, so each bin averages exactly the samples it measured.
    model_sum = _bin_sum(front.normal_speed * usable, bins, n_bins)
    measured_sum = _bin_sum(measured * usable, bins, n_bins)
    counts = _bin_sum(usable, bins, n_bins)

    occupied = counts.reshape(-1) > 0
    if not bool(occupied.any()):
        raise RuntimeError(
            "the front-velocity term found no usable measurement anywhere on the "
            "model's front -- every sample fell outside the measured rasters or on "
            "pixels with no valid measurement. The front has left the imaged domain; "
            "this is a diverged run, not a configuration to work around."
        )
    counts = counts[occupied]
    difference = (model_sum[occupied] - measured_sum[occupied]) / (counts * plan.v_ref)
    return (difference**2).mean()


def _off_nose_cap(front: FrontSamples) -> torch.Tensor:
    """1 where the level-set estimate is trustworthy, 0 on the nose cap.

    The estimate is a difference taken at a fixed point over a whole frame, so it
    is first-order in the distance the front travels -- and the nose is where
    that distance is largest against the smallest radius of curvature. Measured
    on a trained run (fvb-w10-s0, Series-1): the nose cap is 10% of the profile
    bins but carried 76% of this term's entire residual, at a mean squared error
    of 0.054 against 0.0014 on the body. Worse, the error is signed and one-way
    -- the model sat +0.194 ABOVE the measurement there, because ``fv_apex`` had
    correctly pulled the nose onto its tracked displacement and this term was
    pulling it back down. Two terms disagreeing at one point, with the biased one
    outvoting the exact one four to one.

    So each estimator supervises only where it is valid: the profile term carries
    the body, where the front moves ~2 px per frame against a ~75 px radius and
    the estimate is solid, and the nose is left to ``fv_apex``, which tracks a
    real feature and involves no level-set approximation at all.

    The ROOT cap stays in. The root barely moves, so its displacement is small
    and the estimate is good there (measured error 0.0003), and "the root does
    not move" is a real thing to hold the front to.
    """
    return (~((front.on_cap > 0) & (front.u >= 0.5))).float()


def _apex_loss(ctx: FrontVelocityContext, plan: FrontVelocityPlan) -> torch.Tensor:
    """Squared error in the nose apex's 2-D displacement over each frame pair.

    The model's own finite difference over the pair's ``dt``, against the
    measured one -- like against like, so no discretisation error enters.
    """
    shift = ctx.model.apex(plan.times_next) - ctx.model.apex(plan.times)
    dt = plan.times_next - plan.times
    return (((shift - plan.apex_shift) / (dt * plan.v_ref)) ** 2).mean()


def _pair_index(plan: FrontVelocityPlan, t: torch.Tensor) -> torch.Tensor:
    """Which raster slice each front sample belongs to, ``(N,)``.

    The front was sampled AT ``plan.times``, and ``_sample_grid`` only reshapes
    and repeats those values -- it does no arithmetic on them -- so an exact
    search is exact and no tolerance is involved.

    That invariant is CHECKED rather than trusted. If a future front ever
    perturbed its times, ``searchsorted`` would quietly hand every sample to a
    neighbouring frame pair and supervise it against the wrong measurement --
    a wrong answer with no symptom, which is the one failure mode this feature
    is built to refuse.
    """
    times = plan.times.reshape(-1).contiguous()
    index = torch.searchsorted(times, t.reshape(-1).contiguous()).clamp(max=times.numel() - 1)
    if not torch.equal(times[index], t.reshape(-1)):
        raise RuntimeError(
            "front samples did not come back at the times they were asked for, so "
            "they cannot be matched to the frame pair each was measured against. "
            "The front-velocity term requires model.front(t, ...) to sample "
            "exactly at t."
        )
    return index


def _sample_rasters(
    plan: FrontVelocityPlan, points: torch.Tensor, pair: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly sample the measured velocity at ``points``, and a 0/1 weight
    saying whether that sample is a measurement at all -- 0 outside the rasters,
    or where any of the four surrounding pixels had no usable measurement. The
    velocity is returned unmasked; the caller applies the weight once, where it
    also divides by it.

    Sampling the measured field at the MODEL's front (rather than pairing points
    between the two contours) is what makes this well-defined while the two are
    still far apart: off the interface the measured field is the level-set
    extension of the front's normal speed, so a nearby point reads a sensible
    value instead of a mismatched partner's.

    Detached in full, query point included: the target is a measurement, not a
    function of the model, and the gradient of a bilinearly interpolated pixel
    field is a piecewise-constant staircase -- noise, dressed as a direction.
    """
    _, height, width = plan.v_n.shape
    with torch.no_grad():
        col = (points[:, 0] - plan.origin[0]) / plan.spacing[0]
        row = (points[:, 1] - plan.origin[1]) / plan.spacing[1]
        inside = ((col >= 0) & (col <= width - 1) & (row >= 0) & (row <= height - 1)).float()

        col_lo = col.floor().clamp(0, width - 2)
        row_lo = row.floor().clamp(0, height - 2)
        fx = (col - col_lo).clamp(0.0, 1.0)
        fy = (row - row_lo).clamp(0.0, 1.0)
        base = pair * (height * width) + row_lo.long() * width + col_lo.long()
        corners = (base, base + 1, base + width, base + width + 1)
        blend = (
            (1 - fx) * (1 - fy),
            fx * (1 - fy),
            (1 - fx) * fy,
            fx * fy,
        )

        def interpolate(field: torch.Tensor) -> torch.Tensor:
            flat = field.reshape(-1)
            return sum(w * flat[c] for w, c in zip(blend, corners, strict=True))

        weight = interpolate(plan.weight)
        usable = inside * (weight > 1.0 - VALID_TOL).float()
        return interpolate(plan.v_n).unsqueeze(1), usable.unsqueeze(1)


def _bin_index(
    plan: FrontVelocityPlan, front: FrontSamples, pair: torch.Tensor
) -> torch.Tensor:
    """Which profile bin each front sample falls in, ``(N,)``.

    One set of bins per (frame pair, segment): each time is its own measurement,
    and the four segments are kept apart because at the same ``u`` the upper and
    lower profiles are different points, with different normals -- averaging them
    together would confuse a widening bubble with a drifting one.

    Position within a segment is ``u`` along the body and the sweep angle on a
    cap, where ``u`` is pinned at 0 or 1 and so cannot locate anything. The two
    kinds of segment carry different sample counts, so they get their own bin
    counts and their own offsets into the flat index.
    """
    on_body = front.on_cap.reshape(-1) <= 0
    upper = front.side.reshape(-1) > 0
    at_root = front.u.reshape(-1) < 0.5

    # Segment starts, laid out [upper body | lower body | root cap | nose cap].
    body, cap = plan.bins_body, plan.bins_cap
    offset = torch.where(
        on_body,
        torch.where(upper, 0, body),
        torch.where(at_root, 2 * body, 2 * body + cap),
    )
    position = torch.where(
        on_body,
        front.u.reshape(-1),
        front.angle.reshape(-1) / math.pi + 0.5,
    ).detach()
    span = torch.where(on_body, body, cap)
    within = (position * span).long().clamp(torch.zeros_like(span), span - 1)
    return pair * plan.bins_per_time + offset + within


def _bin_sum(values: torch.Tensor, bins: torch.Tensor, n_bins: int) -> torch.Tensor:
    """``values`` summed into ``n_bins`` bins, differentiably."""
    return torch.zeros(n_bins, 1, device=values.device, dtype=values.dtype).index_add(
        0, bins, values
    )
