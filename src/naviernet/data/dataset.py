"""Dataset: loads the preprocessed tensors and samples training points.

Two choices here carry most of the weight:

**Supervision targets are smoothed, not binary.** The network is fit against
``sigmoid(-sdf / eps)`` rather than the raw 0/1 mask. Fitting a
smeared-but-controlled profile is far easier than fitting a step, and the
half-thickness ``eps`` is ours to anneal later.

**Sampling is interface-weighted.** Points are drawn with probability peaking at
the interface, so supervision and collocation both concentrate where the physics
actually happens instead of being wasted on uniform bulk liquid.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

from naviernet.utils.paths import RunPaths

# The alpha level defining the interface itself: alpha = sigmoid(phi/eps) is 0.5
# exactly on phi's zero contour. This is a definition, NOT the tunable IoU
# threshold in cfg.evaluation.threshold -- do not wire it to config.
INTERFACE_ALPHA = 0.5


def mask_x_extent(mask: np.ndarray) -> tuple[int, int] | None:
    """The first and last column of a 2-D boolean vapour mask, or ``None`` when
    the mask is empty. The single convention every root/edge measurement uses."""
    cols = np.nonzero(mask.any(axis=0))[0]
    if cols.size == 0:
        return None
    return int(cols[0]), int(cols[-1])


@dataclass(frozen=True)
class Domain:
    """Space-time bounds, derived from the tensors rather than assumed."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    t_min: float
    t_max: float
    x_pin: float  # streamwise station of the pinned nucleation cavity

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


class BubbleDataset:
    """Preprocessed tensors plus the samplers the trainer draws from."""

    def __init__(self, cfg, paths: RunPaths, device: str = "cpu"):
        if not paths.tensors.exists():
            raise FileNotFoundError(
                f"{paths.tensors} not found -- run the preprocess stage first:\n"
                f"  naviernet stage=preprocess dataset={cfg.dataset}"
            )

        archive = np.load(paths.tensors)
        self.alpha = archive["alpha"]  # [T, H, W]
        self.sdf = archive["sdf"]
        self.valid = archive["valid"]
        self.masks_camera = archive["masks_camera"]
        self.x = archive["x_star"]
        self.y = archive["y_star"]
        self.t = archive["t_star"]
        self.meta = json.loads(str(archive["meta"]))

        self.cfg = cfg
        self.device = device
        self.eps = float(cfg.model.alpha_eps)

        n_rows = self.alpha.shape[0]
        # Row -> 1-based camera frame. Identity unless the series excludes
        # frames; archives written before exclusions existed have no such key.
        self.frame_numbers: list[int] = [
            int(n) for n in self.meta.get("frame_numbers", range(1, n_rows + 1))
        ]
        # Rows of the growth event -- the prefix every per-frame stage iterates.
        self.n_event = int(self.meta.get("n_frames_event", n_rows))
        # `training.holdout_frame` is the 0-based position in the unexcluded
        # sequence (camera frame - 1). Resolve it to a row, so that excluding a
        # frame can never quietly shift supervision onto the holdout. -1 (train
        # on all frames) and an excluded holdout both resolve to "no row".
        holdout_camera = int(cfg.training.holdout_frame) + 1
        self.holdout_row = (
            self.frame_numbers.index(holdout_camera)
            if holdout_camera in self.frame_numbers
            else -1
        )

        # Validation axis A: a fraction of the event's frames, held out of
        # supervision as an in-distribution validation set (see `_split_rows`).
        # Excluded frames are already gone from the archive, so this is taken over
        # the rows that remain. Composes with `holdout_row`: the held-out set below
        # is their union, so a frame counted twice is harmless.
        self.split_rows: list[int] = self._split_rows(
            float(cfg.training.val_fraction), str(cfg.training.val_strategy)
        )

        self.domain = Domain(
            x_min=float(self.x[0]),
            x_max=float(self.x[-1]),
            y_min=float(self.y[0]),
            y_max=float(self.y[-1]),
            t_min=float(self.t[0]),
            t_max=float(self.t[-1]),
            x_pin=float(self.meta["x_pin_star"]),
        )

        n_t, n_y, n_x = self.alpha.shape
        ti, yi, xi = np.meshgrid(np.arange(n_t), np.arange(n_y), np.arange(n_x), indexing="ij")
        self._ti, self._yi, self._xi = ti.ravel(), yi.ravel(), xi.ravel()

        # Sampling weight: a Gaussian bump on the interface plus a small floor
        # so the bulk is not starved entirely. Invalid pixels get zero.
        weights = np.exp(-((self.sdf / (4 * self.eps)) ** 2)) + 0.02
        weights = (weights * self.valid).ravel()

        # Every row held out of supervision: the split (axis A) and the legacy
        # single frame, composed. This union is the honest in-distribution
        # validation set -- the frames whose IoU is never a memorisation statement.
        held_out = set(self.split_rows)
        if self.holdout_row >= 0:
            held_out.add(self.holdout_row)
        self.validation_rows: list[int] = sorted(held_out)
        trainable = ~np.isin(self._ti, self.validation_rows) & (weights > 0)
        self._train_idx = np.where(trainable)[0]
        probabilities = weights[self._train_idx]
        self._train_p = probabilities / probabilities.sum()

    def _split_rows(self, fraction: float, strategy: str) -> list[int]:
        """Rows of the event held out as the axis-A validation split.

        ``fraction`` of the ``n_event`` growth frames, chosen by ``strategy``:
        ``tail`` holds the last frames (extrapolation), ``scatter`` holds interior
        evenly-spaced frames (interpolation). A positive fraction always holds at
        least one frame, and never the whole event -- at least one training frame
        survives, so an over-large fraction cannot starve supervision.
        """
        if fraction <= 0.0 or self.n_event < 2:
            return []

        count = max(1, math.ceil(fraction * self.n_event))

        if strategy == "tail":
            count = min(count, self.n_event - 1)
            return list(range(self.n_event - count, self.n_event))
        if strategy == "scatter":
            # Choose from the interior rows only, so both endpoints stay in
            # training and the model genuinely interpolates. There are exactly
            # ``n_event - 2`` interior rows, so cap the count at that (a series too
            # short to hold an interior frame gets no scatter split) and pick
            # evenly-spaced *distinct* rows -- never collapsing onto one another.
            interior = list(range(1, self.n_event - 1))
            count = min(count, len(interior))
            if count == 0:
                return []
            picks = np.linspace(0, len(interior) - 1, count)
            return sorted({interior[int(round(p))] for p in picks})
        raise ValueError(f"unknown val_strategy: {strategy!r} (want 'tail' or 'scatter')")

    @property
    def split_frames(self) -> list[int]:
        """Camera frame numbers of the axis-A validation-split rows only."""
        return [self.frame_numbers[row] for row in self.split_rows]

    @property
    def validation_frames(self) -> list[int]:
        """Camera frame numbers of every frame held out of supervision -- the
        validation split and the legacy holdout frame, composed. This is the set
        the in-distribution validation IoU is scored over."""
        return [self.frame_numbers[row] for row in self.validation_rows]

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(n_frames, height_px, width_px)``."""
        return self.alpha.shape

    @property
    def groups(self) -> dict[str, float] | None:
        """The dataset's dimensionless groups, recorded in its tensors at
        preprocess time. ``None`` for archives written before that existed (they
        must be re-preprocessed to join a multi-dataset run)."""
        return self.meta.get("groups")

    @property
    def event_frames(self) -> list[int]:
        """Camera frame numbers of the growth event, in row order."""
        return self.frame_numbers[: self.n_event]

    @cached_property
    def pin_anchor(self) -> tuple[float, float]:
        """The bubble-root anchor ``(x*, y*)``: where the interface stays for all t.

        Measured from the data, not assumed: over the *training-visible* event
        frames (held-out rows excluded, so the anchor can never leak validation
        information), the bubble's two x-extent edges are tracked and the
        temporally stationary one -- the nucleation-side root -- is taken. The
        anchor is that edge's median x* and the median centre of its vapour
        column in y*. Orientation-agnostic: whichever edge is stationary wins,
        so a flipped series needs no special-casing.
        """
        rows = [r for r in range(self.n_event) if r not in set(self.validation_rows)]
        col = self._stationary_root_column(rows)
        return float(self.x[col]), self._root_y_center(rows, col)

    def _vapor(self, row: int) -> np.ndarray:
        """The row's valid vapour mask (see :data:`INTERFACE_ALPHA`)."""
        return (self.alpha[row] > INTERFACE_ALPHA) & (self.valid[row] > 0)

    @cached_property
    def supervised_growth_rate(self) -> float:
        """Measured bubble-area growth rate over the last two *training-visible*
        event frames, in area* per t*.

        The kinematic growth constraints normalize by this reference so their
        terms are O(1). Data-visible only: held-out rows never enter, so the
        reference can leak nothing about the extrapolation window.
        """
        rows = [r for r in range(self.n_event) if r not in set(self.validation_rows)]
        if len(rows) < 2:
            raise ValueError(
                "training.kinematics needs a growth-rate reference, but fewer than "
                "two event frames are training-visible."
            )
        first, last = rows[-2], rows[-1]
        area = [float(self._vapor(r).mean()) * self.domain.area for r in (first, last)]
        rate = (area[1] - area[0]) / (float(self.t[last]) - float(self.t[first]))
        if rate <= 0:
            raise ValueError(
                f"training.kinematics needs a growing supervised tail, but the measured "
                f"rate over frames {self.frame_numbers[first]}->{self.frame_numbers[last]} "
                f"is {rate:.4g} (<= 0)."
            )
        return rate

    def _stationary_root_column(self, rows: list[int]) -> int:
        """The median column of the temporally stationary bubble edge -- of the two
        x-extent edges across ``rows``, the one with the smaller temporal spread
        (ties keep the low edge, deterministically)."""
        lo_cols, hi_cols = [], []
        for r in rows:
            extent = mask_x_extent(self._vapor(r))
            if extent is None:
                continue
            lo_cols.append(extent[0])
            hi_cols.append(extent[1])
        if not lo_cols:
            raise ValueError(
                "model.hard_pin needs a bubble-root anchor, but no training frame "
                "has any vapour (alpha > 0.5) -- check the dataset's masks."
            )
        root_cols = lo_cols if np.std(lo_cols) <= np.std(hi_cols) else hi_cols
        return int(round(float(np.median(root_cols))))

    def _root_y_center(self, rows: list[int], col: int) -> float:
        """The median (over ``rows``) centre y* of the vapour column at ``col``."""
        centers = []
        for r in rows:
            vapor_rows = np.nonzero(self._vapor(r)[:, col])[0]
            if vapor_rows.size:
                centers.append(float(self.y[vapor_rows].mean()))
        if not centers:
            raise ValueError(
                "model.hard_pin found a stationary bubble edge but no vapour column "
                f"at its median station (column {col}) -- the masks look degenerate."
            )
        return float(np.median(centers))

    def _coords(self, idx: np.ndarray) -> np.ndarray:
        """Map flat tensor indices to ``(x, y, t)`` coordinates."""
        return np.stack(
            [self.x[self._xi[idx]], self.y[self._yi[idx]], self.t[self._ti[idx]]],
            axis=1,
        ).astype(np.float32)

    def sample_supervised(self, n: int, rng) -> tuple[torch.Tensor, torch.Tensor]:
        """Interface-weighted supervised points drawn from the training frames."""
        idx = self._train_idx[rng.choice(len(self._train_idx), n, p=self._train_p)]
        coords = self._coords(idx)
        target = 1.0 / (1.0 + np.exp(self.sdf.ravel()[idx] / self.eps))
        return (
            torch.tensor(coords, device=self.device),
            torch.tensor(target[:, None].astype(np.float32), device=self.device),
        )

    def sample_collocation(self, n: int, rng) -> torch.Tensor:
        """PDE points: half uniform over the domain, half jittered off the interface."""
        d = self.domain
        n_uniform = n // 2
        uniform = np.stack(
            [
                rng.uniform(d.x_min, d.x_max, n_uniform),
                rng.uniform(d.y_min, d.y_max, n_uniform),
                rng.uniform(d.t_min, d.t_max, n_uniform),
            ],
            axis=1,
        ).astype(np.float32)

        idx = self._train_idx[rng.choice(len(self._train_idx), n - n_uniform, p=self._train_p)]
        # Jitter off the pixel grid so residuals are not evaluated only where
        # supervision already pins the solution.
        near_interface = self._coords(idx)
        near_interface += rng.normal(0, 0.01, near_interface.shape).astype(np.float32)

        points = np.concatenate([uniform, near_interface], axis=0)
        return torch.tensor(points, device=self.device, requires_grad=True)

    def sample_boundary(self, n: int, rng) -> tuple[torch.Tensor, torch.Tensor]:
        """Inlet points (``x*=0``) and side-wall points (``y*=y_min`` or ``y_max``)."""
        d = self.domain
        t = rng.uniform(d.t_min, d.t_max, n).astype(np.float32)

        inlet = np.stack(
            [np.zeros(n, np.float32), rng.uniform(d.y_min, d.y_max, n).astype(np.float32), t],
            axis=1,
        )
        wall_y = np.where(rng.random(n) < 0.5, d.y_min, d.y_max).astype(np.float32)
        walls = np.stack(
            [rng.uniform(d.x_min, d.x_max, n).astype(np.float32), wall_y, t], axis=1
        )
        return (
            torch.tensor(inlet, device=self.device),
            torch.tensor(walls, device=self.device),
        )

    def frame_grid(self, frame: int, stride: int = 2):
        """Full pixel grid of one frame: ``(points, ground_truth, grid_shape)``."""
        _, height, width = self.alpha.shape
        yy, xx = np.meshgrid(
            np.arange(0, height, stride), np.arange(0, width, stride), indexing="ij"
        )
        points = np.stack(
            [
                self.x[xx.ravel()],
                self.y[yy.ravel()],
                np.full(xx.size, self.t[frame]),
            ],
            axis=1,
        ).astype(np.float32)
        truth = self.alpha[frame, ::stride, ::stride]
        return torch.tensor(points, device=self.device), truth, yy.shape
