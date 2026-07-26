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
from dataclasses import dataclass

import numpy as np
import torch

from naviernet.utils.paths import RunPaths


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

        trainable = (self._ti != self.holdout_row) & (weights > 0)
        self._train_idx = np.where(trainable)[0]
        probabilities = weights[self._train_idx]
        self._train_p = probabilities / probabilities.sum()

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(n_frames, height_px, width_px)``."""
        return self.alpha.shape

    @property
    def event_frames(self) -> list[int]:
        """Camera frame numbers of the growth event, in row order."""
        return self.frame_numbers[: self.n_event]

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
