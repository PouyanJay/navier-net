"""Filesystem layout for a single run.

Two directories, deliberately kept apart:

``data/processed/<dataset>/``
    Preprocessed tensors. Keyed by *dataset*, because segmentation depends only
    on the raw frames and the imaging settings -- many runs share one copy.

``outputs/<run_name>/``
    Everything a particular run produced: checkpoints, metrics, figures, video,
    and Hydra's snapshot of the config that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Resolved, absolute-or-relative paths for one run."""

    raw_dir: Path
    processed_dir: Path
    output_dir: Path

    @classmethod
    def from_config(cls, cfg) -> RunPaths:
        return cls(
            raw_dir=Path(cfg.paths.raw_dir),
            processed_dir=Path(cfg.paths.processed_dir),
            output_dir=Path(cfg.paths.output_dir),
        )

    # -- dataset-scoped ----------------------------------------------------
    def raw_frame(self, n: int) -> Path:
        """Path to the n-th raw TIFF (1-based, matching the camera numbering)."""
        return self.raw_dir / f"{n}.tif"

    @property
    def tensors(self) -> Path:
        return self.processed_dir / "tensors.npz"

    @property
    def qc_figure(self) -> Path:
        return self.processed_dir / "qc_preprocess.png"

    # -- run-scoped --------------------------------------------------------
    @property
    def checkpoints_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def checkpoint(self) -> Path:
        return self.checkpoints_dir / "ckpt.pt"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def video_dir(self) -> Path:
        return self.output_dir / "video"

    @property
    def video(self) -> Path:
        return self.video_dir / "growth.mp4"

    @property
    def video_frames_dir(self) -> Path:
        return self.video_dir / "frames"

    @property
    def groups_json(self) -> Path:
        return self.output_dir / "dimensionless_groups.json"

    @property
    def metrics_json(self) -> Path:
        return self.output_dir / "metrics.json"

    @property
    def trajectory_json(self) -> Path:
        return self.output_dir / "trajectory.json"

    def ensure(self) -> RunPaths:
        """Create every writable directory. Safe to call repeatedly."""
        for d in (
            self.processed_dir,
            self.output_dir,
            self.checkpoints_dir,
            self.figures_dir,
            self.video_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self
