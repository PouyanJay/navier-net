"""Figures, quality-control plots, and video rendering.

Matplotlib is switched to the non-interactive Agg backend here, before any
submodule imports pyplot, so rendering works identically over SSH, in CI, and
inside a Hydra multirun with no display attached.
"""

import matplotlib

matplotlib.use("Agg")

from naviernet.viz.figures import (  # noqa: E402
    continuous_dynamics,
    frames_matching,
    overlays_on_raw,
    render_all_figures,
    trajectories,
)
from naviernet.viz.qc import qc_figure  # noqa: E402
from naviernet.viz.video import render_video  # noqa: E402

__all__ = [
    "continuous_dynamics",
    "frames_matching",
    "overlays_on_raw",
    "qc_figure",
    "render_all_figures",
    "render_video",
    "trajectories",
]
