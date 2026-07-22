"""Quality-control figure for the preprocessing stage.

This is the figure to look at before trusting anything downstream: if wall
detection or segmentation went wrong, it shows up here as a kinked growth
curve, crossing interface contours, or a signed-distance field with the wrong
sign inside the bubble.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from naviernet.physics.groups import reference_time_ms
from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths

log = get_logger(__name__)


def qc_figure(cfg, paths: RunPaths, alpha, sdf, xs, ys, ts, um_per_px, x_pin) -> None:
    """Growth kinematics, interface evolution, and an example SDF."""
    n_event = cfg.experiment.n_frames_event
    t_ms = ts * reference_time_ms(cfg.scales)

    # Bubble length in um: streamwise extent of the mask, frame by frame.
    lengths_um = np.array(
        [np.ptp(np.nonzero((frame > 0.5).any(axis=0))[0]) * um_per_px for frame in alpha]
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    ax = axes[0]
    ax.plot(t_ms[:n_event], lengths_um[:n_event], "o-", label="measured")
    fit = np.polyfit(t_ms[:n_event], lengths_um[:n_event], 1)
    ax.plot(t_ms, np.polyval(fit, t_ms), "--", label=f"fit dL/dt = {fit[0]:.0f} mm/s")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("L (um)")
    ax.set_title("Growth kinematics")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i in range(0, len(ts), 2):
        ax.contour(xs, ys, alpha[i], [0.5], linewidths=1.2)
    ax.axvline(x_pin, color="k", ls=":", label="pinned cavity")
    ax.set_aspect("equal")
    ax.set_xlabel("x* (downstream)")
    ax.set_ylabel("y*")
    ax.set_title("Interface evolution")

    ax = axes[2]
    image = ax.imshow(
        sdf[len(ts) // 2],
        extent=[xs[0], xs[-1], ys[-1], ys[0]],
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    plt.colorbar(image, ax=ax, label="signed distance (negative inside vapour)")
    ax.set_title("Signed distance field (mid frame)")

    fig.tight_layout()
    fig.savefig(paths.qc_figure, dpi=130)
    plt.close(fig)
    log.info("QC figure written to %s", paths.qc_figure)
