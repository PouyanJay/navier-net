"""Result figures.

``frames_matching``
    Measured versus predicted contours, one panel per camera frame, each
    labelled with its IoU and the holdout frame called out in red.

``continuous_dynamics``
    What the PINN buys over the raw frames: the interface at arbitrarily fine
    timesteps, with the ten camera instants overlaid as dotted contours.

``trajectories``
    Nose position and vapour area, continuous prediction against measurement.

``overlays_on_raw``
    The predicted contour drawn back onto the original camera images, in camera
    orientation -- the least mediated comparison available.
"""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from naviernet.evaluation import (
    measured_trajectory,
    nose_trajectory,
    predict_alpha,
    predict_alpha_fullres,
)
from naviernet.physics.groups import reference_time_ms
from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths

log = get_logger(__name__)

MEASURED_COLOUR = "k"
PREDICTED_COLOUR = "r"
HOLDOUT_COLOUR = "crimson"


def frames_matching(cfg, model, data, paths: RunPaths) -> None:
    """Per-frame measured (black) versus predicted (red dashed) contours."""
    stride = cfg.evaluation.stride
    threshold = cfg.evaluation.threshold
    xs, ys = data.x[::stride], data.y[::stride]
    n_event = cfg.experiment.n_frames_event
    holdout = int(cfg.training.holdout_frame)

    fig, axes = plt.subplots(n_event, 1, figsize=(11, 1.6 * n_event), sharex=True)
    for i in range(n_event):
        ax = axes[i]
        measured = data.alpha[i, ::stride, ::stride]
        predicted = predict_alpha(model, data, data.t[i], stride)

        intersection = (predicted > threshold) & (measured > threshold)
        union = (predicted > threshold) | (measured > threshold)
        iou = intersection.sum() / max(union.sum(), 1)

        ax.contourf(xs, ys, measured, [threshold, 1.1], colors=["#c6dbef"])
        ax.contour(xs, ys, measured, [threshold], colors=MEASURED_COLOUR, linewidths=1.6)
        ax.contour(
            xs,
            ys,
            predicted,
            [threshold],
            colors=PREDICTED_COLOUR,
            linewidths=1.4,
            linestyles="--",
        )

        is_holdout = i == holdout
        ax.text(
            0.01,
            0.72,
            f"frame {i + 1}   t = {i * cfg.experiment.dt_frame_ms:.1f} ms   "
            f"IoU = {iou:.3f}" + ("   *HOLDOUT*" if is_holdout else ""),
            transform=ax.transAxes,
            fontsize=9,
            color=HOLDOUT_COLOUR if is_holdout else "k",
        )
        ax.set_aspect("equal")

    axes[0].set_title("Measured (black) vs PINN (red dashed) — flow left to right")
    axes[-1].set_xlabel("x* (downstream)")
    fig.tight_layout()
    _save(fig, paths.figures_dir / "frames_matching.png", dpi=120)


def continuous_dynamics(cfg, model, data, paths: RunPaths, n_t: int = 37) -> None:
    """Continuous predicted interface, coloured by time, over the camera frames."""
    stride = cfg.evaluation.stride
    threshold = cfg.evaluation.threshold
    xs, ys = data.x[::stride], data.y[::stride]
    t_ref_ms = reference_time_ms(cfg.scales)

    fig, ax = plt.subplots(figsize=(12, 4.2))
    times = np.linspace(data.t[0], data.t[-2], n_t)
    cmap = plt.cm.viridis

    for k, t in enumerate(times):
        predicted = predict_alpha(model, data, t, stride)
        ax.contour(xs, ys, predicted, [threshold], colors=[cmap(k / n_t)], linewidths=1.0)

    for i in range(cfg.experiment.n_frames_event):
        ax.contour(
            xs,
            ys,
            data.alpha[i, ::stride, ::stride],
            [threshold],
            colors=MEASURED_COLOUR,
            linewidths=0.7,
            linestyles=":",
        )

    ax.axvline(data.domain.x_pin, color="gray", ls="--", lw=1)
    scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, times[-1] * t_ref_ms))
    plt.colorbar(scalar_map, ax=ax, label="t (ms)")
    ax.set_xlabel("x* (downstream, flow to the right)")
    ax.set_ylabel("y*")
    ax.set_title("PINN continuous interface (colour) vs measured frames (dotted)")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, paths.figures_dir / "continuous_dynamics.png")


def trajectories(cfg, model, data, paths: RunPaths) -> None:
    """Nose position and vapour area: continuous prediction vs measurement."""
    holdout = int(cfg.training.holdout_frame)
    t_ref_ms = reference_time_ms(cfg.scales)

    t_star, nose, area = nose_trajectory(cfg, model, data)
    t_ms = t_star * t_ref_ms
    meas_t, meas_nose, meas_area = measured_trajectory(cfg, data)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(t_ms, nose, "r-", label="PINN (continuous)")
    ax.plot(meas_t, meas_nose, "ko", label="measured")
    if holdout >= 0:
        ax.plot(
            meas_t[holdout],
            meas_nose[holdout],
            "s",
            ms=11,
            mfc="none",
            mec=HOLDOUT_COLOUR,
            label="holdout",
        )
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("nose x*")
    ax.set_title("Nose trajectory")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_ms, area, "r-", label="PINN")
    ax.plot(meas_t, meas_area, "ko", label="measured")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("vapour area A*")
    ax.set_title("Projected vapour area growth")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, paths.figures_dir / "trajectories.png")


def overlays_on_raw(cfg, model, data, paths: RunPaths) -> None:
    """Predicted contour drawn onto the raw camera frames, in camera orientation."""
    threshold = cfg.evaluation.threshold
    y0, y1 = data.meta["y_roi"]
    holdout = int(cfg.training.holdout_frame)

    tiles = []
    for i in range(cfg.experiment.n_frames_event):
        raw = np.asarray(Image.open(paths.raw_frame(i + 1)).convert("RGB"))
        # Un-flip x to return from downstream-positive to camera orientation.
        predicted = predict_alpha_fullres(model, data, data.t[i])[:, ::-1]
        mask = (predicted > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        canvas = raw.copy()
        cv2.drawContours(canvas[y0:y1, :], contours, -1, (255, 30, 30), 4)
        Image.fromarray(canvas[y0 - 60 : y1 + 60, :]).save(
            paths.figures_dir / f"pinn_on_image_f{i + 1:02d}.png"
        )

        tile = canvas[y0 - 52 : y1 + 18, :]
        is_holdout = i == holdout
        # ASCII only: cv2 renders with Hershey fonts, which have no glyphs beyond it.
        label = f"frame {i + 1}  t={i * cfg.experiment.dt_frame_ms:.1f} ms" + (
            "   [HOLDOUT - never seen]" if is_holdout else ""
        )
        cv2.putText(
            tile,
            label,
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 40, 40) if is_holdout else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    contact_sheet = paths.figures_dir / "pinn_on_images_all.png"
    Image.fromarray(np.vstack(tiles)).save(contact_sheet)
    log.info("wrote %s", contact_sheet)


def render_all_figures(cfg, model, data, paths: RunPaths) -> None:
    """Every result figure, into the run's ``figures/`` directory."""
    paths.ensure()
    frames_matching(cfg, model, data, paths)
    continuous_dynamics(cfg, model, data, paths)
    trajectories(cfg, model, data, paths)
    overlays_on_raw(cfg, model, data, paths)
    log.info("figures written to %s", paths.figures_dir)


def _save(fig, destination, dpi: int = 130) -> None:
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)
    log.info("wrote %s", destination)
