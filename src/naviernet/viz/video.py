"""Slow-motion video of the continuous PINN reconstruction.

The camera recorded ten frames 0.5 ms apart. The network is continuous in time,
so it can be sampled as finely as we like between them -- that is what this
renders, at roughly a thousand times slower than real time.

Pipeline: build a clean background plate (per-pixel maximum across the raw
frames, then inpaint the residual dark structure), evaluate alpha at each fine
timestep, composite with a cool vapour tint and a layered interface glow, flash
the measured contour whenever the clock passes a real camera instant, overlay a
HUD, and encode with ffmpeg.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from naviernet.evaluation import predict_alpha_fullres
from naviernet.physics.groups import reference_time_ms
from naviernet.utils.logging import get_logger
from naviernet.utils.paths import RunPaths

log = get_logger(__name__)

# Composite styling
VAPOUR_TINT = np.array([70, 80, 105], np.float32)
VAPOUR_OPACITY = 0.82
# Concentric contour passes, outermost first: width in px -> BGR-ish colour.
GLOW_LAYERS = (
    (13, (60, 20, 10)),
    (7, (160, 60, 20)),
    (3, (255, 140, 40)),
    (1, (255, 230, 180)),
)
HUD_BACKGROUND = (12, 12, 16)
FLASH_TOLERANCE_MS = 0.10  # how close the clock must be to a camera instant


def background_plate(cfg, paths: RunPaths) -> np.ndarray:
    """Bubble-free background: per-pixel max across frames, then inpainted.

    The bubble is dark and moves, so the brightest value each pixel takes across
    the sequence is almost always bubble-free. Whatever dark structure survives
    (the heater traces, which never move) is inpainted away.
    """
    frames = [
        np.asarray(Image.open(paths.raw_frame(n)).convert("RGB"))
        for n in range(1, cfg.experiment.n_frames_raw + 1)
    ]
    plate = np.max(np.stack(frames), axis=0)

    grey = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
    holes = cv2.dilate(
        ((grey < cfg.video.background_dark_thresh).astype(np.uint8)) * 255,
        np.ones((5, 5), np.uint8),
    )
    return cv2.inpaint(plate, holes, 7, cv2.INPAINT_TELEA)


def _bubble_length_range(data, um_per_px: float) -> tuple[float, float]:
    """Length bounds for the HUD progress bar, taken from the measured masks."""
    lengths = [
        np.ptp(np.nonzero(mask.any(axis=0))[0]) * um_per_px
        for mask in data.masks_camera
        if mask.any()
    ]
    return float(min(lengths)), float(max(lengths))


def render_video(cfg, model, data, paths: RunPaths, n_t: int | None = None) -> Path:
    """Render the MP4. Returns the path written."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; it is required for the video stage "
            "(macOS: `brew install ffmpeg`, Debian: `apt install ffmpeg`)"
        )

    vcfg = cfg.video
    n_t = int(n_t if n_t is not None else vcfg.n_timesteps)
    fps = int(vcfg.fps)
    threshold = cfg.evaluation.threshold

    paths.ensure()
    plate = background_plate(cfg, paths)
    y0, y1 = data.meta["y_roi"]
    um_per_px = data.meta["um_per_px"]
    t_ref_ms = reference_time_ms(cfg.scales)
    dt_frame_ms = cfg.experiment.dt_frame_ms
    n_event = cfg.experiment.n_frames_event

    times = np.linspace(data.t[0], data.t[-2], n_t)
    # Real time between rendered frames, versus the wall-clock time each frame
    # is displayed for: their ratio is the slow-motion factor quoted in the HUD.
    dt_real_s = (times[1] - times[0]) * t_ref_ms * 1e-3
    slowdown = int(round(1.0 / (dt_real_s * fps)))

    crop_top, crop_bottom = y0 - 110, y1 + 150
    view_height = crop_bottom - crop_top
    length_min, length_max = _bubble_length_range(data, um_per_px)

    frames_dir = paths.video_frames_dir
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    log.info("rendering %d frames at %dx slow motion", n_t, slowdown)
    frame = None
    for i, t in enumerate(times):
        # Un-flip x: the video is shown in camera orientation.
        alpha_cam = predict_alpha_fullres(model, data, t)[:, ::-1]
        t_ms = t * t_ref_ms
        mask = alpha_cam > threshold

        canvas = plate.copy()

        # Tint the vapour, fading in across the diffuse interface.
        fade = np.clip((alpha_cam - 0.35) / 0.3, 0, 1)[..., None]
        region = canvas[y0:y1, :].astype(np.float32)
        region = region * (1 - VAPOUR_OPACITY * fade) + VAPOUR_TINT * VAPOUR_OPACITY * fade
        canvas[y0:y1, :] = region.astype(np.uint8)

        # Layered interface glow, drawn widest-and-dimmest first.
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        glow = np.zeros_like(canvas[y0:y1, :])
        for width_px, colour in GLOW_LAYERS:
            cv2.drawContours(glow, contours, -1, colour, width_px)
        canvas[y0:y1, :] = cv2.add(canvas[y0:y1, :], cv2.GaussianBlur(glow, (0, 0), 2))

        # Flash the measured contour as the clock passes each camera instant.
        k = round(t_ms / dt_frame_ms)
        if k < n_event and abs(t_ms - k * dt_frame_ms) < FLASH_TOLERANCE_MS:
            measured, _ = cv2.findContours(
                data.masks_camera[k], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(canvas[y0:y1, :], measured, -1, (255, 255, 255), 1)

        frame = canvas[crop_top:crop_bottom, :].copy()
        columns = np.where(mask.any(axis=0))[0]
        length_um = (columns.max() - columns.min()) * um_per_px if len(columns) else 0.0
        _draw_hud(cfg, frame, t_ms, slowdown, length_um, (length_min, length_max), view_height)

        Image.fromarray(frame).save(frames_dir / f"f{i:04d}.png")

    # Hold the last frame briefly so the video does not cut dead on the end.
    for j in range(1, vcfg.hold_frames + 1):
        Image.fromarray(frame).save(frames_dir / f"f{n_t - 1 + j:04d}.png")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "f%04d.png"),
            "-vf",
            f"scale={vcfg.width}:-2:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            str(paths.video),
        ],
        check=True,
    )
    shutil.rmtree(frames_dir)
    log.info("video written to %s", paths.video)
    return paths.video


def _draw_hud(cfg, frame, t_ms, slowdown, length_um, length_range, view_height) -> None:
    """Title bar, clock, and the bubble-length readout with a progress bar."""
    width = frame.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    flow_arrow = "<--" if cfg.experiment.flow_direction == "right_to_left" else "-->"

    cv2.rectangle(frame, (0, 0), (width, 64), HUD_BACKGROUND, -1)
    cv2.putText(
        frame,
        # ASCII only: cv2 renders with Hershey fonts, which have no glyphs beyond it.
        f"{cfg.experiment.fluid} confined bubble growth - PINN reconstruction",
        (14, 26),
        font,
        0.62,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"t = {t_ms:5.2f} ms    {slowdown}x slow motion    flow {flow_arrow}",
        (14, 52),
        font,
        0.62,
        (120, 200, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(frame, (0, view_height - 46), (width, view_height), HUD_BACKGROUND, -1)
    cv2.putText(
        frame,
        f"bubble length = {length_um:4.0f} um",
        (14, view_height - 16),
        font,
        0.62,
        (255, 180, 80),
        1,
        cv2.LINE_AA,
    )

    bar_x0, bar_x1 = 330, width - 24
    cv2.rectangle(
        frame, (bar_x0, view_height - 30), (bar_x1, view_height - 16), (60, 60, 70), 1
    )
    low, high = length_range
    progress = np.clip((length_um - low) / max(high - low, 1e-9), 0, 1)
    fill = int(progress * (bar_x1 - bar_x0 - 2))
    cv2.rectangle(
        frame,
        (bar_x0 + 1, view_height - 29),
        (bar_x0 + 1 + fill, view_height - 17),
        (255, 140, 40),
        -1,
    )
