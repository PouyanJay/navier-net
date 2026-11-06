"""Ghost frames: an invented history of the bubble before the first image.

The images start with the bubble already 0.57 mm long, so a solve that starts
there has to be told the state at frame 1: how long each part of the floor has
been covered, the cooled layer around the bubble, the flow's memory. Ghost
frames move the start back to an instant where the state is simple (a small
bubble on the cavity, undisturbed flow, the heaters' steady temperature) and let
the solver compute frame 1 (``experiment.ghost_duration_ms``).

The history is a rule, not a measurement:

- **Growth.** The area is the parabola through the seed's area at the start
  that joins frame 1 with its area and with the growth rate the first two
  images show. It is the smoothest growth that does so.
- **Shape** (``experiment.ghost_shape``). ``similar``: the first image's
  outline shrunk about its root apex, the same shape at every size. Its nose
  then moves at less than half the speed the images start with, because the
  real bubble grows by lengthening. ``stretch``: the same outline scaled
  separately along and across the channel; it starts as the ``similar`` seed,
  widens to the first image's width, and by the first image only lengthens, so
  the nose's speed joins the images' too. ``blob``: a round blob on the cavity
  that grows until it is as wide as the first image's bubble, after which its
  nose advances at that width, the outline blending station by station from
  the blob into the first image's.

The seed is the first ghost frame; under every shape rule it has the area of
a disc ``experiment.ghost_seed_um`` across, so the rules share one growth.

Ghost frames are one camera interval apart and numbered backwards from the
first image: frame 0 is one interval before it, frame -1 two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path

from naviernet.data.outline import outline_path
from naviernet.data.preprocess import signed_distance
from naviernet.utils.paths import RunPaths

# ``experiment.ghost_shape``.
GHOST_SHAPES = ("similar", "stretch", "blob")
# Bisection steps for the blend that gives an area: 2^-40 of the blend's range.
BLEND_STEPS = 40


@dataclass(frozen=True)
class GhostOutline:
    """The ghost frames' instants (``t_ref``), ascending and ending one camera
    interval before the first image, and their closed outlines ``(n, M, 2)`` in
    the measured outline's station layout."""

    times: np.ndarray
    stations: np.ndarray


def ghost_frame_count(experiment) -> int:
    """How many ghost frames ``experiment.ghost_duration_ms`` asks for: the
    duration in camera intervals, rounded; zero when the ghost start is off."""
    duration = float(getattr(experiment, "ghost_duration_ms", 0.0) or 0.0)
    if duration < 0.0:
        raise ValueError(f"experiment.ghost_duration_ms={duration} must be >= 0")
    return int(round(duration / float(experiment.dt_frame_ms)))


def polygon_area(curve: np.ndarray) -> float:
    """The area a closed curve ``(M, 2)`` encloses (shoelace)."""
    x, y = curve[:, 0], curve[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def growth_areas(
    elapsed: np.ndarray, first_area: float, first_rate: float, seed_area: float
) -> np.ndarray:
    """The ghost frames' areas: the parabola in time that is ``seed_area`` at
    the earliest instant and reaches ``first_area`` at rate ``first_rate`` at
    the first image. ``elapsed`` is each instant's time before the first image
    (positive, the earliest first)."""
    duration = float(elapsed[0])
    longest = 2.0 * (first_area - seed_area) / first_rate
    if seed_area >= first_area or first_rate <= 0.0 or duration > longest:
        raise ValueError(
            "the ghost growth must rise from the seed to the first image: a seed of area "
            f"{seed_area:.4g} reaching {first_area:.4g} at rate {first_rate:.4g} allows at "
            f"most {longest:.4g} of ghost time, got {duration:.4g} (L_ref and t_ref units)"
        )
    bend = (seed_area - first_area + first_rate * duration) / duration**2
    return first_area - first_rate * elapsed + bend * elapsed**2


def _round_blob(first: np.ndarray, axis: float, radius: float) -> np.ndarray:
    """A circle of ``radius`` whose rear sits on the first outline's root apex,
    in that outline's station layout: each station at the same share of the way
    round as on the first outline."""
    half = len(first) // 2
    steps = np.hypot(*np.diff(first[: half + 1], axis=0).T)
    share = np.r_[0.0, np.cumsum(steps)] / steps.sum()
    angle = np.pi * share
    x = first[half, 0] + radius * (1.0 + np.cos(angle))
    upper = np.column_stack([x, axis + radius * np.sin(angle)])
    lower = upper[-2:0:-1] * np.array([1.0, -1.0]) + np.array([0.0, 2.0 * axis])
    return np.concatenate([upper, lower])


def _blend_for_area(blob: np.ndarray, first: np.ndarray, area: float) -> float:
    """The share of the first outline in the blob-to-outline blend whose area is
    ``area`` (the area rises with the share)."""
    low, high = 0.0, 1.0
    for _ in range(BLEND_STEPS):
        middle = 0.5 * (low + high)
        if polygon_area((1.0 - middle) * blob + middle * first) < area:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _similar_shapes(first: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """The first outline shrunk about its root apex to each area."""
    root = first[len(first) // 2]
    scales = np.sqrt(areas / polygon_area(first))
    return root + scales[:, None, None] * (first - root)[None]


def _stretch_shapes(first: np.ndarray, areas: np.ndarray, progress: np.ndarray) -> np.ndarray:
    """The first outline scaled about its root apex, separately along and
    across the channel. The width goes from the similar seed's to the first
    outline's as ``progress`` (the share of the ghost time gone by) goes from 0
    to 1; the length follows from each area."""
    root = first[len(first) // 2]
    seed = np.sqrt(areas[0] / polygon_area(first))
    # A smooth step: the width is level at both ends, so the seed starts by
    # lengthening and the last ghost frames only lengthen.
    across = seed + (1.0 - seed) * progress**2 * (3.0 - 2.0 * progress)
    along = areas / (polygon_area(first) * across)
    scale = np.stack([along, across], axis=1)
    return root + scale[:, None, :] * (first - root)[None]


def _blob_shapes(first: np.ndarray, axis: float, areas: np.ndarray) -> np.ndarray:
    """A round blob on the cavity while the area is below that of a circle as
    wide as the first outline, the blend of that circle into the first outline
    above it."""
    width = 2.0 * float(np.abs(first[:, 1] - axis).max())
    widest_blob = _round_blob(first, axis, 0.5 * width)
    blob_area = polygon_area(widest_blob)
    shapes = []
    for area in areas:
        if area <= blob_area:
            shapes.append(_round_blob(first, axis, 0.5 * width * np.sqrt(area / blob_area)))
        else:
            share = _blend_for_area(widest_blob, first, float(area))
            shapes.append((1.0 - share) * widest_blob + share * first)
    return np.stack(shapes)


def ghost_shapes(
    first: np.ndarray, axis: float, areas: np.ndarray, progress: np.ndarray, rule: str
) -> np.ndarray:
    """One closed outline per area, ``(n, M, 2)``, by the shape ``rule``;
    ``progress`` is each frame's share of the ghost time gone by."""
    if rule == "similar":
        return _similar_shapes(first, areas)
    if rule == "stretch":
        return _stretch_shapes(first, areas, progress)
    if rule == "blob":
        return _blob_shapes(first, axis, areas)
    raise ValueError(f"experiment.ghost_shape must be one of {GHOST_SHAPES}, got {rule!r}")


def ghost_outline(
    experiment, l_ref_um: float, times: np.ndarray, stations: np.ndarray, axis: float
) -> GhostOutline | None:
    """The ghost frames before a measured outline (its ``times`` and
    ``stations``), or ``None`` when the ghost start is off."""
    count = ghost_frame_count(experiment)
    if count == 0:
        return None
    times = np.asarray(times, dtype=np.float64)
    stations = np.asarray(stations, dtype=np.float64)
    interval = float(times[1] - times[0])
    elapsed = interval * np.arange(count, 0, -1)
    first_area = polygon_area(stations[0])
    first_rate = (polygon_area(stations[1]) - first_area) / interval
    seed_diameter = float(experiment.ghost_seed_um) / float(l_ref_um)
    areas = growth_areas(elapsed, first_area, first_rate, 0.25 * np.pi * seed_diameter**2)
    rule = str(getattr(experiment, "ghost_shape", "similar"))
    return GhostOutline(
        times=times[0] - elapsed,
        stations=ghost_shapes(stations[0], axis, areas, 1.0 - elapsed / elapsed[0], rule),
    )


def load_ghost_outline(cfg, paths: RunPaths) -> GhostOutline | None:
    """The ghost frames of this run's dataset, built from the outline the
    preprocess stage traced; ``None`` when the ghost start is off."""
    if ghost_frame_count(cfg.experiment) == 0:
        return None
    path = outline_path(paths)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: experiment.ghost_duration_ms grows the ghost frames from "
            f"the traced outline, so re-run the preprocess stage "
            f"(naviernet stage=preprocess dataset={cfg.dataset})"
        )
    with np.load(path) as archive:
        return ghost_outline(
            cfg.experiment,
            float(cfg.scales.L_ref_um),
            archive["t_star"],
            archive["stations"],
            float(archive["axis_star"]),
        )


def ghost_masks(
    stations: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The ghost outlines on the tensors' pixel grid: the vapour mask ``(n, H,
    W)`` (1 where the pixel's centre is inside) and the signed distance to its
    edge in ``L_ref`` (negative inside), as preprocessing stores the images'
    own."""
    pixel = float(xs[1] - xs[0])
    alpha = np.zeros((len(stations), len(ys), len(xs)), dtype=np.float32)
    sdf = np.zeros_like(alpha)
    for row, curve in enumerate(stations):
        # Only the pixels of the outline's bounding box can be inside it.
        cols = np.nonzero((xs >= curve[:, 0].min()) & (xs <= curve[:, 0].max()))[0]
        rows = np.nonzero((ys >= curve[:, 1].min()) & (ys <= curve[:, 1].max()))[0]
        grid_x, grid_y = np.meshgrid(xs[cols], ys[rows])
        inside = Path(curve).contains_points(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
        alpha[row][np.ix_(rows, cols)] = inside.reshape(len(rows), len(cols))
        sdf[row] = signed_distance(alpha[row].astype(np.uint8)) * pixel
    return alpha, sdf
