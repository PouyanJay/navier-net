"""Ghost frames: an invented history of the bubble before the first image.

The images start with the bubble already 0.57 mm long, so a solve that starts
there has to be told the state at frame 1: how long each part of the floor has
been covered, the cooled layer around the bubble, the flow's memory. Ghost
frames move the start back to an instant where the state is simple (a small
bubble on the cavity, undisturbed flow, the heaters' steady temperature) and let
the solver compute frame 1 (``experiment.ghost_duration_ms``).

The history is a rule, not a measurement:

- **Growth** (``experiment.ghost_growth``). ``parabola``: the area is the
  parabola through the seed's area at the start that joins frame 1 with its
  area and with the growth rate the first two images show, over the ghost time
  asked for. It is the smoothest growth that does so, but nothing ties it to the
  heater. ``energy``: the area grows as fast as the floor's flux evaporates the
  film under the footprint (the solver's own film law: the local flux, a film
  that lasts its thickness over that flux from the instant the footprint
  covers a spot), integrated forward from the seed until it reaches the first
  image's area. The ghost time is then an output, and the rate the growth
  arrives with at the first image, against the images' own, says whether the
  heater is right. The outline's own evaporation is not in it.
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
from naviernet.physics.heating import DEPLETION_SMOOTHING, wall_flux_ratio_np
from naviernet.utils.paths import RunPaths

# ``experiment.ghost_shape``.
GHOST_SHAPES = ("similar", "stretch", "blob")
# ``experiment.ghost_growth``.
GHOST_GROWTHS = ("parabola", "energy")
# Bisection steps for the blend that gives an area: 2^-40 of the blend's range.
BLEND_STEPS = 40
# The energy growth's time step and the cells it integrates the footprint on
# (both in L_ref units: 0.02 t_ref is 30 us and 1/150 L_ref is 2 um at the
# 300 um scale, both far below the growth's own scales).
ENERGY_STEP = 0.02
ENERGY_CELL = 1.0 / 150.0


@dataclass(frozen=True)
class GhostOutline:
    """The ghost frames' instants (``t_ref``), ascending and ending one camera
    interval before the first image, and their closed outlines ``(n, M, 2)`` in
    the measured outline's station layout. Under the energy growth, also the
    time the seed took to reach the first image (``t_ref``) and the rate it
    arrived with over the rate the first two images show."""

    times: np.ndarray
    stations: np.ndarray
    duration: float | None = None
    rate_ratio: float | None = None


@dataclass(frozen=True)
class FilmLaw:
    """What the energy growth integrates, in ``L_ref`` and ``t_ref`` units:
    the floor's flux at any station as a dilation rate (``L_ref^2`` of
    projected vapour per ``L_ref^2`` of wet footprint per ``t_ref``), and the
    film's life under unit flux ratio."""

    zones: dict[str, float]
    growth_per_area: float  # q_wall / (rho_v h_lv H) on 1 / t_ref
    life: float  # rho_l h_lv delta / q_wall on t_ref

    @classmethod
    def from_config(cls, cfg) -> FilmLaw:
        from naviernet.physics.groups import compute_groups

        exp, fluid = cfg.experiment, cfg.fluid
        if str(getattr(exp, "footprint_wetting", "uniform")) != "microlayer":
            raise ValueError(
                "experiment.ghost_growth=energy grows the seed by evaporating the film the "
                "floor's flux dries, so it needs experiment.footprint_wetting=microlayer"
            )
        groups = compute_groups(cfg)
        zones = {
            key: float(groups[key])
            for key in (
                "q_upstream_ratio",
                "heater_edge_star",
                "q_hot_ratio",
                "hot_zone_end_star",
            )
            if key in groups
        }
        t_ref = groups["t_ref_ms"] * 1e-3
        q_wall = float(exp.q_wall_W_cm2) * 1e4
        depth = float(exp.channel_height_um) * 1e-6
        return cls(
            zones=zones,
            growth_per_area=q_wall / (fluid.rho_v * fluid.h_lv * depth) * t_ref,
            life=float(groups["microlayer_tau_star_per_um"]) * float(exp.microlayer_um),
        )


def ghost_frame_count(experiment) -> int:
    """How many ghost frames ``experiment.ghost_duration_ms`` asks for: the
    duration in camera intervals, rounded; zero when the ghost start is off.
    Under the energy growth it is the most the growth may take."""
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


def _stretch_shapes(
    first: np.ndarray, areas: np.ndarray, progress: np.ndarray, seed_area: float
) -> np.ndarray:
    """The first outline scaled about its root apex, separately along and
    across the channel. The width goes from the similar seed's (the seed of
    ``seed_area``) to the first outline's as ``progress`` (the share of the
    ghost time gone by) goes from 0 to 1; the length follows from each area."""
    root = first[len(first) // 2]
    seed = np.sqrt(seed_area / polygon_area(first))
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
    first: np.ndarray,
    axis: float,
    areas: np.ndarray,
    progress: np.ndarray,
    rule: str,
    seed_area: float | None = None,
) -> np.ndarray:
    """One closed outline per area, ``(n, M, 2)``, by the shape ``rule``;
    ``progress`` is each frame's share of the ghost time gone by, ``seed_area``
    the seed's (the first area when not given)."""
    if rule == "similar":
        return _similar_shapes(first, areas)
    if rule == "stretch":
        seed = float(areas[0]) if seed_area is None else seed_area
        return _stretch_shapes(first, np.asarray(areas, dtype=float), progress, seed)
    if rule == "blob":
        return _blob_shapes(first, axis, areas)
    raise ValueError(f"experiment.ghost_shape must be one of {GHOST_SHAPES}, got {rule!r}")


def energy_growth(
    first: np.ndarray,
    axis: float,
    seed_area: float,
    rule: str,
    law: FilmLaw,
    longest: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The area's history under the film law, from the seed until it reaches
    the first outline's area: ``(elapsed, areas)``, the time before the first
    image of each integration step (descending from the seed's) and the area
    then. The footprint at each step is the shape rule's at that area, the
    widening of ``stretch`` measured by the share of the area gained (the time
    is not known yet). Raises when the floor cannot grow the seed, or not
    within ``longest`` (``t_ref``)."""
    first_area = polygon_area(first)
    lo = first.min(axis=0) - ENERGY_CELL
    hi = first.max(axis=0) + ENERGY_CELL
    gx = np.arange(lo[0], hi[0], ENERGY_CELL)
    gy = np.arange(lo[1], hi[1], ENERGY_CELL)
    cells = np.column_stack([v.ravel() for v in np.meshgrid(gx, gy)])
    ratio = wall_flux_ratio_np(cells[:, 0], law.zones)
    life = law.life / np.maximum(ratio, 1e-6)
    covered_at = np.full(len(cells), np.inf)
    record_t, record_a = [0.0], [seed_area]
    elapsed, area = 0.0, seed_area
    while area < first_area:
        progress = np.array([(area - seed_area) / (first_area - seed_area)])
        shape = ghost_shapes(first, axis, np.array([area]), progress, rule, seed_area)[0]
        inside = Path(shape).contains_points(cells)
        covered_at = np.where(inside & np.isinf(covered_at), elapsed, covered_at)
        z = (covered_at + life - elapsed) / (DEPLETION_SMOOTHING * life)
        wet = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        rate = law.growth_per_area * float((ratio * wet)[inside].sum()) * ENERGY_CELL**2
        # Below a millionth of a fully heated, wet footprint: the smoothed
        # heater edge's tail, not heat.
        if rate <= 1e-6 * law.growth_per_area * area:
            raise ValueError(
                "experiment.ghost_growth=energy: the floor under the seed puts no heat into "
                f"its film (area {area:.4g} L_ref^2 after {elapsed:.3g} t_ref), so the seed "
                "cannot grow; move the heater's edge or the seed"
            )
        area += rate * ENERGY_STEP
        elapsed += ENERGY_STEP
        record_t.append(elapsed)
        record_a.append(area)
        if elapsed > longest:
            raise ValueError(
                "experiment.ghost_growth=energy: the seed has not reached the first image's "
                f"area within experiment.ghost_duration_ms ({area:.4g} of {first_area:.4g} "
                "L_ref^2); the floor's flux or the film is too small, or allow more ghost time"
            )
    return elapsed - np.array(record_t), np.array(record_a)


def ghost_outline(
    experiment,
    l_ref_um: float,
    times: np.ndarray,
    stations: np.ndarray,
    axis: float,
    law: FilmLaw | None = None,
) -> GhostOutline | None:
    """The ghost frames before a measured outline (its ``times`` and
    ``stations``), or ``None`` when the ghost start is off. The energy growth
    needs the film ``law``."""
    count = ghost_frame_count(experiment)
    if count == 0:
        return None
    times = np.asarray(times, dtype=np.float64)
    stations = np.asarray(stations, dtype=np.float64)
    interval = float(times[1] - times[0])
    first_area = polygon_area(stations[0])
    first_rate = (polygon_area(stations[1]) - first_area) / interval
    seed_diameter = float(experiment.ghost_seed_um) / float(l_ref_um)
    seed_area = 0.25 * np.pi * seed_diameter**2
    rule = str(getattr(experiment, "ghost_shape", "similar"))
    growth = str(getattr(experiment, "ghost_growth", "parabola"))
    if growth == "parabola":
        elapsed = interval * np.arange(count, 0, -1)
        areas = growth_areas(elapsed, first_area, first_rate, seed_area)
        progress = 1.0 - elapsed / elapsed[0]
        duration = rate_ratio = None
    elif growth == "energy":
        if law is None:
            raise ValueError("experiment.ghost_growth=energy needs the film law")
        history_t, history_a = energy_growth(
            stations[0], axis, seed_area, rule, law, count * interval
        )
        duration = float(history_t[0])
        # Frames one camera interval apart back from the first image, as far as
        # the seed's instant; the earliest is the first camera instant after it.
        elapsed = interval * np.arange(int(duration / interval), 0, -1)
        if len(elapsed) == 0:
            raise ValueError(
                f"experiment.ghost_growth=energy: the seed reaches the first image in "
                f"{duration:.3g} t_ref, less than one camera interval; use a smaller seed"
            )
        areas = np.interp(-elapsed, -history_t, history_a)
        progress = 1.0 - elapsed / duration
        rate_ratio = float((history_a[-1] - history_a[-2]) / ENERGY_STEP / first_rate)
    else:
        raise ValueError(
            f"experiment.ghost_growth must be one of {GHOST_GROWTHS}, got {growth!r}"
        )
    return GhostOutline(
        times=times[0] - elapsed,
        stations=ghost_shapes(stations[0], axis, areas, progress, rule, seed_area),
        duration=duration,
        rate_ratio=rate_ratio,
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
    law = None
    if str(getattr(cfg.experiment, "ghost_growth", "parabola")) == "energy":
        law = FilmLaw.from_config(cfg)
    with np.load(path) as archive:
        return ghost_outline(
            cfg.experiment,
            float(cfg.scales.L_ref_um),
            archive["t_star"],
            archive["stations"],
            float(archive["axis_star"]),
            law,
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
