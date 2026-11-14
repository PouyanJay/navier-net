"""Ghost frames (``experiment.ghost_duration_ms``): an invented history before
the first image, so the solve starts from a small bubble and computes the state
at the first image."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from naviernet import audit
from naviernet.data import ghost
from naviernet.data import outline as tracing
from naviernet.data.dataset import BubbleDataset
from naviernet.models.measured_outline import MeasuredOutline

from .conftest import CAPSULE_DT, CAPSULE_FRAMES, CAPSULE_Y_CENTER, staged_walled_capsule_run

AXIS = 0.5
ROOT_X = 0.3
STATIONS = 64
L_REF_UM = 300.0
# Three ghost frames at the fixture's 0.5 ms camera interval.
GHOST = ["experiment.ghost_duration_ms=1.5", "experiment.ghost_seed_um=60"]


def _egg(length: float, half_width: float = 0.2) -> np.ndarray:
    """A closed outline in the measured layout: station 0 the nose apex, the
    upper half first, the root apex (fixed at ``ROOT_X``) half way round."""
    q = 2.0 * np.pi * np.arange(STATIONS) / STATIONS
    return np.column_stack(
        [ROOT_X + 0.5 * length * (1.0 + np.cos(q)), AXIS + half_width * np.sin(q)]
    )


def _experiment(shape: str = "similar", duration_ms: float = 2.0) -> SimpleNamespace:
    return SimpleNamespace(
        ghost_duration_ms=duration_ms, dt_frame_ms=0.5, ghost_seed_um=60.0, ghost_shape=shape
    )


def _measured() -> tuple[np.ndarray, np.ndarray]:
    """Three frames of the egg lengthening at a steady rate."""
    times = np.array([0.0, 0.25, 0.5])
    return times, np.stack([_egg(1.0 + 0.4 * t) for t in times])


def _ghost(shape: str = "similar", duration_ms: float = 2.0) -> ghost.GhostOutline:
    times, stations = _measured()
    return ghost.ghost_outline(_experiment(shape, duration_ms), L_REF_UM, times, stations, AXIS)


def test_no_ghost_frames_without_a_duration():
    times, stations = _measured()

    assert (
        ghost.ghost_outline(_experiment(duration_ms=0.0), L_REF_UM, times, stations, AXIS)
        is None
    )


def test_ghost_frames_are_a_camera_interval_apart_and_end_before_the_first_image():
    made = _ghost(duration_ms=2.0)

    assert made.times == pytest.approx([-1.0, -0.75, -0.5, -0.25])
    assert made.stations.shape == (4, STATIONS, 2)


def test_the_growth_starts_at_the_seed_and_joins_the_first_image():
    step = 1e-4
    elapsed = np.array([4.0, 3.0, 2.0, 1.0, step, 0.0])

    areas = ghost.growth_areas(elapsed, first_area=1.0, first_rate=0.3, seed_area=0.2)

    assert areas[0] == pytest.approx(0.2)
    assert areas[-1] == pytest.approx(1.0)
    assert (areas[-1] - areas[-2]) / step == pytest.approx(0.3, rel=1e-3)
    assert np.all(np.diff(areas) > 0)


def test_a_ghost_time_too_long_for_the_growth_to_keep_rising_is_refused():
    with pytest.raises(ValueError, match="at most"):
        ghost.growth_areas(np.array([10.0, 5.0]), first_area=1.0, first_rate=0.3, seed_area=0.2)


@pytest.mark.parametrize("shape", ghost.GHOST_SHAPES)
def test_every_shape_rule_follows_the_same_growth_from_the_same_seed_area(shape):
    seed_area = 0.25 * np.pi * (60.0 / L_REF_UM) ** 2

    areas = [ghost.polygon_area(curve) for curve in _ghost(shape).stations]

    assert areas[0] == pytest.approx(seed_area, rel=1e-3)
    assert np.all(np.diff(areas) > 0)
    assert areas == pytest.approx(
        [ghost.polygon_area(curve) for curve in _ghost("similar").stations], rel=1e-3
    )


@pytest.mark.parametrize("shape", ghost.GHOST_SHAPES)
def test_every_ghost_bubble_sits_on_the_first_images_root_about_the_axis(shape):
    stations = _ghost(shape).stations
    upper, lower = stations[:, 1 : STATIONS // 2], stations[:, : STATIONS // 2 : -1]

    assert stations[:, STATIONS // 2] == pytest.approx(np.tile([ROOT_X, AXIS], (4, 1)))
    assert upper[..., 0] == pytest.approx(lower[..., 0])
    assert upper[..., 1] - AXIS == pytest.approx(AXIS - lower[..., 1])


def test_similar_ghosts_have_the_first_images_shape_at_every_size():
    first = _measured()[1][0]

    for curve in _ghost("similar").stations:
        length = curve[0, 0] - curve[STATIONS // 2, 0]
        width = 2.0 * (curve[:, 1] - AXIS).max()
        assert length / width == pytest.approx(1.0 / 0.4)
        assert (curve - curve[STATIONS // 2]) / length == pytest.approx(
            (first - first[STATIONS // 2]) / 1.0
        )


def test_stretched_ghosts_start_similar_and_only_lengthen_into_the_first_image():
    first = _measured()[1][0]
    stretched = ghost.ghost_outline(
        _experiment("stretch", duration_ms=4.0), L_REF_UM, *_measured(), AXIS
    ).stations
    widths = 2.0 * (stretched[:, :, 1] - AXIS).max(axis=1)
    lengths = stretched[:, 0, 0] - ROOT_X

    assert stretched[0] == pytest.approx(_ghost("similar", duration_ms=4.0).stations[0])
    assert np.all(np.diff(lengths) > 0), "the nose never retreats"
    assert widths[-1] == pytest.approx(2.0 * (first[:, 1] - AXIS).max(), rel=0.05)
    # The widening slows into the first image: the nose takes over the growth.
    assert widths[-1] - widths[-2] < 0.7 * (widths[4] - widths[3])


def test_the_blob_starts_round():
    seed = _ghost("blob").stations[0]
    radius = 0.5 * 60.0 / L_REF_UM
    centre = np.array([0.5 * (seed[0, 0] + ROOT_X), AXIS])

    # A polygon of the disc's area, so its corners sit just outside the disc.
    assert np.linalg.norm(seed - centre, axis=1) == pytest.approx(radius, rel=5e-3)


def test_an_unknown_shape_rule_is_refused():
    with pytest.raises(ValueError, match="ghost_shape"):
        _ghost("square")


# --- The energy growth: the seed grows as fast as the floor dries the film


def _law(growth: float = 1.0, life: float = 1e9, zones: dict | None = None) -> ghost.FilmLaw:
    return ghost.FilmLaw(zones=zones or {}, growth_per_area=growth, life=life)


def test_the_energy_growth_is_exponential_while_the_whole_footprint_stays_wet():
    first = _egg(1.0)
    seed_area = 0.2 * ghost.polygon_area(first)

    elapsed, areas = ghost.energy_growth(first, AXIS, seed_area, "similar", _law(2.0), 10.0)

    # dA/dt = 2 A: the area doubles its log every half time unit.
    assert areas[0] == pytest.approx(seed_area)
    assert areas[-1] >= ghost.polygon_area(first)
    assert elapsed[0] == pytest.approx(np.log(areas[-1] / seed_area) / 2.0, rel=0.05)
    assert np.all(np.diff(elapsed) < 0) and np.all(np.diff(areas) > 0)


def test_a_film_that_dries_slows_the_energy_growth():
    first = _egg(1.0)
    seed_area = 0.2 * ghost.polygon_area(first)

    wet, _ = ghost.energy_growth(first, AXIS, seed_area, "similar", _law(2.0), 10.0)
    drying, _ = ghost.energy_growth(
        first, AXIS, seed_area, "similar", _law(2.0, life=0.5), 50.0
    )

    assert drying[0] > 1.2 * wet[0]


def test_the_energy_growth_refuses_a_seed_the_floor_cannot_heat_or_a_time_too_short():
    first = _egg(1.0)
    seed_area = 0.2 * ghost.polygon_area(first)
    cold = {"q_upstream_ratio": 0.0, "heater_edge_star": first[:, 0].max() + 1.0}

    with pytest.raises(ValueError, match="cannot grow"):
        ghost.energy_growth(first, AXIS, seed_area, "similar", _law(zones=cold), 10.0)
    with pytest.raises(ValueError, match="within"):
        ghost.energy_growth(first, AXIS, seed_area, "similar", _law(0.1), 1.0)


def test_the_energy_ghost_frames_sit_on_camera_instants_and_arrive_at_the_first_image():
    times, stations = _measured()
    experiment = _experiment("stretch", duration_ms=5.0)
    experiment.ghost_growth = "energy"

    made = ghost.ghost_outline(experiment, L_REF_UM, times, stations, AXIS, _law(1.0))

    # At most the ghost time asked for: 5 ms is ten camera intervals of 0.25 t_ref.
    assert made.duration is not None and 0 < made.duration <= 2.5
    assert made.times == pytest.approx(-0.25 * np.arange(len(made.times), 0, -1))
    assert made.times[0] >= -made.duration
    areas = [ghost.polygon_area(curve) for curve in made.stations]
    assert np.all(np.diff(areas) > 0) and areas[-1] < ghost.polygon_area(stations[0])
    # The seed of a disc 60 um across: the earliest frame is at or above it.
    assert areas[0] >= 0.25 * np.pi * (60.0 / L_REF_UM) ** 2 - 1e-9
    # dA/dt = A arrives at the first area; the eggs lengthen 10% per 0.25, so 0.4 A.
    assert made.rate_ratio == pytest.approx(2.5, rel=0.05)


def test_the_energy_growth_needs_the_film_law_and_an_unknown_growth_is_refused():
    times, stations = _measured()
    experiment = _experiment()
    experiment.ghost_growth = "energy"
    with pytest.raises(ValueError, match="film law"):
        ghost.ghost_outline(experiment, L_REF_UM, times, stations, AXIS)
    experiment.ghost_growth = "linear"
    with pytest.raises(ValueError, match="ghost_growth"):
        ghost.ghost_outline(experiment, L_REF_UM, times, stations, AXIS)


def test_the_ghost_masks_cover_the_outlines_area_and_are_negative_inside():
    xs = np.linspace(0.0, 2.0, 201)
    ys = np.linspace(0.0, 1.0, 101)
    stations = _ghost("similar").stations

    alpha, sdf = ghost.ghost_masks(stations, xs, ys)

    pixel = (xs[1] - xs[0]) * (ys[1] - ys[0])
    assert alpha.sum(axis=(1, 2)) * pixel == pytest.approx(
        [ghost.polygon_area(curve) for curve in stations], rel=0.03
    )
    assert np.all(sdf[alpha > 0.5] < 0) and np.all(sdf[alpha < 0.5] > 0)


# --- The dataset and the outline with ghost frames


def _write_outline(paths) -> None:
    """A lengthening egg on the fixture's time axis, where the preprocess stage
    would leave the traced outline."""
    times = np.arange(CAPSULE_FRAMES) * CAPSULE_DT
    np.savez(
        tracing.outline_path(paths),
        t_star=times,
        stations=np.stack([_egg(0.5 + 1.0 * t, half_width=0.12) for t in times]),
        axis_star=CAPSULE_Y_CENTER,
        nose_cap_q=0.1,
        root_cap_q=0.1,
    )


def _staged(tmp_path, overrides=()):
    cfg, paths = staged_walled_capsule_run(tmp_path, list(overrides))
    _write_outline(paths)
    return cfg, paths


def test_the_dataset_puts_the_ghost_frames_before_the_images(tmp_path):
    cfg, paths = _staged(tmp_path, GHOST)

    data = BubbleDataset(cfg, paths)

    assert data.n_ghost == 3 and data.n_event == CAPSULE_FRAMES + 3
    assert data.frame_numbers[:4] == [-2, -1, 0, 1]
    assert data.t[:4] == pytest.approx([-3 * CAPSULE_DT, -2 * CAPSULE_DT, -CAPSULE_DT, 0.0])
    assert data.domain.t_min == pytest.approx(-3 * CAPSULE_DT)
    assert data.alpha.shape == data.sdf.shape == data.valid.shape == data.masks_camera.shape
    assert [data.is_ghost(row) for row in (0, 2, 3)] == [True, True, False]
    assert data.image_rows == list(range(3, CAPSULE_FRAMES + 3))
    areas = data.alpha[:4].sum(axis=(1, 2))
    assert 0 < areas[0] < areas[1] < areas[2] < areas[3]


def test_the_images_are_untouched_without_a_ghost_start(tmp_path):
    cfg, paths = _staged(tmp_path)

    data = BubbleDataset(cfg, paths)

    assert data.n_ghost == 0 and data.n_event == CAPSULE_FRAMES
    assert data.frame_numbers[0] == 1 and data.t[0] == 0.0


def test_ghost_frames_are_never_held_out_and_do_not_move_the_root_anchor(tmp_path):
    held = ["training.val_fraction=0.2", "training.val_strategy=tail"]
    plain = BubbleDataset(*_staged(tmp_path / "plain", held))

    ghosted = BubbleDataset(*_staged(tmp_path / "ghost", [*GHOST, *held]))

    assert ghosted.validation_frames == plain.validation_frames
    assert ghosted.holdout_row == -1, "holdout_frame=-1 is no frame, not ghost frame 0"
    assert ghosted.pin_anchor == pytest.approx(plain.pin_anchor)


def test_a_ghost_start_needs_the_traced_outline(tmp_path):
    cfg, paths = staged_walled_capsule_run(tmp_path, GHOST)

    with pytest.raises(FileNotFoundError, match="ghost_duration_ms"):
        BubbleDataset(cfg, paths)


def test_the_outline_runs_through_the_ghost_frames_into_the_images(tmp_path):
    cfg, paths = _staged(tmp_path, GHOST)
    made = ghost.load_ghost_outline(cfg, paths)

    outline = MeasuredOutline.load(tracing.outline_path(paths), made)

    assert outline.times.tolist() == pytest.approx([*made.times, *np.arange(6) * CAPSULE_DT])
    nose = outline.nose(outline.times[:4]).numpy()
    assert nose[:3] == pytest.approx(made.stations[:, 0, 0], abs=1e-5)
    assert np.all(np.diff(nose) > 0)


# --- The audit's verdicts are about the images


def test_the_scan_verdicts_judge_only_the_instants_from_the_first_image_on():
    scan = {
        "t_star": [-1.0, -0.5, 0.0, 0.5],
        "flux_max_abs": [0.9, 0.8, 0.01, 0.02],
        "residual_rms": {"div": [0.7, 0.6, 0.03, 0.04]},
    }
    acceptance = SimpleNamespace(residual_rms_max=0.05, flux_tol=0.05)

    verdicts = audit._scan_verdicts(scan, acceptance, t_first_image=0.0)

    assert verdicts["flux_in_time"] == {"worst": 0.02, "max": 0.05, "pass": True}
    assert verdicts["residuals_in_time"]["worst"] == 0.04
