"""The measured outline (``model.measured_outline``): the interface traced from the
images as one fair curve per frame, interpolated between frames, standing in for
the trained capsule from the physics stage on."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from naviernet import training, warmup_cache
from naviernet.data import outline as tracing
from naviernet.data.dataset import BubbleDataset
from naviernet.models.geometry import GeometricInterface
from naviernet.models.measured_outline import MeasuredInterface, MeasuredOutline

from .conftest import (
    CAPSULE_DT,
    CAPSULE_FRAMES,
    CAPSULE_GROWTH_PER_FRAME,
    CAPSULE_RADIUS,
    CAPSULE_S0,
    CAPSULE_X_ROOT,
    CAPSULE_Y_CENTER,
    staged_walled_capsule_run,
)

# A circle that drifts and grows linearly, so its outline is exact between frames.
CENTRE_X, CENTRE_Y, RADIUS = 1.0, 0.5, 0.2
DRIFT, GROWTH = 0.3, 0.05
TIMES = np.linspace(0.0, 1.0, 5)
STATIONS = 128


def _circle_stations(t: float, n: int = STATIONS) -> np.ndarray:
    q = np.arange(n) / n
    radius = RADIUS + GROWTH * t
    return np.column_stack(
        [
            CENTRE_X + DRIFT * t + radius * np.cos(2 * np.pi * q),
            CENTRE_Y + radius * np.sin(2 * np.pi * q),
        ]
    )


def _circle() -> MeasuredOutline:
    stations = np.stack([_circle_stations(t) for t in TIMES])
    return MeasuredOutline(TIMES, stations, CENTRE_Y, nose_cap_q=0.2, root_cap_q=0.1)


def _points(n: int = 500, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * torch.tensor([1.0, 0.8]) + torch.tensor([0.6, 0.1])
    t = torch.rand(n, 1, generator=g) * float(TIMES[-1])
    return torch.cat([xy, t], dim=1)


def _circle_distance(x: torch.Tensor) -> torch.Tensor:
    t = x[:, 2:3]
    centre = torch.cat([CENTRE_X + DRIFT * t, torch.full_like(t, CENTRE_Y)], dim=1)
    return RADIUS + GROWTH * t - (x[:, :2] - centre).norm(dim=1, keepdim=True)


def test_the_field_is_the_signed_distance_to_the_outline_at_each_points_own_time():
    outline = _circle()
    x = _points()

    phi = outline.phi(x)

    assert torch.allclose(phi, _circle_distance(x), atol=1e-5)


def test_the_fields_derivatives_are_the_distances():
    outline = _circle()
    x = _points(300, seed=1)
    x = x[_circle_distance(x).abs().squeeze(1) < 0.1].clone().requires_grad_(True)

    phi = outline.phi(x)
    grad = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
    laplacian = sum(
        torch.autograd.grad(grad[:, i].sum(), x, retain_graph=True)[0][:, i] for i in (0, 1)
    )

    t = x[:, 2]
    centre = torch.stack([CENTRE_X + DRIFT * t, torch.full_like(t, CENTRE_Y)], dim=1)
    r = (x[:, :2] - centre).norm(dim=1)
    assert torch.allclose(grad[:, :2].norm(dim=1), torch.ones_like(r), atol=1e-4)
    # The distance to a circle has level sets of curvature 1/r: its Laplacian is -1/r.
    assert torch.allclose(laplacian, -1.0 / r, rtol=1e-2)


def test_the_front_carries_the_outlines_own_curvature_normal_and_speed():
    outline = _circle()
    t = torch.tensor([0.1, 0.55, 0.9])

    front = outline.front(t, n_body=24, n_cap=8)

    times = front.points[:, 2]
    radius = RADIUS + GROWTH * times
    centre = torch.stack([CENTRE_X + DRIFT * times, torch.full_like(times, CENTRE_Y)], dim=1)
    radial = (front.points[:, :2] - centre) / radius.unsqueeze(1)
    assert torch.allclose((front.points[:, :2] - centre).norm(dim=1), radius, atol=1e-5)
    assert torch.allclose(front.normal, radial, atol=1e-4), "outward"
    assert torch.allclose(front.kappa_par.squeeze(1), 1.0 / radius, rtol=1e-3)
    expected_speed = GROWTH + DRIFT * radial[:, 0]
    assert torch.allclose(front.normal_speed.squeeze(1), expected_speed, atol=1e-4)
    assert torch.allclose(
        outline.nose(t), CENTRE_X + DRIFT * t + RADIUS + GROWTH * t, atol=1e-5
    )


def test_the_front_comes_in_the_capsules_layout():
    outline = _circle()
    n_body, n_cap, times = 10, 6, torch.tensor([0.2, 0.7])

    front = outline.front(times, n_body, n_cap)

    segment = [n_body * 2, n_body * 2, n_cap * 2, n_cap * 2]
    upper, lower, root, nose = torch.split(front.points, segment)
    assert (upper[:, 1] > CENTRE_Y).all() and (lower[:, 1] < CENTRE_Y).all()
    assert torch.equal(upper[:n_body, 2], torch.full((n_body,), 0.2)), "time-major"
    assert (root[:, 0] < nose[:, 0].min()).all(), "the root cap upstream of the nose"
    _, _, root_u, nose_u = torch.split(front.u, segment)
    assert (root_u == 0).all() and (nose_u == 1).all()
    on_cap = torch.split(front.on_cap, segment)
    assert (on_cap[0] == 0).all() and (on_cap[2] == 1).all()
    caps = torch.cat([root, nose])
    sides = torch.cat(torch.split(front.side, segment)[2:]).squeeze(1)
    assert ((caps[:, 1] > CENTRE_Y) == (sides > 0)).all(), "a cap's side is its half"


def test_beyond_the_frames_the_outline_moves_on_at_its_end_rate():
    outline = _circle()
    late = torch.tensor([[2.0, CENTRE_Y, 1.2]])

    assert torch.allclose(outline.phi(late), _circle_distance(late), atol=1e-4)


def _ellipse_band(a: float, b: float, centre: tuple[float, float], width: float, shape):
    """A dark rim of ``2 width`` pixels around an ellipse, as segmentation leaves it."""
    inside = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(inside, (centre, (2 * a, 2 * b), 0.0), 1, -1)
    to_edge_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    to_edge_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
    return (
        ((to_edge_in < width) & (inside > 0)) | ((to_edge_out < width) & (inside == 0))
    ).astype(np.uint8)


def test_the_pen_traces_a_rim_back_to_the_curve_at_its_centre():
    a, b, centre = 200.0, 60.0, (300.0, 100.0)
    band = _ellipse_band(a, b, centre, 8, (200, 600))

    rim = tracing.rim_centre(band, axis_row=centre[1])
    stroke = tracing.PenStroke()
    x, y, _, kappa = stroke.trace(stroke.fit(rim, um_per_px=1.0))

    assert rim[0, 0] < rim[-1, 0] and (rim[:, 1] >= 0).all(), "apex to apex, upper half"
    on_ellipse = ((x - centre[0]) / a) ** 2 + (y / b) ** 2
    assert np.abs(np.sqrt(on_ellipse) - 1.0).max() * b < 1.0, "within a pixel"
    flank = np.abs(x - centre[0]) < 0.5 * a
    assert np.allclose(kappa[flank], _ellipse_curvature(x[flank] - centre[0], a, b), atol=1e-3)


def _ellipse_curvature(dx: np.ndarray, a: float, b: float) -> np.ndarray:
    cos = dx / a
    sin = np.sqrt(np.clip(1.0 - cos**2, 0.0, None))
    return a * b / (a**2 * sin**2 + b**2 * cos**2) ** 1.5


def test_stations_hold_their_distance_from_each_apex_as_the_bubble_grows():
    short, long = tracing.station_arc(700.0), tracing.station_arc(1500.0)
    near = tracing.END_ARC_UM * 0.9

    assert np.all(np.diff(short) > 0) and np.all(np.diff(long) > 0)
    assert np.allclose(short[short < near], long[: (short < near).sum()])
    assert np.allclose(
        700.0 - short[short > 700.0 - near], (1500.0 - long)[-(short > 700.0 - near).sum() :]
    )
    with pytest.raises(ValueError, match="too short"):
        tracing.station_arc(300.0)


# --- The model and the trainer


MEASURED = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.mirror_symmetry=true",
    "model.measured_outline=true",
    "training.stage_b_warmup_steps=2",
    "training.freeze_geometry=true",
    "training.log_every=1",
]


def _write_capsule_outline(paths) -> None:
    """The fixture capsule's outline, as preprocess would trace it."""
    times = np.arange(CAPSULE_FRAMES) * CAPSULE_DT
    stations = np.stack(
        [_stadium(CAPSULE_S0 + k * CAPSULE_GROWTH_PER_FRAME) for k in range(CAPSULE_FRAMES)]
    )
    np.savez(
        tracing.outline_path(paths),
        t_star=times,
        stations=stations,
        axis_star=CAPSULE_Y_CENTER,
        nose_cap_q=0.1,
        root_cap_q=0.1,
    )


def _stadium(nose: float, n: int = 96) -> np.ndarray:
    """The capsule from the root to ``nose``, counter-clockwise from the nose apex,
    evenly spaced along it."""
    length, radius = nose - CAPSULE_X_ROOT, CAPSULE_RADIUS
    perimeter = 2 * length + 2 * np.pi * radius
    s = np.arange(n) / n * perimeter
    arcs = np.cumsum([0.0, np.pi * radius / 2, length, np.pi * radius, length])
    points = []
    for d in s:
        if d < arcs[1]:
            angle = d / radius
            points.append(
                (nose + radius * np.cos(angle), CAPSULE_Y_CENTER + radius * np.sin(angle))
            )
        elif d < arcs[2]:
            points.append((nose - (d - arcs[1]), CAPSULE_Y_CENTER + radius))
        elif d < arcs[3]:
            angle = 0.5 * np.pi + (d - arcs[2]) / radius
            points.append(
                (
                    CAPSULE_X_ROOT + radius * np.cos(angle),
                    CAPSULE_Y_CENTER + radius * np.sin(angle),
                )
            )
        elif d < arcs[4]:
            points.append((CAPSULE_X_ROOT + (d - arcs[3]), CAPSULE_Y_CENTER - radius))
        else:
            angle = 1.5 * np.pi + (d - arcs[4]) / radius
            points.append(
                (nose + radius * np.cos(angle), CAPSULE_Y_CENTER + radius * np.sin(angle))
            )
    return np.asarray(points)


def _staged(tmp_path, extra=()):
    cfg, paths = staged_walled_capsule_run(tmp_path, [*MEASURED, *extra])
    _write_capsule_outline(paths)
    return cfg, paths


def test_the_capsule_stands_until_the_outline_is_engaged(tmp_path):
    cfg, paths = _staged(tmp_path)
    data = BubbleDataset(cfg, paths)
    priors = training._geometry_priors(cfg, data)
    torch.manual_seed(0)
    capsule = GeometricInterface(priors, mirror_axis=CAPSULE_Y_CENTER)
    torch.manual_seed(0)
    measured = MeasuredInterface(priors, mirror_axis=CAPSULE_Y_CENTER)
    x = torch.rand(64, 3) * torch.tensor([2.0, 1.0, 0.5])

    assert torch.equal(capsule(x), measured(x)), "the same capsule, the same draws"
    with pytest.raises(RuntimeError, match="no outline is bound"):
        measured.engage(True)
    measured.bind_outline(MeasuredOutline.load(tracing.outline_path(paths)))
    measured.engage(True)
    assert torch.equal(measured(x), measured.outline.phi(x))


def test_the_physics_stage_trains_on_the_measured_outline(tmp_path):
    cfg, paths = _staged(tmp_path, ["training.steps=3"])

    model, _, state = training.train(cfg, paths)

    geometry = model.nets["phi"]
    assert state["done"] == 3 and geometry.engaged
    t = torch.tensor([[0.15]])
    front, measured = model.front(t, 8, 4), geometry.outline.front(t, 8, 4)
    assert torch.equal(front.points, measured.points)
    loaded, _, _ = training.load_model(cfg, paths)
    assert loaded.nets["phi"].engaged, "scored on the outline it trained on"


def test_without_its_file_the_outline_says_how_to_make_it(tmp_path):
    cfg, paths = staged_walled_capsule_run(tmp_path, MEASURED)

    with pytest.raises(FileNotFoundError, match="stage=preprocess"):
        training.train(cfg, paths)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("training.freeze_geometry=false", "freeze_geometry"),
        ("model.mirror_symmetry=false", "mirror_symmetry"),
    ],
)
def test_the_outline_needs_the_frozen_geometry_and_the_mirror(tmp_path, override, message):
    cfg, _ = _staged(tmp_path, [override])

    with pytest.raises(ValueError, match=message):
        training._validate_training_config(cfg)


def test_switching_the_outline_on_keeps_the_shared_warm_up(tmp_path):
    cfg, paths = _staged(tmp_path)
    off, _ = staged_walled_capsule_run(tmp_path, [*MEASURED, "model.measured_outline=false"])
    keys = ("brinkman",)

    assert (
        warmup_cache.warmup_key(cfg, paths, keys)[0]
        == warmup_cache.warmup_key(off, paths, keys)[0]
    )
