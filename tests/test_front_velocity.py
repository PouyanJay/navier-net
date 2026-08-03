"""Measured front velocity: giving the interface's RATE a real target.

The shape is supervised at every training frame; its rate is not. Under the
front-geometry parameterisation the profile travels along ``rate(u)`` and the
nose along its own integrated rate, and those are exactly what carry the shape
past the last supervised frame -- yet nothing measures them.

Two masks cannot give a surface point's full material velocity (a curve sliding
along itself looks identical between frames), but only the NORMAL component
moves an interface, so what is measurable is complete for the shape's evolution.
The nose apex is the exception: it is geometrically distinguishable, so its full
2-D displacement is honest.

The synthetic dataset here is a capsule growing along +x at a known rate with a
fixed radius, written with a TRUE signed distance field. Its normal velocity is
known analytically: the nose advances at ``ds/dt``, and the sides -- whose
radius never changes -- do not move at all. That analytic answer is what the
measurement is held to.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tests.conftest import make_config

# The synthetic capsule: a fixed root, a fixed radius, a nose advancing one
# tenth of x* per frame on a time axis whose step is one tenth of t*.
H_PX, W_PX = 48, 96
X_MAX, Y_MAX = 2.0, 1.0
X_ROOT, Y_CENTER, RADIUS = 0.3, 0.5, 0.12
S0, GROWTH_PER_FRAME, DT = 0.6, 0.1, 0.1
N_FRAMES = 6
TRUE_NOSE_SPEED = GROWTH_PER_FRAME / DT  # x* per t*

GEO = ["model.front_geometry=true"]
FV = [*GEO, "training.front_velocity=true"]


def _capsule_sdf(xs: np.ndarray, ys: np.ndarray, nose: float) -> np.ndarray:
    """Signed distance to the capsule spanning ``X_ROOT..nose``, negative inside."""
    grid_x, grid_y = np.meshgrid(xs, ys)
    nearest_x = np.clip(grid_x, X_ROOT, nose)
    return np.hypot(grid_x - nearest_x, grid_y - Y_CENTER) - RADIUS


def _write_capsule(path) -> None:
    """A growing capsule, staged as preprocess would leave it."""
    xs = np.linspace(0.0, X_MAX, W_PX, dtype=np.float32)
    ys = np.linspace(0.0, Y_MAX, H_PX, dtype=np.float32)
    sdf = np.stack([_capsule_sdf(xs, ys, S0 + k * GROWTH_PER_FRAME) for k in range(N_FRAMES)])
    alpha = (sdf < 0).astype(np.float32)
    meta = {
        "x_pin_star": X_ROOT - RADIUS,
        "t_ref_ms": 1.0,
        "n_frames_usable": N_FRAMES,
        "n_frames_event": N_FRAMES,
        "frame_numbers": list(range(1, N_FRAMES + 1)),
    }
    np.savez_compressed(
        path,
        alpha=alpha,
        sdf=sdf.astype(np.float32),
        valid=np.ones_like(alpha, dtype=np.uint8),
        masks_camera=(alpha > 0.5).astype(np.uint8),
        x_star=xs,
        y_star=ys,
        t_star=(np.arange(N_FRAMES) * DT).astype(np.float32),
        meta=json.dumps(meta),
    )


def _staged(tmp_path, overrides=None):
    """The capsule dataset staged for a tiny run, as the CLI would compose it."""
    from naviernet.utils.paths import RunPaths

    cfg = make_config(
        [
            f"paths.root={tmp_path}",
            "model.hidden=8",
            "model.layers=2",
            "model.fourier_feats=4",
            "training.steps=2",
            "training.n_data=16",
            "training.n_coll=16",
            "training.n_bc=8",
            "training.holdout_frame=-1",
            *(overrides or []),
        ]
    )
    paths = RunPaths.from_config(cfg)
    paths.ensure()
    paths.tensors.parent.mkdir(parents=True, exist_ok=True)
    _write_capsule(paths.tensors)
    return cfg, paths


def _dataset(tmp_path, overrides=None):
    from naviernet.data.dataset import BubbleDataset
    from naviernet.utils.paths import RunPaths

    cfg, _ = _staged(tmp_path, overrides)
    return cfg, BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")


def _model_and_plan(tmp_path, overrides=None):
    from naviernet.models.pinn import BubblePINN
    from naviernet.physics import front_velocity
    from naviernet.training import _geometry_priors

    cfg, data = _dataset(tmp_path, [*FV, *(overrides or [])])
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    plan = front_velocity.build_plan(data, cfg.training, cfg.model, "cpu")
    return cfg, data, model, plan


# --- The measurement itself, against analytic answers -----------------------


def test_normal_velocity_recovers_an_expanding_circle(tmp_path):
    """An expanding circle's interface advances at dr/dt everywhere on it."""
    from naviernet.data.velocimetry import normal_velocity

    xs = np.linspace(-1.0, 1.0, 81)
    grid_x, grid_y = np.meshgrid(xs, xs)
    distance = np.hypot(grid_x, grid_y)
    spacing = (float(xs[1] - xs[0]),) * 2
    rate, dt = 0.3, 0.05

    v_n, resolvable = normal_velocity(distance - 0.4, distance - (0.4 + rate * dt), dt, spacing)

    near = resolvable & (np.abs(distance - 0.4) < 0.1)
    assert np.allclose(v_n[near], rate, atol=1e-3)


def test_normal_velocity_keeps_only_the_normal_part_of_a_translation(tmp_path):
    """A circle sliding sideways at ``vx`` moves its own front at ``vx * n_x``.

    The tangential part of the motion is invisible in the masks -- and correctly
    so: it does not change the shape.
    """
    from naviernet.data.velocimetry import normal_velocity

    xs = np.linspace(-1.0, 1.0, 121)
    grid_x, grid_y = np.meshgrid(xs, xs)
    spacing = (float(xs[1] - xs[0]),) * 2
    speed, dt, radius = 0.4, 0.05, 0.5

    v_n, resolvable = normal_velocity(
        np.hypot(grid_x, grid_y) - radius,
        np.hypot(grid_x - speed * dt, grid_y) - radius,
        dt,
        spacing,
    )

    # On the circle the outward normal is the radial direction, so n_x = x/r.
    distance = np.hypot(grid_x, grid_y)
    on_front = resolvable & (np.abs(distance - radius) < 0.05)
    expected = speed * grid_x[on_front] / distance[on_front]
    assert np.allclose(v_n[on_front], expected, atol=2e-2)


def test_normal_velocity_reports_an_unresolvable_gradient_rather_than_a_number():
    """Where the field is flat there is no normal to move along. Dividing by that
    gradient would turn segmentation noise into an arbitrarily fast front."""
    from naviernet.data.velocimetry import normal_velocity

    flat = np.zeros((16, 16))
    v_n, resolvable = normal_velocity(flat, flat + 0.1, 0.1, (0.05, 0.05))

    assert not resolvable.any()
    assert np.all(v_n == 0.0)


def test_normal_velocity_rejects_a_non_positive_interval():
    from naviernet.data.velocimetry import normal_velocity

    field = np.zeros((8, 8))
    with pytest.raises(ValueError, match="must be > 0"):
        normal_velocity(field, field, 0.0, (0.1, 0.1))


# --- Which frames may be paired --------------------------------------------


def test_supervised_pairs_span_every_consecutive_training_frame(tmp_path):
    _, data = _dataset(tmp_path)
    assert data.supervised_pairs == [(k, k + 1) for k in range(N_FRAMES - 1)]


def test_a_pair_touching_the_holdout_frame_is_dropped(tmp_path):
    """Differencing a held-out mask would make the validation set a training
    target. Both pairs that touch it go -- the gap is never bridged."""
    _, data = _dataset(tmp_path, ["training.holdout_frame=2"])  # 0-based -> row 2

    assert data.validation_rows == [2]
    assert (1, 2) not in data.supervised_pairs
    assert (2, 3) not in data.supervised_pairs
    assert data.supervised_pairs == [(0, 1), (3, 4), (4, 5)]


def test_a_pair_across_an_excluded_camera_frame_is_dropped(tmp_path):
    """An excluded frame leaves a real gap on the time axis; a difference taken
    across it measures a different interval, not the same quantity."""
    from naviernet.data.dataset import BubbleDataset
    from naviernet.utils.paths import RunPaths

    cfg, paths = _staged(tmp_path)
    archive = dict(np.load(paths.tensors))
    meta = json.loads(str(archive["meta"]))
    meta["frame_numbers"] = [1, 2, 4, 5, 6, 7]  # frame 3 excluded at preprocess
    archive["meta"] = json.dumps(meta)
    np.savez_compressed(paths.tensors, **archive)

    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    assert (1, 2) not in data.supervised_pairs  # rows 1 and 2 are frames 2 and 4
    assert data.supervised_pairs == [(0, 1), (2, 3), (3, 4), (4, 5)]


def test_the_measured_apex_sits_on_the_front_and_on_the_centreline(tmp_path):
    _, data = _dataset(tmp_path)
    x_apex, y_apex = data.front_apex(0)

    assert x_apex == pytest.approx(S0 + RADIUS, abs=2 * X_MAX / W_PX)
    assert y_apex == pytest.approx(Y_CENTER, abs=2 * Y_MAX / H_PX)


# --- The plan: what the measurement says about a known capsule --------------


def test_the_plan_measures_the_nose_advancing_and_the_sides_standing_still(tmp_path):
    """The analytic answer for this capsule: the nose advances at ``ds/dt``, and
    the sides -- fixed radius -- do not move at all.

    Measured unblurred, so this holds the estimator itself to the analytic answer;
    what the blur costs is its own test below.
    """
    _, _, _, plan = _model_and_plan(tmp_path, ["training.fv_smooth_px=0.0"])

    x_index = int(round((S0 - 0.0) / (X_MAX / (W_PX - 1))))
    y_index = int(round(Y_CENTER / (Y_MAX / (H_PX - 1))))
    nose_column = int(round((S0 + RADIUS) / (X_MAX / (W_PX - 1))))
    measured = plan.v_n[0].numpy()

    assert measured[y_index, nose_column] == pytest.approx(TRUE_NOSE_SPEED, rel=0.05)
    # A point on the upper flank, mid-body: its radius never changes.
    flank = int(round((Y_CENTER + RADIUS) / (Y_MAX / (H_PX - 1))))
    assert measured[flank, x_index] == pytest.approx(0.0, abs=0.02)


def test_the_blur_costs_accuracy_only_where_the_cap_is_barely_resolved(tmp_path):
    """What the smoothing is worth knowing about: blurring a signed distance
    field rounds a CONVEX cap, so the nose reads slow when the cap spans only a
    few pixels. The error falls away as the cap resolves -- at this fixture's 5.7
    px it costs ~16%, by 23 px it is ~1%, and the real datasets' nose cap is ~75
    px across. That is the whole justification for the 1.5 px default.
    """
    from naviernet.data.velocimetry import normal_velocity

    def nose_speed(width_px: int) -> float:
        height_px = int(width_px * Y_MAX / X_MAX)
        xs = np.linspace(0.0, X_MAX, width_px)
        ys = np.linspace(0.0, Y_MAX, height_px)
        spacing = (float(ys[1] - ys[0]), float(xs[1] - xs[0]))
        v_n, _ = normal_velocity(
            _capsule_sdf(xs, ys, S0),
            _capsule_sdf(xs, ys, S0 + GROWTH_PER_FRAME),
            DT,
            spacing,
            1.5,
        )
        row = int(round(Y_CENTER / (Y_MAX / (height_px - 1))))
        return float(v_n[row, int(round((S0 + RADIUS) / (X_MAX / (width_px - 1))))])

    coarse, fine = nose_speed(W_PX), nose_speed(8 * W_PX)
    assert coarse < 0.9 * TRUE_NOSE_SPEED
    assert fine == pytest.approx(TRUE_NOSE_SPEED, rel=0.01)


def test_the_measured_apex_shift_is_the_true_growth(tmp_path):
    _, _, _, plan = _model_and_plan(tmp_path)

    assert plan.apex_shift.shape == (N_FRAMES - 1, 2)
    assert float(plan.apex_shift[0, 0]) == pytest.approx(GROWTH_PER_FRAME, abs=X_MAX / W_PX)
    assert float(plan.apex_shift[0, 1]) == pytest.approx(0.0, abs=Y_MAX / H_PX)


def test_the_plan_refuses_to_run_with_nothing_to_measure(tmp_path):
    """Every frame held out leaves no honest interval. Contributing zero here
    would be a term that reports success and supervises nothing."""
    from naviernet.physics import front_velocity

    cfg, data = _dataset(tmp_path, [*FV, "training.val_fraction=1.0"])
    with pytest.raises(ValueError, match="at least one pair of consecutive"):
        front_velocity.build_plan(data, cfg.training, cfg.model, "cpu")


# --- The loss ---------------------------------------------------------------


def test_the_loss_reaches_the_geometry_that_sets_the_rate(tmp_path):
    """A term that cannot move ``rate(u)`` cannot do the job it exists for."""
    from naviernet.physics import front_velocity

    cfg, _, model, plan = _model_and_plan(tmp_path)
    total, record = front_velocity.front_velocity_losses(
        front_velocity.FrontVelocityContext(model, cfg.training), plan
    )
    total.backward()

    assert set(record) == {"fv_normal", "fv_apex"}
    assert torch.isfinite(total) and float(total.detach()) > 0
    gradients = [p.grad for p in model.nets["phi"].rate_net.parameters() if p.grad is not None]
    assert gradients and any(float(g.abs().sum()) > 0 for g in gradients)


def test_supervising_the_measurement_moves_the_nose_onto_it(tmp_path):
    """The point of the term, end to end: optimise it alone and the model's own
    apex displacement converges on the measured one."""
    from naviernet.physics import front_velocity

    cfg, _, model, plan = _model_and_plan(tmp_path, ["training.fv_weight=0.0"])
    geometry = model.nets["phi"]
    context = front_velocity.FrontVelocityContext(model, cfg.training)

    def apex_error() -> float:
        with torch.no_grad():
            shift = geometry.apex(plan.times_next) - geometry.apex(plan.times)
        return float((shift - plan.apex_shift).abs().mean())

    before = apex_error()
    optimiser = torch.optim.Adam(geometry.parameters(), lr=0.02)
    for _ in range(150):
        optimiser.zero_grad()
        total, _ = front_velocity.front_velocity_losses(context, plan)
        total.backward()
        optimiser.step()

    assert apex_error() < 0.25 * before


def test_averaging_more_samples_per_bin_keeps_more_mask_noise_out(tmp_path):
    """The reason the profile is binned at all, as a measurement.

    Salt the measured rasters and see how much of that noise reaches the term.
    Averaging is the only thing standing between single-pixel segmentation noise
    and the learned rate, so more samples per bin must let less through.
    """
    from dataclasses import replace

    from naviernet.physics import front_velocity

    generator = torch.Generator().manual_seed(0)
    leaked = []
    for per_bin in (1, 16):
        cfg, _, model, plan = _model_and_plan(
            tmp_path,
            ["training.fv_apex_weight=0.0", f"training.fv_samples_per_bin={per_bin}"],
        )
        context = front_velocity.FrontVelocityContext(model, cfg.training)
        noise = torch.randn(plan.v_n.shape, generator=generator) * 0.2
        clean, _ = front_velocity.front_velocity_losses(context, plan)
        salted, _ = front_velocity.front_velocity_losses(
            context, replace(plan, v_n=plan.v_n + noise)
        )
        leaked.append(abs(float(salted.detach()) - float(clean.detach())))

    assert leaked[1] < 0.5 * leaked[0]


def test_the_caps_are_binned_too_rather_than_one_sample_each(tmp_path):
    """A bin count says nothing about how much averaging it buys. Stated as
    samples-per-bin, the caps -- which carry a quarter as many samples as the
    body -- get their own, coarser bin count instead of one sample per bin."""
    _, _, _, plan = _model_and_plan(tmp_path)

    assert plan.bins_body == plan.n_body // 4
    assert plan.bins_cap == plan.n_cap // 4
    assert plan.bins_per_time == 2 * (plan.bins_body + plan.bins_cap)


# --- Gates ------------------------------------------------------------------


def test_it_is_off_by_default(cfg):
    assert cfg.training.front_velocity is False


def test_it_refuses_to_run_without_the_front_it_supervises(tmp_path):
    from naviernet.training import train
    from naviernet.utils.paths import RunPaths

    cfg, _ = _staged(tmp_path, ["training.front_velocity=true"])  # no front geometry
    with pytest.raises(ValueError, match="model.front_geometry=true"):
        train(cfg, RunPaths.from_config(cfg))


def test_it_refuses_to_measure_and_then_ignore_the_measurement(tmp_path):
    from naviernet.training import train
    from naviernet.utils.paths import RunPaths

    cfg, _ = _staged(tmp_path, [*FV, "training.fv_weight=0", "training.fv_apex_weight=0"])
    with pytest.raises(ValueError, match="measures the front and then ignores it"):
        train(cfg, RunPaths.from_config(cfg))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("training.fv_samples_per_bin=0", "must be >= 1"),
        ("training.fv_smooth_px=-1.0", "must be >= 0"),
        ("training.fv_weight=-1.0", "must be >= 0"),
    ],
)
def test_a_degenerate_knob_fails_before_any_measurement(tmp_path, override, message):
    from naviernet.training import train
    from naviernet.utils.paths import RunPaths

    cfg, _ = _staged(tmp_path, [*FV, override])
    with pytest.raises(ValueError, match=message):
        train(cfg, RunPaths.from_config(cfg))


# --- End to end -------------------------------------------------------------


def test_a_run_records_both_terms_in_its_history(tmp_path):
    from naviernet.training import train
    from naviernet.utils.paths import RunPaths

    cfg, _ = _staged(tmp_path, FV)
    _, _, state = train(cfg, RunPaths.from_config(cfg))

    assert {"fv_normal", "fv_apex"} <= set(state["hist"][0])


def test_a_run_with_it_off_records_nothing_and_measures_nothing(tmp_path):
    """Inert when disabled: no plan is built and no term reaches the history."""
    from naviernet.training import _front_velocity_plan, train
    from naviernet.utils.paths import RunPaths

    cfg, _ = _staged(tmp_path, GEO)
    _, data, state = train(cfg, RunPaths.from_config(cfg))

    assert _front_velocity_plan(cfg, data, "cpu") is None
    assert not [key for key in state["hist"][0] if key.startswith("fv_")]
