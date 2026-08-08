"""The nucleation root journey (`.claude/plans/nucleation-root-plan.md`).

T0: the root measurement module. The diagnostics' profile stations deliberately
exclude the caps, so nothing existing can see the measured fact this journey is
built on -- the masks disagree with a circle at the root MORE AND MORE as the
event runs (taper exponent p drifting 0.5 -> 0.38, circle residual 12% -> 32%).
These tests pin the measurement itself: exact on shapes whose root geometry is
known analytically, then directional on the real Series-1 masks.
"""

from __future__ import annotations

import numpy as np
import pytest

from naviernet.physics.root import (
    ROOT_WINDOW_FRACTION,
    fit_root_window,
    measured_root_fit,
)
from tests.conftest import make_config

# --- Analytic masks -----------------------------------------------------------
#
# High-resolution rasters, so pixelization error stays far under the effects the
# assertions are about (a 300-px-wide grid puts it at the percent level).

GRID_H, GRID_W = 240, 300
X_MAX, Y_MAX = 1.5, 1.2
XS = np.linspace(0.0, X_MAX, GRID_W)
YS = np.linspace(0.0, Y_MAX, GRID_H)

DISC_CENTER, DISC_RADIUS = (0.7, 0.6), 0.3


def disc_mask() -> np.ndarray:
    """A disc: its root window lies entirely on a true circle, so the fit must
    read the circle's own signature -- taper exponent 1/2, residual ~0."""
    gx, gy = np.meshgrid(XS, YS)
    return np.hypot(gx - DISC_CENTER[0], gy - DISC_CENTER[1]) < DISC_RADIUS


def slab_mask() -> np.ndarray:
    """A rectangle spanning the disc's extent: the fully squared-off limit.
    Constant width means no taper at all -- the far pole of the bluntness axis."""
    gx, gy = np.meshgrid(XS, YS)
    inside_x = (gx > DISC_CENTER[0] - DISC_RADIUS) & (gx < DISC_CENTER[0] + DISC_RADIUS)
    inside_y = (gy > DISC_CENTER[1] - DISC_RADIUS) & (gy < DISC_CENTER[1] + DISC_RADIUS)
    return inside_x & inside_y


def test_the_disc_reads_as_its_own_circle():
    fit = fit_root_window(disc_mask(), XS, YS, y_root=DISC_CENTER[1])
    # W(d) = sqrt(d (2r - d)) ~ d^{1/2} near the apex. The log-log fit runs over
    # the whole window, where the exponent is slightly below 1/2 on average.
    assert 0.40 <= fit.taper_exponent <= 0.55
    # A circle fit to circle points: residual is pixelization only.
    assert fit.circle_rms < 0.03
    # The analytic half-width ratio at quarter-window vs full-window depth.
    depth = ROOT_WINDOW_FRACTION * 2 * DISC_RADIUS
    expected = np.sqrt((0.25 * depth) * (2 * DISC_RADIUS - 0.25 * depth)) / np.sqrt(
        depth * (2 * DISC_RADIUS - depth)
    )
    assert fit.bluntness == pytest.approx(expected, abs=0.03)
    # A symmetric shape carries no bias and no tilt.
    assert abs(fit.updown_bias) < 0.01
    assert abs(fit.tilt) < 0.05


def test_the_slab_reads_blunter_than_the_disc_on_the_taper_measures():
    """Squared-off = lower taper exponent, higher half-width ratio. The circle
    residual is deliberately NOT asserted here: a window short relative to the
    half-width is radially near-equidistant from a distant centre whatever the
    shape (measured: the slab fits a circle to 0.9% radially), so the taper
    measures are the discriminators and the circle fit is kept as the
    plan-§1-comparable diagnostic only."""
    disc = fit_root_window(disc_mask(), XS, YS, y_root=DISC_CENTER[1])
    slab = fit_root_window(slab_mask(), XS, YS, y_root=DISC_CENTER[1])
    assert slab.taper_exponent < 0.15 < disc.taper_exponent
    assert slab.bluntness > 0.9 > disc.bluntness


def test_the_fit_is_orientation_agnostic():
    """A mirrored mask with the root on the RIGHT must measure the same root:
    the root side is chosen by the anchor, not by an assumed orientation."""
    mirrored = disc_mask()[:, ::-1]
    left = fit_root_window(disc_mask(), XS, YS, y_root=DISC_CENTER[1])
    right = fit_root_window(
        mirrored, XS, YS, y_root=DISC_CENTER[1], root_at_left=False
    )
    assert right.taper_exponent == pytest.approx(left.taper_exponent, abs=1e-6)
    assert right.circle_rms == pytest.approx(left.circle_rms, abs=1e-6)
    assert right.bluntness == pytest.approx(left.bluntness, abs=1e-6)


def test_a_vapour_free_mask_fails_loudly():
    with pytest.raises(ValueError, match="no vapour"):
        fit_root_window(np.zeros((GRID_H, GRID_W), dtype=bool), XS, YS, y_root=0.5)


def test_a_window_too_thin_to_fit_reads_nan_not_garbage():
    """Two usable columns cannot support a log-log fit or a circle; the fit says
    so with NaN rather than inventing an exponent."""
    mask = np.zeros((GRID_H, GRID_W), dtype=bool)
    mask[100:140, 10:13] = True  # 3 columns: apex + 2 -- under the minimum
    fit = fit_root_window(mask, XS, YS, y_root=float(YS[120]))
    assert np.isnan(fit.taper_exponent)
    assert np.isnan(fit.circle_rms)


# --- The dataset wrapper and the metrics block --------------------------------


def test_measured_root_fit_reads_the_capsule_fixture(tmp_path):
    """End-to-end over the staged dataset: the analytic capsule's root cap is a
    true circle, and the wrapper must find it through BubbleDataset."""
    from naviernet.data.dataset import BubbleDataset
    from naviernet.utils.paths import RunPaths
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(tmp_path)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    fit = measured_root_fit(data, row=0)
    # The default window (25% of extent) overshoots this short capsule's cap
    # onto its constant-width body, so the log-log slope reads between the
    # circle's 1/2 and the slab's 0 -- the honest full-window value.
    assert 0.15 <= fit.taper_exponent <= 0.45
    assert fit.circle_rms < 0.12
    assert 0.55 <= fit.bluntness <= 0.9


def test_physics_report_carries_the_root_block(tmp_path):
    """The blind spot closed: any front-geometry run's metrics now measure the
    root window, model against masks, per frame."""
    import torch

    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.physics.diagnostics import physics_report
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(tmp_path, ["model.front_geometry=true"])
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    torch.manual_seed(0)
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    report = physics_report(model, data)

    root = report["root"]
    assert root["window_fraction"] == ROOT_WINDOW_FRACTION
    assert len(root["per_frame"]) == len(data.t)
    frame = root["per_frame"][0]
    for key in (
        "taper_exponent",
        "circle_rms",
        "bluntness",
        "model_root_rms",
    ):
        assert np.isfinite(frame[key]), key
    # The capsule model opens circular AND the capsule data is circular at the
    # root, so the model-vs-mask distance is small but honest (init mismatch).
    assert root["model_root_rms_last"] >= 0.0


# --- Series-1: the measurement this journey is built on -----------------------


@pytest.mark.needs_data
class TestSeries1RootDrift:
    """Plan §1, re-measured by the shipped module: the deviation from a circle
    is radial bluntness, it GROWS through the event, and both facts survive
    segmentation-threshold and window-size perturbation (plan §8.2)."""

    @pytest.fixture()
    def data(self):
        from naviernet.data.dataset import BubbleDataset
        from naviernet.utils.paths import RunPaths

        cfg = make_config(
            ["dataset=Series-1", "training.holdout_frame=-1", "training.val_fraction=0"]
        )
        paths = RunPaths.from_config(cfg)
        if not paths.tensors.exists():
            pytest.skip("Series-1 tensors not preprocessed")
        return BubbleDataset(cfg, paths, device="cpu")

    def test_bluntness_grows_and_the_circle_fit_degrades(self, data):
        rows = list(range(data.n_event))
        fits = [measured_root_fit(data, row) for row in rows]
        early = np.nanmean([f.taper_exponent for f in fits[:2]])
        late = np.nanmean([f.taper_exponent for f in fits[-2:]])
        assert late < early - 0.05, (early, late)
        assert fits[-1].circle_rms > fits[0].circle_rms + 0.05
        # Up/down symmetric to first order (plan §1.1): tiny bias, tiny tilt.
        assert all(abs(f.updown_bias) < 0.02 for f in fits)

    def test_the_drift_survives_threshold_and_window_perturbation(self, data):
        for level in (0.4, 0.5, 0.6):
            for window in (0.20, 0.25, 0.30):
                fits = [
                    measured_root_fit(data, row, window=window, alpha_level=level)
                    for row in range(data.n_event)
                ]
                early = np.nanmean([f.taper_exponent for f in fits[:2]])
                late = np.nanmean([f.taper_exponent for f in fits[-2:]])
                assert late < early, (level, window, early, late)
