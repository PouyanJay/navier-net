"""The nose-shape journey (`.claude/plans/nose-shape-plan.md`).

T0: the nose measurement. With the root freed and supervised (PR #53), the
NOSE cap is the one remaining piece of the interface that is a hard-coded
perfect circle -- and the measured head is genuinely non-circular, and only
late: the nose-window bluntness drifts 0.61 -> 1.83 through the Series-1
event, crossing 1.0, which means the head OVERHANGS (wider a quarter-window
in than at the window edge -- a mushroomed head no circle can draw at any
radius). These tests pin the measurement itself: exact on shapes whose nose
geometry is known analytically, then the measured table on the real masks.

The machinery is the root's own (`fit_root_window` mirrored), so the nose
module is thin wrappers, and these tests focus on what is NEW: orientation
(the nose is the extent edge FARTHER from the pin), the overhang being
measurable at all, and the `physics.nose` block.
"""

from __future__ import annotations

import numpy as np
import pytest

from naviernet.physics.nose import (
    measured_nose_fit,
    model_nose_window_rms,
)
from naviernet.physics.root import ROOT_WINDOW_FRACTION, Raster, fit_root_window
from tests.conftest import make_config

# --- Analytic masks -----------------------------------------------------------
#
# The same high-resolution grid the root's analytic tests use: pixelization
# error stays far under the effects the assertions are about.

GRID_H, GRID_W = 240, 300
X_MAX, Y_MAX = 1.5, 1.2
XS = np.linspace(0.0, X_MAX, GRID_W)
YS = np.linspace(0.0, Y_MAX, GRID_H)

DISC_CENTER, DISC_RADIUS = (0.7, 0.6), 0.3


def disc_mask() -> np.ndarray:
    """A disc: its nose window lies entirely on a true circle, so the fit must
    read the circle's own signature from the RIGHT edge."""
    gx, gy = np.meshgrid(XS, YS)
    return np.hypot(gx - DISC_CENTER[0], gy - DISC_CENTER[1]) < DISC_RADIUS


def mushroom_mask() -> np.ndarray:
    """A capsule whose head is a wider, shallow bulge: the analytic overhang.

    Body: a horizontal capsule of radius 0.12 from x = 0.2 to x = 0.9. Head:
    the front half of a disc of radius 0.2 about (0.9, 0.6). The widest point
    of the head (0.2) sits INTERIOR to the nose edge while the window's far
    end reads the 0.12 body -- the mushroomed-head regime frames 9-11 measure,
    as a shape whose window profile is known exactly.
    """
    gx, gy = np.meshgrid(XS, YS)
    body_x = np.clip(gx, 0.2, 0.9)
    capsule = np.hypot(gx - body_x, gy - 0.6) < 0.12
    bulge = (np.hypot(gx - 0.9, gy - 0.6) < 0.2) & (gx > 0.86)
    return capsule | bulge


def test_the_disc_nose_reads_as_its_own_circle():
    """The mirror of the root's disc test: measured from the RIGHT edge, the
    same circle signature -- taper ~1/2, tiny residual, sub-1 bluntness."""
    fit = fit_root_window(
        Raster(disc_mask(), XS, YS), y_root=DISC_CENTER[1], root_at_left=False
    )
    assert 0.40 <= fit.taper_exponent <= 0.55
    assert fit.circle_rms < 0.03
    depth = ROOT_WINDOW_FRACTION * 2 * DISC_RADIUS
    expected = np.sqrt((0.25 * depth) * (2 * DISC_RADIUS - 0.25 * depth)) / np.sqrt(
        depth * (2 * DISC_RADIUS - depth)
    )
    assert fit.bluntness == pytest.approx(expected, abs=0.03)


def test_the_mushroom_head_reads_an_overhang():
    """Bluntness ABOVE 1 is the fact no circle can express: the head is wider
    a quarter-window in than at the window's full depth. The measurement must
    see it -- nothing in the fit may assume ratios <= 1. (The whole-window
    taper exponent stays positive here -- the log-log fit spans the tip's
    circular rise too -- which is exactly why the bluntness ratio, not the
    taper, is the overhang's own measure and the supervised scalar.)"""
    fit = fit_root_window(Raster(mushroom_mask(), XS, YS), y_root=0.6, root_at_left=False)
    assert fit.bluntness > 1.2
    # The disc, by contrast, stays safely below 1.
    disc = fit_root_window(
        Raster(disc_mask(), XS, YS), y_root=DISC_CENTER[1], root_at_left=False
    )
    assert disc.bluntness < 0.9


# --- The mushroom fixture and the dataset wrapper -----------------------------

# The mushroom capsule: a THIN body (so the head's overhang has room inside the
# window) closed by a circular root and a bulged nose -- the front slice of a
# wider disc riding on the advancing tip. The analytic mushroomed-head dataset:
# its measured nose bluntness sits WELL above 1 at every frame while its root
# stays circular, which is what makes it both the orientation proof here and
# the T2 mechanism fixture (does the supervision actually move w_nose?).
MUSHROOM_BODY_RADIUS = 0.06
MUSHROOM_BULGE_RADIUS = 0.14
# The bulge keeps only its front slice: everything behind s + this ledge offset
# is cut away, so the head is wide but SHALLOW and the window's full-depth
# station lands on the thin body behind the ledge -- the overhang's signature.
# 0.05 keeps the ledge clear of the earliest frame's full-depth station (at a
# smaller offset that station falls ON the ledge and the ratio flips regime).
MUSHROOM_LEDGE_OFFSET = 0.05
# The fixture's own grid: twice the plain capsule's resolution, because the
# bulge's features (0.06-0.14) are a few pixels at 96x48 and the interpolated
# bluntness turns noisy exactly where the assertions are.
MUSHROOM_W_PX, MUSHROOM_H_PX = 192, 96


def mushroom_capsule_writer(path, n_frames: int = 6) -> None:
    """The mushroom-head dataset: SDF-exact (away from the corner rings, the
    blunt-root fixture's own accepted caveat) via min/max of primitive SDFs."""
    import json

    from tests.conftest import (
        CAPSULE_DT,
        CAPSULE_GROWTH_PER_FRAME,
        CAPSULE_S0,
        CAPSULE_T_REF_MS,
        CAPSULE_X_MAX,
        CAPSULE_X_ROOT,
        CAPSULE_Y_CENTER,
        CAPSULE_Y_MAX,
    )

    xs = np.linspace(0.0, CAPSULE_X_MAX, MUSHROOM_W_PX, dtype=np.float32)
    ys = np.linspace(0.0, CAPSULE_Y_MAX, MUSHROOM_H_PX, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)

    def frame_sdf(nose: float) -> np.ndarray:
        nearest_x = np.clip(gx, CAPSULE_X_ROOT, nose)
        body = np.hypot(gx - nearest_x, gy - CAPSULE_Y_CENTER) - MUSHROOM_BODY_RADIUS
        disc = np.hypot(gx - nose, gy - CAPSULE_Y_CENTER) - MUSHROOM_BULGE_RADIUS
        bulge = np.maximum(disc, (nose + MUSHROOM_LEDGE_OFFSET) - gx)
        return np.minimum(body, bulge)

    sdf = np.stack(
        [frame_sdf(CAPSULE_S0 + k * CAPSULE_GROWTH_PER_FRAME) for k in range(n_frames)]
    )
    alpha = (sdf < 0).astype(np.float32)
    meta = {
        "x_pin_star": CAPSULE_X_ROOT - MUSHROOM_BODY_RADIUS,
        "t_ref_ms": CAPSULE_T_REF_MS,
        "n_frames_usable": n_frames,
        "n_frames_event": n_frames,
        "frame_numbers": list(range(1, n_frames + 1)),
    }
    np.savez_compressed(
        path,
        alpha=alpha,
        sdf=sdf.astype(np.float32),
        valid=np.ones_like(alpha, dtype=np.uint8),
        masks_camera=(alpha > 0.5).astype(np.uint8),
        x_star=xs,
        y_star=ys,
        t_star=(np.arange(n_frames) * CAPSULE_DT).astype(np.float32),
        meta=json.dumps(meta),
    )


def test_measured_nose_fit_reads_the_far_edge_from_the_pin(tmp_path):
    """Orientation: the nose is the x-extent edge FARTHER from the measured pin
    anchor -- the exact complement of the root's rule. On the mushroom fixture
    only the NOSE overhangs, so reading bluntness > 1 at the nose and a
    circular sub-1 root proves the wrapper measures its actual end."""
    from naviernet.data.dataset import BubbleDataset
    from naviernet.physics.nose import measured_nose_edge
    from naviernet.physics.root import measured_root_fit
    from naviernet.utils.paths import RunPaths
    from tests.conftest import staged_run

    cfg, paths = staged_run(tmp_path, write=mushroom_capsule_writer)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    nose = measured_nose_fit(data, row=0)
    root = measured_root_fit(data, row=0)
    assert nose.bluntness > 1.1
    assert root.bluntness < 0.9
    # And the window's anchor ADVANCES with the head, unlike the root's fixed pin.
    edges = [measured_nose_edge(data, row) for row in range(data.n_event)]
    assert all(b > a for a, b in zip(edges, edges[1:], strict=False))


def test_physics_report_carries_the_nose_block(tmp_path):
    """The nose sibling of physics.root: any front-geometry run's metrics now
    measure the nose window, model against masks, per frame."""
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

    nose = report["nose"]
    assert nose["window_fraction"] == ROOT_WINDOW_FRACTION
    assert len(nose["per_frame"]) == len(data.t)
    frame = nose["per_frame"][0]
    for key in ("taper_exponent", "circle_rms", "bluntness", "model_nose_rms"):
        assert np.isfinite(frame[key]), key
    assert nose["model_nose_rms_last"] >= 0.0
    # The summary carries the drift's endpoints -- the bench gate's numbers.
    assert np.isfinite(nose["bluntness_first"])
    assert np.isfinite(nose["bluntness_last"])


def test_model_nose_rms_reads_the_nose_cap_not_the_root(tmp_path):
    """The distance is normalised by and sampled at the NOSE cap: a model whose
    nose sits exactly on the measured interface reads ~0 whatever the root
    does. The capsule fixture's first frame IS the init capsule's target, so
    the number is finite and bounded by the init mismatch."""
    import torch

    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths
    from tests.conftest import staged_capsule_run

    cfg, paths = staged_capsule_run(tmp_path, ["model.front_geometry=true"])
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    torch.manual_seed(0)
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    rms = model_nose_window_rms(model, data, row=0)
    assert np.isfinite(rms)
    assert 0.0 <= rms < 1.0


# --- Series-1: the measurement this journey is built on -----------------------


@pytest.mark.needs_data
class TestSeries1NoseDrift:
    """Plan §0 fact 1, re-measured by the shipped module: the head starts
    near-circular, blunts through the event, and OVERHANGS by the last frames
    (bluntness crossing 1.0) -- and the facts survive threshold and window
    perturbation (plan §8)."""

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

    def test_the_measured_table_reproduces(self, data):
        """The plan's own numbers, by frame: bluntness 0.605 (f1) -> 1.833
        (f11), circle RMS 0.022 -> 0.189."""
        by_frame = {
            int(frame): measured_nose_fit(data, row)
            for row, frame in enumerate(data.frame_numbers)
        }
        assert by_frame[1].bluntness == pytest.approx(0.605, abs=0.05)
        assert by_frame[5].bluntness == pytest.approx(0.982, abs=0.05)
        assert by_frame[11].bluntness == pytest.approx(1.833, abs=0.08)
        assert by_frame[1].circle_rms == pytest.approx(0.022, abs=0.02)
        assert by_frame[11].circle_rms == pytest.approx(0.189, abs=0.03)

    def test_the_overhang_appears_and_the_drift_is_late(self, data):
        """The head is near-circular EARLY and overhangs LATE -- the two facts
        that scope the feature (frames 9-11 are where the model's error is)."""
        fits = [measured_nose_fit(data, row) for row in range(data.n_event)]
        assert fits[0].bluntness < 0.75
        assert fits[-1].bluntness > 1.0  # the overhang no circle can draw
        assert fits[-1].circle_rms > fits[0].circle_rms + 0.05

    def test_the_drift_survives_threshold_and_window_perturbation(self, data):
        for level in (0.4, 0.5, 0.6):
            for window in (0.20, 0.25, 0.30):
                fits = [
                    measured_nose_fit(data, row, window=window, alpha_level=level)
                    for row in range(data.n_event)
                ]
                assert fits[-1].bluntness > fits[0].bluntness + 0.3, (level, window)
                assert fits[-1].bluntness > 1.0, (level, window)
