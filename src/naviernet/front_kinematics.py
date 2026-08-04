"""The front-velocity report: how fast the interface moved, and where on it.

The run's artifacts say a great deal about the interface's SHAPE and almost
nothing about its RATE. ``trajectory.json`` carries the nose's position; every
other record is a static agreement score. Yet the rate is the quantity that
carries a front-geometry solution past the last supervised frame, and the one a
reader has to differentiate a chart by eye to recover.

This module measures it, from both sides:

- what the MODEL thinks -- :attr:`FrontSamples.normal_speed` and the apex's own
  velocity, taken analytically from the parameterisation rather than differenced;
- what the MASKS say -- the level-set estimate in :mod:`naviernet.data.velocimetry`.

**Why an output and not (only) a loss.** Supervising the model with this
measurement benched neutral: ``v_n`` is a time derivative of the same masks the
shape term already fits to ~0.93 IoU, so it adds no information to the
optimisation. It adds a great deal to a *reader*, who has none of it. That
asymmetry is the whole justification for this module.

**What is and is not observable.** Two masks cannot give a surface point's
material velocity -- a curve sliding along itself looks identical between frames,
so the tangential component is unobservable. That costs nothing for the shape,
which only the normal component moves, but it does mean every arrow and every
profile here is a normal projection and must be labelled as one. The nose apex is
the exception: it is geometrically distinguishable, so its full 2-D velocity is
honest, and it is reported separately for exactly that reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from naviernet.evaluation import (
    measured_trajectory,
    nose_trajectory,
    physical_series,
    reference_length_um,
)
from naviernet.physics.front_velocity import (
    SEGMENT_NAMES,
    FrontSegments,
    bin_count,
    build_plan,
    front_segments,
    off_nose_cap,
    sample_measured,
    time_index,
)
from naviernet.utils.logging import get_logger

log = get_logger(__name__)

# Digits kept on a reported speed. The µm/ms axis runs O(0.1-100) for these
# datasets, so four decimals is well past the measurement's own precision and
# well short of dumping float noise into an artifact a human reads.
SPEED_DIGITS = 4


def write_report(cfg, model, data, path: Path) -> dict:
    """Build this dataset's front-velocity report and persist it as JSON.

    Written beside ``trajectory.json`` and in the same physical units, so the
    speed charts can be read as the slopes of the position charts.
    """
    report = build_report(cfg, model, data)
    path.write_text(json.dumps(report, allow_nan=False))
    log.info("front-velocity report written to %s", path)
    return report


def build_report(cfg, model, data) -> dict:
    """Everything the results view charts about the front's motion.

    ``front_geometry`` is not decoration: without an explicit front there is no
    ``normal_speed`` and no apex to differentiate, so the blocks that need one
    are ``None`` and the consumer says why rather than rendering an empty axis.

    Every block shares ONE continuous time axis -- the trajectory's own -- so the
    charts can be scrubbed together and a reader comparing two of them is
    comparing the same instants.
    """
    scale = _Scale.of(cfg, data)
    predicted = nose_trajectory(cfg, model, data)
    front_geometry = bool(getattr(model, "front_geometry", False))
    return {
        "front_geometry": front_geometry,
        "nose_speed": _nose_speed(cfg, data, scale, predicted),
        "apex": _apex_velocity(model, data, scale, predicted.times) if front_geometry else None,
        "profile": _profile(cfg, model, data, scale, predicted.times)
        if front_geometry
        else None,
    }


class _Scale:
    """The dataset's own conversion from non-dimensional units to physical ones.

    A speed is a length over a time, so it needs both references at once and
    every block here needs the same pair -- read once, from the dataset that
    recorded them, rather than re-derived per block.
    """

    def __init__(self, l_ref_um: float, t_ref_ms: float) -> None:
        self.l_ref_um = l_ref_um
        self.t_ref_ms = t_ref_ms

    @classmethod
    def of(cls, cfg, data) -> _Scale:
        return cls(reference_length_um(cfg, data), float(data.meta["t_ref_ms"]))

    @property
    def speed(self) -> float:
        """x* per t* -> µm per ms."""
        return self.l_ref_um / self.t_ref_ms

    def speeds(self, values) -> list[float | None]:
        """A non-dimensional speed series in µm/ms, NaN dropped to ``None``."""
        return physical_series(values, self.speed, SPEED_DIGITS)

    def times(self, values) -> list[float | None]:
        return physical_series(values, self.t_ref_ms, 4)


def _nose_speed(cfg, data, scale: _Scale, predicted) -> dict:
    """How fast the bubble's leading edge advances, model and measured.

    The model side is the derivative of the same continuous trajectory
    ``trajectory.json`` records, so the two artifacts are consistent by
    construction: this chart is literally the slope of that one. It needs no
    explicit front -- the nose is read off the predicted mask -- so it is the one
    block every run has.
    """
    return {
        "t_ms": scale.times(predicted.times),
        "v_um_per_ms": scale.speeds(np.gradient(predicted.nose, predicted.times)),
        "measured": _measured_nose_speed(cfg, data, scale),
    }


def _measured_nose_speed(cfg, data, scale: _Scale) -> dict:
    """The camera's own answer: one finite difference per consecutive frame pair.

    Deliberately not ``np.gradient`` over the whole series. An excluded camera
    frame leaves a real gap, and a central difference would quietly bridge it --
    reporting a coarser estimate of a different interval as though it were the
    same measurement. :attr:`BubbleDataset.report_pairs` is the set of intervals
    that honestly exist.

    Each value is plotted at the midpoint of the interval it measures, not at
    either endpoint: a difference describes the span, and anchoring it to an end
    would sit every measured point half a frame away from the continuous curve
    it is read against.
    """
    measured = measured_trajectory(cfg, data)  # times already in ms
    pairs = data.report_pairs

    times, speeds, heldout = [], [], []
    for row, nxt in pairs:
        interval = measured.times[nxt] - measured.times[row]
        times.append(0.5 * (measured.times[row] + measured.times[nxt]))
        # x* per ms -> µm per ms: only the LENGTH needs referencing, the time
        # axis is already physical here.
        speeds.append((measured.nose[nxt] - measured.nose[row]) / interval)
        heldout.append(data.spans_held_out((row, nxt)))
    return {
        "t_ms": physical_series(times, 1.0, 4),
        "v_um_per_ms": physical_series(speeds, scale.l_ref_um, SPEED_DIGITS),
        "heldout": heldout,
    }


def _apex_velocity(model, data, scale: _Scale, times) -> dict:
    """The nose apex's full 2-D velocity -- the one honest ``(vx, vy)`` here.

    Everywhere else on the interface only the normal component is observable: a
    curve sliding along itself looks identical between frames, so a generic point
    has no frame-to-frame correspondence to track. The apex is geometrically
    distinguishable, so it does, and both components mean something.

    The model side is taken by AUTODIFF rather than differenced. Under this
    parameterisation the apex is a differentiable function of time, so its exact
    velocity is available and there is no reason to report an estimate whose
    error depends on a step size nobody chose.
    """
    positions, velocity = _apex_path(model, data, times)
    return {
        "t_ms": scale.times(times),
        "x_um": physical_series(positions[:, 0], scale.l_ref_um, 2),
        "y_um": physical_series(positions[:, 1], scale.l_ref_um, 2),
        "vx_um_per_ms": scale.speeds(velocity[:, 0]),
        "vy_um_per_ms": scale.speeds(velocity[:, 1]),
        "measured": _measured_apex_velocity(data, scale),
    }


def _apex_path(model, data, times) -> tuple[np.ndarray, np.ndarray]:
    """The apex's position and its exact velocity at each of ``times``."""
    import torch

    t = torch.tensor(
        np.asarray(times, dtype=np.float32).reshape(-1, 1), device=data.device
    ).requires_grad_(True)
    apex = model.apex(t)
    # One grad call per component: `apex` is (N, 2) and each column is a
    # separate function of t. Summing within a column is safe because sample i
    # depends only on t_i, so the sum's gradient is each sample's own.
    velocity = [
        torch.autograd.grad(apex[:, axis].sum(), t, retain_graph=axis == 0)[0]
        for axis in (0, 1)
    ]
    return (
        apex.detach().cpu().numpy(),
        torch.cat(velocity, dim=1).detach().cpu().numpy(),
    )


def _profile(cfg, model, data, scale: _Scale, times) -> dict:
    """The normal-speed profile along the front: one ``s`` axis, two views of it.

    ``times`` holds one model-against-measured profile per reportable frame pair
    -- the only instants a measurement exists at. ``kymograph`` is the model's own
    profile over the continuous time axis, which needs no measurement and so can
    be dense: the whole history of the front's motion as one image.
    """
    grid = _ProfileGrid.of(cfg)
    return {
        "s": [round(value, 5) for value in grid.centres],
        "segments": grid.segment_records(),
        "times": _measured_profiles(cfg, model, data, scale, grid),
        "kymograph": _kymograph(model, data, scale, grid, times),
    }


def _measured_profiles(cfg, model, data, scale: _Scale, grid: _ProfileGrid) -> list[dict]:
    """One model-against-measured profile per reportable frame pair.

    Empty when the dataset offers no consecutive pair -- a single-frame series,
    or one whose camera frames are all non-adjacent. There is nothing to compare
    against then, and the kymograph beside it still carries the model's own
    answer.

    The model front is evaluated at each pair's FIRST frame, which is where the
    level-set estimate lives (``v_n`` is the speed of the ``sdf_now`` interface).
    The same convention the loss uses, so this chart shows the quantity the term
    actually compared.
    """
    if not data.report_pairs:
        log.info(
            "no consecutive frame pairs in %s: reporting the model's own profile "
            "with nothing measured to compare it against",
            data.cfg.dataset,
        )
        return []

    plan = build_plan(data, cfg, data.device, pairs=data.report_pairs)
    model_speed, measured = _profile_rows(model, plan, grid, scale)
    return [
        {
            "t_ms": round(float(plan.times[i]) * scale.t_ref_ms, 4),
            "frames": [data.frame_numbers[row], data.frame_numbers[nxt]],
            "heldout": data.spans_held_out((row, nxt)),
            "model": model_speed[i],
            "measured": measured[i],
        }
        for i, (row, nxt) in enumerate(plan.pairs)
    ]


def _kymograph(model, data, scale: _Scale, grid: _ProfileGrid, times) -> dict:
    """The model's normal speed over (position, time) -- the whole history at once.

    Model only, and necessarily so: the measurement exists at frame pairs alone,
    while the parameterisation is continuous in time. Rendering nine measured
    rows as an image beside a dense model one would invite a comparison that the
    sampling does not support; the profile chart is where the two meet.
    """
    import torch

    t = torch.tensor(np.asarray(times, dtype=np.float32).reshape(-1, 1), device=data.device)
    front = model.front(t, grid.n_body, grid.n_cap)
    row = time_index(t, front.points[:, 2:3]).cpu().numpy()
    flat = row * grid.total + grid.bin_of(front_segments(front))
    speed = front.normal_speed.detach().cpu().numpy().reshape(-1)
    binned = _bin_mean(speed, flat, len(times) * grid.total).reshape(len(times), grid.total)
    return {
        "t_ms": scale.times(times),
        "v_um_per_ms": [scale.speeds(values) for values in binned],
    }


class _ProfileGrid:
    """The fixed ``s`` axis every profile is binned onto.

    ``s`` is ONE traversal of the closed front: the root cap from its lower seam
    to its upper one, up the upper body to the nose, around the nose cap, and
    back down the lower body. That ordering is what makes the profile a curve
    rather than four disconnected plots, and the kymograph an image rather than
    four strips.

    The segment boundaries are fixed fractions of the axis -- each segment's
    share of the total bin count -- rather than true arclength. Arclength grows
    with the bubble, which would move every boundary from frame to frame and
    leave the kymograph with no stable vertical axis.
    """

    # The traversal, and how each segment's own parameter maps onto it. The caps
    # sweep from the lower seam to the upper one, so the nose cap -- entered from
    # the upper body -- runs backwards.
    ORDER = (
        ("root_cap", False),
        ("upper_body", False),
        ("nose_cap", True),
        ("lower_body", True),
    )

    def __init__(self, n_body: int, n_cap: int, per_bin: int) -> None:
        self.n_body, self.n_cap = n_body, n_cap
        bins_body, bins_cap = bin_count(n_body, per_bin), bin_count(n_cap, per_bin)
        self.bins = {"upper_body": bins_body, "lower_body": bins_body}
        self.bins.update({"root_cap": bins_cap, "nose_cap": bins_cap})
        self.total = sum(self.bins[name] for name, _ in self.ORDER)
        self.starts: dict[str, int] = {}
        offset = 0
        for name, _ in self.ORDER:
            self.starts[name] = offset
            offset += self.bins[name]

    @classmethod
    def of(cls, cfg) -> _ProfileGrid:
        """The front's own sample counts, binned by the rule the loss bins by --
        so a reader of this chart is looking at the quantity the term compared,
        not a differently-resolved cousin of it."""
        return cls(
            int(cfg.model.front_body_samples),
            int(cfg.model.front_cap_samples),
            int(cfg.training.fv_samples_per_bin),
        )

    @property
    def centres(self) -> list[float]:
        """Each bin's centre on the 0-1 traversal."""
        return [(i + 0.5) / self.total for i in range(self.total)]

    def segment_records(self) -> list[dict]:
        """Where each segment sits on the axis, and whether the MEASUREMENT is
        trustworthy there -- the nose cap is the one place it is not."""
        return [
            {
                "name": name,
                "bin_start": self.starts[name],
                "bin_end": self.starts[name] + self.bins[name],
                "s_start": self.starts[name] / self.total,
                "s_end": (self.starts[name] + self.bins[name]) / self.total,
                "measured": name != "nose_cap",
            }
            for name, _ in self.ORDER
        ]

    def bin_of(self, segments: FrontSegments) -> np.ndarray:
        """Each front sample's bin on the traversal, ``(N,)``."""
        index = segments.index.cpu().numpy()
        position = segments.position.cpu().numpy()
        bins = np.zeros(index.shape, dtype=np.int64)
        for name, backwards in self.ORDER:
            selected = index == SEGMENT_NAMES.index(name)
            span = self.bins[name]
            along = position[selected]
            within = np.clip(
                ((1.0 - along if backwards else along) * span).astype(int), 0, span - 1
            )
            bins[selected] = self.starts[name] + within
        return bins


def _profile_rows(model, plan, grid: _ProfileGrid, scale: _Scale):
    """One binned model profile and one binned measured profile per frame pair.

    The model's own speed is reported over the WHOLE front, including the nose
    cap. Only the measurement is untrustworthy there; blanking the model too
    would hide the very place the two estimators disagreed, which is the finding
    that shaped this feature.
    """
    front = model.front(plan.times, plan.n_body, plan.n_cap)
    pair = time_index(plan.times, front.points[:, 2:3])
    measured, usable = sample_measured(plan, front.points, pair)
    # The same validity rule the loss applies, from the same function.
    usable = usable * off_nose_cap(front)

    bins = grid.bin_of(front_segments(front))
    flat = pair.cpu().numpy() * grid.total + bins
    n_rows = len(plan.pairs)
    cells = n_rows * grid.total

    model_mean = _bin_mean(front.normal_speed.detach().cpu().numpy().reshape(-1), flat, cells)
    usable_np = usable.detach().cpu().numpy().reshape(-1)
    measured_mean = _bin_mean(
        measured.detach().cpu().numpy().reshape(-1), flat, cells, weights=usable_np
    )
    return (
        [scale.speeds(row) for row in model_mean.reshape(n_rows, grid.total)],
        [scale.speeds(row) for row in measured_mean.reshape(n_rows, grid.total)],
    )


def _bin_mean(values, bins, n_bins: int, weights=None) -> np.ndarray:
    """``values`` averaged into ``n_bins`` bins; NaN where a bin took none.

    NaN rather than zero: a bin nothing landed in has no measurement, and zero
    is a perfectly plausible speed that a reader would take at face value.
    """
    weights = np.ones_like(values) if weights is None else weights
    counts = np.bincount(bins, weights=weights, minlength=n_bins)
    totals = np.bincount(bins, weights=values * weights, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / counts, np.nan)


def _measured_apex_velocity(data, scale: _Scale) -> dict:
    """What the masks say the apex did, over each consecutive frame pair.

    ``front_apex`` is the segmented front edge and the centre of the vapour
    column standing at it -- a tracked feature, so this is a displacement, with
    no level-set approximation anywhere in it.
    """
    times, vx, vy, heldout = [], [], [], []
    for row, nxt in data.report_pairs:
        interval = float(data.t[nxt]) - float(data.t[row])
        here, there = data.front_apex(row), data.front_apex(nxt)
        times.append(0.5 * (float(data.t[row]) + float(data.t[nxt])))
        vx.append((there[0] - here[0]) / interval)
        vy.append((there[1] - here[1]) / interval)
        heldout.append(data.spans_held_out((row, nxt)))
    return {
        "t_ms": scale.times(times),
        "vx_um_per_ms": scale.speeds(vx),
        "vy_um_per_ms": scale.speeds(vy),
        "heldout": heldout,
    }
