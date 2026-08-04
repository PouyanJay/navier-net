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
    """
    scale = _Scale.of(cfg, data)
    return {
        "front_geometry": bool(getattr(model, "front_geometry", False)),
        "nose_speed": _nose_speed(cfg, model, data, scale),
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


def _nose_speed(cfg, model, data, scale: _Scale) -> dict:
    """How fast the bubble's leading edge advances, model and measured.

    The model side is the derivative of the same continuous trajectory
    ``trajectory.json`` records, so the two artifacts are consistent by
    construction: this chart is literally the slope of that one. It needs no
    explicit front -- the nose is read off the predicted mask -- so it is the one
    block every run has.
    """
    predicted = nose_trajectory(cfg, model, data)
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
