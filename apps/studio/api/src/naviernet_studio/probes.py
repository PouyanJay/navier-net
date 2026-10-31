"""Probe geometry: where a line, a cross-section or a set of points samples
the fields, in model coordinates, for one instant."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, Field

from naviernet_studio import geometry
from naviernet_studio.scene import Scene

SectionKind = Literal[
    "centreline",
    "wall_top",
    "wall_bottom",
    "gap_top",
    "gap_bottom",
    "mid_gap_top",
    "mid_gap_bottom",
]
ProbeKind = Literal["line", "points", "section"]

DEFAULT_SAMPLES = 200
MAX_SAMPLES = 2000
MAX_POINTS = 500
# Columns searched for the bubble edge when a section is tied to it.
EDGE_SEARCH_ROWS = 801
DEFAULT_BEHIND_NOSE_UM = 150.0
DEFAULT_MID_GAP_SPAN_UM = (450.0, 40.0)


class ProbeRequest(BaseModel):
    kind: ProbeKind
    quantities: list[str] = Field(min_length=1)
    frames: list[int] | None = None
    t_ms: list[float] | None = None
    # line: polyline vertices; points: the points themselves (µm).
    points_um: list[tuple[float, float]] | None = None
    n: int = Field(default=DEFAULT_SAMPLES, ge=2, le=MAX_SAMPLES)
    section: SectionKind | None = None
    # gap sections: distance behind the nose of the cross-section (µm).
    behind_nose_um: float = DEFAULT_BEHIND_NOSE_UM
    # mid-gap lines: from .. to µm behind the nose.
    span_um: tuple[float, float] = DEFAULT_MID_GAP_SPAN_UM
    # gap sections: how far from the edge into the gap to start (fraction).
    edge_offset: float = Field(default=0.0, ge=0.0, le=0.9)

    def instants(self, scene: Scene) -> list[tuple[int | None, float]]:
        """``(frame, t_star)`` pairs the request names."""
        out: list[tuple[int | None, float]] = []
        for frame in self.frames or []:
            out.append((frame, scene.t_star_of(frame, None)))
        for t in self.t_ms or []:
            t_star = scene.t_star_of(None, t)
            out.append((scene.frame_of(t_star), t_star))
        if not out:
            raise LookupError("name at least one frame or t_ms")
        return out


class Stations:
    """Sample positions in µm with their along-probe coordinate."""

    def __init__(self, x_um: np.ndarray, y_um: np.ndarray, s_um: np.ndarray, label: str):
        self.x_um, self.y_um, self.s_um, self.label = x_um, y_um, s_um, label

    def star(self, scene: Scene) -> tuple[np.ndarray, np.ndarray]:
        return self.x_um / scene.scales.length_um, self.y_um / scene.scales.length_um


def resample_polyline(vertices_um: np.ndarray, n: int) -> Stations:
    """``n`` points spaced evenly in arclength along the polyline."""
    steps = np.linalg.norm(np.diff(vertices_um, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(cumulative[-1])
    s = np.linspace(0.0, total, n)
    x = np.interp(s, cumulative, vertices_um[:, 0])
    y = np.interp(s, cumulative, vertices_um[:, 1])
    return Stations(x, y, s, "arclength")


def _edge_y_star(scene: Scene, x_star: float, t_star: float, upper: bool) -> float | None:
    """The bubble's edge in the column ``x_star``: the largest (upper) or
    smallest (lower) y with vapour, or None when the column holds none."""
    data = scene.data
    yc = 0.5 * float(data.y_lo + data.y_hi)
    ys = (
        np.linspace(yc, float(data.y_hi), EDGE_SEARCH_ROWS)
        if upper
        else np.linspace(float(data.y_lo), yc, EDGE_SEARCH_ROWS)
    )
    points = torch.tensor(
        np.column_stack([np.full_like(ys, x_star), ys, np.full_like(ys, t_star)]),
        dtype=torch.float32,
    )
    model = scene.model
    with scene.lock, torch.no_grad():
        if getattr(model, "front_geometry", False):
            inside = model._phi_with_frozen_geometry(points, None).numpy().ravel() > 0
        else:
            inside = model.alpha(points).numpy().ravel() > geometry.INTERFACE_LEVEL
    if not inside.any():
        return None
    return float(ys[inside].max() if upper else ys[inside].min())


def stations_for(scene: Scene, request: ProbeRequest, t_star: float) -> Stations:  # noqa: PLR0911
    """Where the probe samples at this instant."""
    scales, data = scene.scales, scene.data
    L = scales.length_um
    if request.kind == "points":
        pts = np.asarray(request.points_um or [], dtype=float)
        if len(pts) == 0 or len(pts) > MAX_POINTS:
            raise ValueError(f"points need 1 to {MAX_POINTS} entries")
        return Stations(pts[:, 0], pts[:, 1], np.arange(len(pts), dtype=float), "index")
    if request.kind == "line":
        pts = np.asarray(request.points_um or [], dtype=float)
        if len(pts) < 2:
            raise ValueError("a line needs at least two vertices")
        return resample_polyline(pts, request.n)
    section = request.section
    x_lo, x_hi = float(data.domain.x_min) * L, float(data.domain.x_max) * L
    y_lo, y_hi = float(data.y_lo) * L, float(data.y_hi) * L
    yc = 0.5 * (y_lo + y_hi)
    if section == "centreline":
        return resample_polyline(np.array([[x_lo, yc], [x_hi, yc]]), request.n)
    if section == "wall_top":
        return resample_polyline(np.array([[x_lo, y_hi], [x_hi, y_hi]]), request.n)
    if section == "wall_bottom":
        return resample_polyline(np.array([[x_lo, y_lo], [x_hi, y_lo]]), request.n)
    nose_um = geometry.interface(scene, t_star)["nose_um"]
    if not np.isfinite(nose_um):
        raise LookupError("no bubble at this instant, so no section tied to its nose")
    upper = section in ("gap_top", "mid_gap_top")
    if section in ("gap_top", "gap_bottom"):
        x_um = nose_um - request.behind_nose_um
        edge = _edge_y_star(scene, x_um / L, t_star, upper)
        wall = y_hi if upper else y_lo
        start = yc if edge is None else edge * L + request.edge_offset * (wall - edge * L)
        stations = resample_polyline(np.array([[x_um, start], [x_um, wall]]), request.n)
        stations.label = "distance from the bubble edge"
        return stations
    far, near = request.span_um
    xs_um = np.linspace(nose_um - far, nose_um - near, request.n)
    mids = []
    for x_um in xs_um:
        edge = _edge_y_star(scene, x_um / L, t_star, upper)
        wall = y_hi if upper else y_lo
        mids.append(0.5 * ((edge * L) + wall) if edge is not None else 0.5 * (yc + wall))
    stations = Stations(xs_um, np.asarray(mids), nose_um - xs_um, "distance behind the nose")
    return stations
