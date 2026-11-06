"""Pure helpers: probe resampling, contour arclength, CSV and NaN handling."""

from __future__ import annotations

import math

import numpy as np
from naviernet_studio import exports, geometry, probes


def test_resample_polyline_spaces_points_evenly_in_arclength():
    stations = probes.resample_polyline(np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]]), 8)

    assert stations.s_um[0] == 0.0 and abs(stations.s_um[-1] - 7.0) < 1e-9
    steps = np.diff(stations.s_um)
    assert np.allclose(steps, steps[0])
    assert abs(stations.x_um[-1] - 3.0) < 1e-9 and abs(stations.y_um[-1] - 4.0) < 1e-9


def test_loop_arclength_starts_at_the_root_and_runs_along_the_upper_edge():
    angles = np.linspace(0, 2 * math.pi, 65)[:-1]
    ring = np.column_stack([100 + 50 * np.cos(angles), 150 + 20 * np.sin(angles)])

    vertices, s = geometry.loop_arclength(ring, y_center_um=150.0)

    assert vertices[0, 0] == ring[:, 0].min()
    assert vertices[1, 1] > 150.0
    assert s[0] == 0.0 and s[-1] > 0.0 and np.all(np.diff(s) >= 0)


def test_csv_text_writes_provenance_as_comments_and_blanks_for_nan():
    text = exports.csv_text(
        ["a", "b"],
        [[1.0, float("nan")], [2.5, 3.0]],
        {"run": "r", "quantities": {"a": {"label": "A", "unit": "Pa", "how": "model"}}},
    )

    lines = text.splitlines()
    assert lines[0] == "# run: r"
    assert lines[1].startswith("# a: A [Pa]")
    assert lines[2] == "a,b"
    assert lines[3] == "1.0,"


def test_without_nan_turns_non_finite_values_into_null():
    assert exports.without_nan(
        {"x": [1.0, float("nan"), float("inf")], "y": {"z": float("-inf")}}
    ) == {
        "x": [1.0, None, None],
        "y": {"z": None},
    }
