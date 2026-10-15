"""The seed-ensemble test: do independently trained runs agree on every field?

A hidden field determined by the images and the equations comes out the same
whatever the random seed; one that is not keeps the random shape it was born
with, and the 2026-10 audit found exactly that in the published checkpoint. This
script reads the field snapshots `evaluate` writes (audit_fields.npz) for
several runs of one configuration and reports, per field and frame, the
correlation of the whole field and of its symmetry-breaking part between every
pair of runs, against the `seed_corr_min` threshold the runs' audits carry.

Usage (from the repo root):
    .venv/bin/python scripts/audit_ensemble.py --out outputs/ensemble-trust-wp1.json \
        trust-wp1-s0 trust-wp1-s1 trust-wp1-s2
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

HIDDEN_FIELDS = ("u", "v", "p", "T", "s")
ODD_FIELDS = frozenset({"v"})


def breaking_part(field: np.ndarray, odd: bool) -> np.ndarray:
    """The symmetry-breaking part of a field on a grid symmetric about its middle row."""
    mirrored = field[::-1, :]
    return 0.5 * (field + mirrored) if odd else 0.5 * (field - mirrored)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two arrays; 1.0 when both are constant and equal."""
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    if scale < 1e-24:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(a @ b / scale)


def compare_runs(snapshots: dict[str, dict[str, np.ndarray]]) -> dict:
    """Per field and frame, the pairwise correlations of the full field and of
    its symmetry-breaking part, and the minimum of each over pairs and frames."""
    names = list(snapshots)
    frames = sorted({key.split("/")[0] for key in snapshots[names[0]] if "/" in key}, key=int)
    report: dict = {"runs": names, "fields": {}}
    for field in HIDDEN_FIELDS:
        per_frame = {}
        for frame in frames:
            key = f"{frame}/{field}"
            if any(key not in snapshots[name] for name in names):
                continue
            pairs = {}
            for a, b in itertools.combinations(names, 2):
                fa, fb = snapshots[a][key], snapshots[b][key]
                pairs[f"{a}|{b}"] = {
                    "full": correlation(fa, fb),
                    "breaking": correlation(
                        breaking_part(fa, field in ODD_FIELDS),
                        breaking_part(fb, field in ODD_FIELDS),
                    ),
                }
            per_frame[frame] = pairs
        if per_frame:
            full = [p["full"] for pairs in per_frame.values() for p in pairs.values()]
            report["fields"][field] = {"min_full_corr": min(full), "frames": per_frame}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run names under outputs/")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--out", default=None, help="where to write the JSON report")
    args = parser.parse_args()

    root = Path(args.outputs)
    snapshots = {name: dict(np.load(root / name / "audit_fields.npz")) for name in args.runs}
    report = compare_runs(snapshots)
    threshold = json.loads((root / args.runs[0] / "audit.json").read_text())["thresholds"][
        "seed_corr_min"
    ]
    report["seed_corr_min"] = threshold
    report["pass"] = {
        field: entry["min_full_corr"] >= threshold for field, entry in report["fields"].items()
    }
    out = Path(args.out) if args.out else root / f"ensemble-{args.runs[0]}.json"
    out.write_text(json.dumps(report, indent=2))
    for field, entry in report["fields"].items():
        verdict = "pass" if report["pass"][field] else "FAIL"
        print(f"{field}: min full-field correlation {entry['min_full_corr']:.3f} {verdict}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
