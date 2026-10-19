#!/usr/bin/env python
"""One report per run: the decoupled-stage gate and every figure, as one page.

Reads a finished run (its checkpoint, ``audit.json`` and ``audit_fields.npz``)
and writes ``outputs/<run>/report/``: the figures (outline, velocity,
temperature, pressure, source, the mirror split, the scan between frames, the
jump map, the training record, and the seed comparison when other seeds are
given), ``gate.json`` with the five checks a decoupled solution must pass
before the coupled stage, and ``index.html`` that shows them together.

Usage, from the repository root::

    .venv/bin/python scripts/run_report.py trust-wpf6-s0 [trust-wpf6-s1 ...] [--reaudit]

The first run is the one reported; any further runs are its other seeds, used
only for the seed-agreement check. ``--reaudit`` re-runs the evaluate stage
(with the audit) on the CPU first, for a run audited by an older version of the
code (whose audit lacks the scan or the jump map).
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from audit_ensemble import compare_runs  # noqa: E402

from naviernet.config.schema import Config  # noqa: E402
from naviernet.evaluation import front_position, predict_alpha  # noqa: E402
from naviernet.physics.groups import compute_groups  # noqa: E402
from naviernet.training import load_model  # noqa: E402
from naviernet.utils.paths import RunPaths  # noqa: E402

DPI = 110
SEGMENT_NAMES = ("upper", "lower", "root", "nose")
# Loss groups of the training-record figure: what each panel is about.
LOSS_GROUPS = {
    "images and front": ["data", "fv_normal", "fv_apex", "root_sdf", "root_blunt"],
    "interface conditions": ["kinematic", "laplace", "vof"],
    "flow": ["div", "brinkman", "vapour_potential", "bc", "mass_bal"],
    "temperature": ["energy_liquid", "theta_bc"],
}


# The temperature map's colour bands, in degrees C.
TEMPERATURE_BAND_C = 0.4


# --- loading -----------------------------------------------------------------


def run_config(run: str):
    """The run's own config snapshot over the current schema (fields added since
    the run take their defaults), pointed at this checkout and the CPU."""
    snapshot = REPO / "outputs" / run / ".hydra" / "config.yaml"
    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(snapshot))
    cfg.paths.root = str(REPO)
    cfg.run_name = run
    cfg.training.device = "cpu"
    return cfg


def reaudit(run: str) -> None:
    from naviernet.pipeline import Pipeline

    cfg = run_config(run)
    cfg.stage = "evaluate"
    cfg.evaluation.audit = True
    OmegaConf.set_readonly(cfg, True)
    Pipeline(cfg).evaluate()


class Run:
    """A finished run's audit, snapshots, record and scales."""

    def __init__(self, name: str):
        self.name = name
        self.dir = REPO / "outputs" / name
        self.cfg = run_config(name)
        self.audit = json.loads((self.dir / "audit.json").read_text())
        self.snap = np.load(self.dir / "audit_fields.npz")
        self.state = torch.load(
            self.dir / "checkpoints" / "ckpt.pt", map_location="cpu", weights_only=False
        )["state"]
        groups = compute_groups(self.cfg)
        self.l_um = float(self.cfg.scales.L_ref_um)
        self.u_ref = float(self.cfg.scales.U_ref)
        self.t_ref_ms = float(groups["t_ref_ms"])
        self.dT = float(groups["dT_ref"])
        self.t_sat = float(self.cfg.experiment.T_sat_C)
        self.p_scale = float(self.cfg.fluid.rho_l) * self.u_ref**2
        self.frames = [f["frame"] for f in self.audit["frames"]]
        self.t_ms = {f["frame"]: f["t_star"] * self.t_ref_ms for f in self.audit["frames"]}
        self.xs = self.snap["xs"] * self.l_um
        self.ys = self.snap["ys"] * self.l_um


# --- figures -----------------------------------------------------------------


def _frame_axes(title: str):
    fig, axes = plt.subplots(5, 2, figsize=(13, 9.2))
    fig.suptitle(title, fontsize=11)
    return fig, axes.ravel()


def _outline(ax, run: Run, frame: int) -> None:
    gx, gy = np.meshgrid(run.xs, run.ys)
    ax.contour(gx, gy, run.snap[f"{frame}/alpha"], levels=[0.5], colors="k", linewidths=0.9)
    ax.set_title(f"frame {frame}  t = {run.t_ms[frame]:.2f} ms", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path.name


def fig_outline(run: Run, out: Path) -> tuple[str, dict]:
    """The model's outline over the measured mask at every frame, and the nose
    and the projected area against the measurement."""
    model, data, _ = load_model(run.cfg, RunPaths.from_config(run.cfg))
    model.eval()
    stride = 2
    xs, ys = data.x[::stride], data.y[::stride]
    x0 = float(data.pin_anchor[0])
    fig = plt.figure(figsize=(13, 10))
    grid = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1.4])
    nose_m, nose_d, area_m, area_d, ious = [], [], [], [], []
    gx, gy = np.meshgrid(xs * run.l_um, ys * run.l_um)
    for k in range(data.n_event):
        t = float(data.t[k])
        with torch.no_grad():
            pred = predict_alpha(model, data, t, stride) > 0.5
        meas = data.alpha[k, ::stride, ::stride] > 0.5
        ious.append(float((pred & meas).sum() / max((pred | meas).sum(), 1)))
        nose_m.append(front_position(pred, xs, x0) * run.l_um)
        nose_d.append(front_position(meas, xs, x0) * run.l_um)
        px = (xs[1] - xs[0]) * (ys[1] - ys[0]) * run.l_um**2 * 1e-6
        area_m.append(pred.sum() * px)
        area_d.append(meas.sum() * px)
        ax = fig.add_subplot(grid[k // 2, k % 2])
        ax.imshow(
            meas,
            cmap="Greys",
            alpha=0.35,
            origin="lower",
            aspect="equal",
            extent=[gx.min(), gx.max(), gy.min(), gy.max()],
        )
        ax.contour(gx, gy, meas.astype(float), levels=[0.5], colors="k", linewidths=1.0)
        ax.contour(gx, gy, pred.astype(float), levels=[0.5], colors="tab:red", linewidths=1.2)
        ax.set_title(
            f"frame {data.frame_numbers[k]}  t = {t * run.t_ref_ms:.2f} ms  IoU {ious[-1]:.3f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)
    t_ms = np.array([float(data.t[k]) for k in range(data.n_event)]) * run.t_ref_ms
    speed_d, speed_m = np.polyfit(t_ms, nose_d, 1)[0], np.polyfit(t_ms, nose_m, 1)[0]
    ax = fig.add_subplot(grid[5, 0])
    ax.plot(t_ms, nose_d, "ko-", label="measured nose")
    ax.plot(t_ms, nose_m, "r.--", label="model nose")
    ax.set(xlabel="t (ms)", ylabel="nose x (µm)")
    ax.set_title(
        f"nose speed: measured {speed_d:.0f} mm/s, model {speed_m:.0f} mm/s", fontsize=9
    )
    ax.legend(fontsize=8)
    ax = fig.add_subplot(grid[5, 1])
    ax.plot(t_ms, area_d, "ko-", label="measured area")
    ax.plot(t_ms, area_m, "r.--", label="model area")
    ax.set(xlabel="t (ms)", ylabel="projected area (mm²)")
    ax.legend(fontsize=8)
    fig.suptitle(
        f"{run.name}: black = measured outline, red = model; mean IoU {np.mean(ious):.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    stats = {
        "iou": ious,
        "nose_speed_measured": float(speed_d),
        "nose_speed_model": float(speed_m),
    }
    return _save(fig, out / "outline.png"), stats


def fig_fields(run: Run, out: Path) -> dict[str, str]:
    """Speed with streamlines, temperature, pressure against p_v, and the source."""
    names: dict[str, str] = {}
    gx, gy = np.meshgrid(run.xs, run.ys)
    speeds = [
        np.hypot(run.snap[f"{f}/u"], run.snap[f"{f}/v"]) * run.u_ref * 1e3 for f in run.frames
    ]
    vmax = float(np.percentile(np.concatenate([s.ravel() for s in speeds]), 99))
    fig, axes = _frame_axes(f"{run.name}: speed |u| (mm/s) with streamlines")
    for ax, frame, speed in zip(axes, run.frames, speeds, strict=True):
        im = ax.pcolormesh(gx, gy, speed, cmap="viridis", vmin=0, vmax=vmax, shading="auto")
        ax.streamplot(
            run.xs,
            run.ys,
            run.snap[f"{frame}/u"],
            run.snap[f"{frame}/v"],
            color="w",
            linewidth=0.5,
            density=0.7,
            arrowsize=0.6,
        )
        _outline(ax, run, frame)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="|u| (mm/s)")
    names["velocity"] = _save(fig, out / "velocity.png")

    temps = [run.t_sat + run.snap[f"{f}/T"] * run.dT for f in run.frames]
    lo, hi = min(t.min() for t in temps), max(t.max() for t in temps)
    # Banded at TEMPERATURE_BAND_C, so variations finer than a band (the small
    # spots a continuous scale shows in the bulk) read as one colour.
    edges = np.arange(
        np.floor(lo / TEMPERATURE_BAND_C) * TEMPERATURE_BAND_C,
        np.ceil(hi / TEMPERATURE_BAND_C) * TEMPERATURE_BAND_C + TEMPERATURE_BAND_C / 2,
        TEMPERATURE_BAND_C,
    )
    bands = matplotlib.colors.BoundaryNorm(edges, plt.get_cmap("inferno").N)
    fig, axes = _frame_axes(f"{run.name}: temperature (°C); saturation {run.t_sat} °C")
    for ax, frame, temp in zip(axes, run.frames, temps, strict=True):
        im = ax.pcolormesh(gx, gy, temp, cmap="inferno", norm=bands, shading="auto")
        _outline(ax, run, frame)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="T (°C)", ticks=edges)
    names["temperature"] = _save(fig, out / "temperature.png")

    rel = []
    for f in run.frames:
        p = run.snap[f"{f}/p"] * run.p_scale
        inside = run.snap[f"{f}/alpha"] > 0.5
        rel.append(p - (float(np.median(p[inside])) if inside.any() else 0.0))
    lim = float(np.percentile(np.abs(np.concatenate([r.ravel() for r in rel])), 99))
    fig, axes = _frame_axes(f"{run.name}: pressure relative to the vapour pressure p_v (Pa)")
    for ax, frame, r in zip(axes, run.frames, rel, strict=True):
        im = ax.pcolormesh(gx, gy, r, cmap="coolwarm", vmin=-lim, vmax=lim, shading="auto")
        _outline(ax, run, frame)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="p - p_v (Pa)")
    names["pressure"] = _save(fig, out / "pressure.png")

    sources = [run.snap[f"{f}/s"] * run.u_ref / (run.l_um * 1e-6) for f in run.frames]
    smax = float(np.percentile(np.concatenate([s.ravel() for s in sources]), 99.5))
    fig, axes = _frame_axes(f"{run.name}: vapour source s (1/s)")
    for ax, frame, s in zip(axes, run.frames, sources, strict=True):
        im = ax.pcolormesh(gx, gy, s, cmap="magma", vmin=0, vmax=smax, shading="auto")
        _outline(ax, run, frame)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="s (1/s)")
    names["source"] = _save(fig, out / "source.png")
    return names


def fig_mirror(run: Run, out: Path) -> str:
    """u, v and T at a mid-event frame split into their mirror-even and -odd
    parts; the part the symmetry forbids is what the mirror test measures."""
    frame = run.frames[len(run.frames) // 2]
    fig, axes = plt.subplots(3, 3, figsize=(14, 5.8))
    rows = [
        ("u", False, run.u_ref * 1e3, "mm/s"),
        ("v", True, run.u_ref * 1e3, "mm/s"),
        ("T", False, run.dT, "K above saturation"),
    ]
    for r, (name, odd, scale, unit) in enumerate(rows):
        field = run.snap[f"{frame}/{name}"] * scale
        mirrored = field[::-1, :]
        parts = [
            ("field", field),
            ("even part", 0.5 * (field + mirrored)),
            ("odd part", 0.5 * (field - mirrored)),
        ]
        lim = float(np.abs(field).max()) or 1.0
        forbidden = "even" if odd else "odd"
        for c, (label, values) in enumerate(parts):
            ax = axes[r, c]
            im = ax.pcolormesh(
                run.xs, run.ys, values, cmap="coolwarm", vmin=-lim, vmax=lim, shading="auto"
            )
            _outline(ax, run, frame)
            ax.set_title(
                f"{name}: {label}"
                + (f" (the {forbidden} part should vanish)" if c == 0 else ""),
                fontsize=9,
            )
        fig.colorbar(im, ax=axes[r].tolist(), shrink=0.8, label=unit)
    fig.suptitle(
        f"{run.name}, frame {frame}: mirror split about the channel centreline", fontsize=11
    )
    return _save(fig, out / "mirror.png")


def fig_scan(run: Run, out: Path) -> str | None:
    """Residuals, flux defect and jump residual at the scan's instants, with the
    camera frames marked: the physics between the frames as well as at them."""
    scan = run.audit.get("scan")
    if not scan:
        return None
    t_ms = np.array(scan["t_star"]) * run.t_ref_ms
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for name, values in scan["residual_rms"].items():
        axes[0].semilogy(t_ms, values, label=name, lw=1.2)
    axes[0].axhline(run.audit["thresholds"]["residual_rms_max"], color="k", ls=":", lw=0.8)
    axes[0].set_title("residual RMS per equation", fontsize=10)
    axes[1].plot(t_ms, scan["flux_max_abs"], color="tab:blue")
    axes[1].axhline(run.audit["thresholds"]["flux_tol"], color="k", ls=":", lw=0.8)
    axes[1].set_title("worst flux defect (fraction of the inlet flow)", fontsize=10)
    laplace = [np.nan if v is None else v for v in scan["laplace_rms"]]
    axes[2].plot(t_ms, laplace, color="tab:purple")
    axes[2].set_title("Young-Laplace residual RMS on the front", fontsize=10)
    for ax in axes:
        for frame in run.frames:
            ax.axvline(run.t_ms[frame], color="grey", lw=0.5, alpha=0.5)
        ax.set_xlabel("t (ms); grey lines = camera frames")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        f"{run.name}: the physics at {len(t_ms)} instants across the event "
        "(dotted = threshold)",
        fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, out / "scan.png")


def fig_jump(run: Run, out: Path) -> str | None:
    """Every frame's outline coloured by the shape change, in pixels, the
    Young-Laplace mismatch there would need."""
    if f"{run.frames[0]}/front_jump_px" not in run.snap.files:
        return None
    fig, axes = _frame_axes(f"{run.name}: shape change the jump condition asks for (pixels)")
    limit = run.audit["thresholds"].get("jump_px_max", 2.0)
    vmax = 3 * limit
    for ax, frame in zip(axes, run.frames, strict=True):
        x = run.snap[f"{frame}/front_x"] * run.l_um
        y = run.snap[f"{frame}/front_y"] * run.l_um
        px = run.snap[f"{frame}/front_jump_px"]
        sc = ax.scatter(x, y, c=np.minimum(px, vmax), cmap="RdYlGn_r", vmin=0, vmax=vmax, s=9)
        ax.set_xlim(run.xs[0], run.xs[-1])
        ax.set_ylim(run.ys[0], run.ys[-1])
        ax.set_title(f"frame {frame}  median {np.median(px):.1f} px", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.colorbar(
        sc, ax=axes.tolist(), shrink=0.6, label=f"pixels (gate {limit:g}; capped at {vmax:g})"
    )
    return _save(fig, out / "jump.png")


def fig_losses(runs: list[Run], out: Path) -> str:
    """Every loss term over the run, the gradient norm, and the outline guard."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()
    for i, run in enumerate(runs[:2]):
        style = "-" if i == 0 else "--"
        hist = run.state["hist"]
        steps = [h["step"] for h in hist]
        warm = run.state.get("stage_b_warmup_steps", 0)
        for ax, (title, keys) in zip(axes[:4], LOSS_GROUPS.items(), strict=True):
            for key in keys:
                ys = [h.get(key, np.nan) for h in hist]
                if not np.all(np.isnan(ys)):
                    ax.semilogy(steps, ys, style, lw=1.1, label=f"{key} ({run.name})")
            ax.set_title(title)
            ax.axvline(warm, color="grey", lw=0.8, ls=":")
        axes[4].semilogy(
            steps, [h.get("grad_norm", np.nan) for h in hist], style, label=run.name
        )
        axes[4].set_title("total gradient norm")
        guard = run.state.get("outline_hist", [])
        if guard:
            axes[5].plot(
                [g["step"] for g in guard],
                [g["iou_mean"] for g in guard],
                style,
                marker="o",
                ms=3,
                label=f"guard IoU ({run.name})",
            )
            axes[5].plot(
                [g["step"] for g in guard], [g["floor"] for g in guard], color="red", lw=0.8
            )
        axes[5].set_title("outline during the physics stage")
        axes[5].set_ylim(0.85, 1.0)
    for ax in axes:
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)
        ax.set_xlabel("step")
    fig.suptitle(
        "training record (dotted line: physics engages; losses before weighting)", fontsize=10
    )
    fig.tight_layout()
    return _save(fig, out / "losses.png")


# --- the gate ----------------------------------------------------------------


def _verdict(audit: dict, key: str) -> dict | None:
    return audit["verdicts"].get(key)


def gate(run: Run, seeds: dict | None) -> list[dict]:
    """The five checks a decoupled (frozen-shape) solution must pass before the
    coupled stage, each with its verdicts and the numbers behind them."""
    a = run.audit
    frames = a["frames"]
    worst_by_eq = {}
    for f in frames:
        for eq, v in f["residual_rms"].items():
            worst_by_eq[eq] = max(worst_by_eq.get(eq, 0.0), v)
    scan_by_eq = {eq: max(v) for eq, v in (a.get("scan") or {}).get("residual_rms", {}).items()}
    mirror_by_field = {
        name: max(f["mirror"][name] for f in frames)
        for name in ("u", "v", "p", "T", "s")
        if name in frames[0]["mirror"]
    }
    segments: dict[str, float] = {}
    for f in frames:
        for name, v in ((f.get("front") or {}).get("jump_px_by_segment") or {}).items():
            segments[name] = max(segments.get(name, 0.0), v)
    checks = [
        {
            "id": "equations",
            "name": "The equations hold everywhere",
            "question": "Do momentum, continuity, energy and the vapour flow hold at the frames "
            "and between them?",
            "verdicts": [_verdict(a, "residuals") | {"label": "at the camera frames"}]
            + (
                [_verdict(a, "residuals_in_time") | {"label": "between the frames"}]
                if _verdict(a, "residuals_in_time")
                else []
            ),
            "table": {
                "head": ["equation", "worst RMS at frames", "worst RMS between"],
                "rows": [[eq, worst_by_eq[eq], scan_by_eq.get(eq)] for eq in worst_by_eq],
            },
        },
        {
            "id": "seeds",
            "name": "The seeds agree",
            "question": "Do independently trained seeds give the same velocity, pressure and "
            "temperature?",
            "verdicts": []
            if seeds is None
            else [
                {
                    "label": f"{name}",
                    "worst": seeds["fields"][name]["min_full_corr"],
                    "min": seeds["seed_corr_min"],
                    "pass": seeds["pass"][name],
                }
                for name in seeds["fields"]
            ],
            "pending": None
            if seeds is not None
            else "Needs a second seed of this configuration; one seed cannot be compared.",
        },
        {
            "id": "symmetry",
            "name": "The fields are mirror-symmetric",
            "question": "Is u, p, T even and v odd about the centreline, as the channel is?",
            "verdicts": [
                {
                    "label": name,
                    "worst": v,
                    "max": a["thresholds"]["mirror_max"],
                    "pass": v <= a["thresholds"]["mirror_max"],
                }
                for name, v in mirror_by_field.items()
            ],
        },
        {
            "id": "balances",
            "name": "Mass and heat close",
            "question": "Does the flux through every cross-section equal the inlet flow plus the "
            "vapour made upstream, and does the heat budget close?",
            "verdicts": [_verdict(a, "flux") | {"label": "flux at the frames"}]
            + (
                [_verdict(a, "flux_in_time") | {"label": "flux between the frames"}]
                if _verdict(a, "flux_in_time")
                else []
            )
            + [_verdict(a, "heat") | {"label": "heat budget spread"}],
        },
        {
            "id": "jump",
            "name": "Where the shape blocks the physics",
            "question": "How far would the outline have to move for the Young-Laplace "
            "condition to hold, in camera pixels?",
            "verdicts": [_verdict(a, "jump_shape") | {"label": "90th percentile, worst frame"}]
            if _verdict(a, "jump_shape")
            else [],
            "table": {
                "head": ["segment", "median pixels, worst frame"],
                "rows": [[name, segments[name]] for name in SEGMENT_NAMES if name in segments],
            }
            if segments
            else None,
            "pending": None
            if _verdict(a, "jump_shape")
            else "This run's audit predates the jump map; re-run with --reaudit.",
        },
    ]
    for check in checks:
        verdicts = check["verdicts"]
        check["pass"] = bool(verdicts) and all(v["pass"] for v in verdicts)
    return checks


# --- the page ----------------------------------------------------------------


def _fmt(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3g}"
    return html.escape(str(value))


def _verdict_html(v: dict) -> str:
    bound = f"≥ {_fmt(v['min'])}" if "min" in v else f"≤ {_fmt(v['max'])}"
    state = "pass" if v["pass"] else "fail"
    return (
        f'<li class="v {state}"><span class="lab">{html.escape(v["label"])}</span>'
        f'<span class="num">{_fmt(v["worst"])}</span><span class="bound">{bound}</span>'
        f'<span class="tag">{state}</span></li>'
    )


def _table_html(table: dict | None) -> str:
    if not table:
        return ""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in table["head"])
    rows = "".join(
        "<tr>" + "".join(f"<td>{_fmt(c)}</td>" for c in row) + "</tr>" for row in table["rows"]
    )
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def _figure(src: str | None, caption: str) -> str:
    if not src:
        return ""
    return (
        f'<figure><div class="plate"><img src="figures/{src}" alt="{html.escape(caption)}" '
        f'loading="lazy"></div><figcaption>{html.escape(caption)}</figcaption></figure>'
    )


STYLE = """
/* Layout: a single reading column; the gate as a ledger of checks, then the figures as plates. */
:root {
  --bg: #f4f6f8; --surface: #ffffff; --ink: #18212b; --muted: #5a6876; --rule: #d5dce3;
  --accent: #1f6f8b; --pass: #1d7a4a; --fail: #b3412f; --plate: #ffffff;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #12171d; --surface: #1a2129; --ink: #e3e9ef; --muted: #98a6b3; --rule: #2c3640;
  --accent: #6cb6d1; --pass: #5cc48d; --fail: #ec7a66; --plate: #f4f6f8; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #12171d; --surface: #1a2129; --ink: #e3e9ef; --muted: #98a6b3; --rule: #2c3640;
  --accent: #6cb6d1; --pass: #5cc48d; --fail: #ec7a66; --plate: #f4f6f8; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 28px 56px; }
main { max-width: 1080px; margin: 0 auto; display: grid; gap: 28px; }
h1, h2, h3 { font-family: var(--display); font-weight: 600; text-wrap: balance; margin: 0; }
h1 { font-size: 2rem; letter-spacing: .01em; }
h2 { font-size: 1.35rem; }
h3 { font-size: 1.05rem; }
p { margin: 0; max-width: 68ch; }
.eyebrow { font: 600 .75rem/1 var(--mono); letter-spacing: .08em; text-transform: uppercase;
  color: var(--accent); }
header { display: grid; gap: 10px; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 18px; font: .85rem var(--mono); color: var(--muted); }
.summary { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { font: 600 .8rem var(--mono); padding: 4px 10px; border-radius: 999px;
  border: 1px solid var(--rule); background: var(--surface); }
.chip.pass { color: var(--pass); border-color: var(--pass); }
.chip.fail { color: var(--fail); border-color: var(--fail); }
.chip.pending { color: var(--muted); }
section.check { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
  padding: 18px 20px; display: grid; gap: 12px; min-width: 0; }
section.check > .head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline;
  flex-wrap: wrap; }
.state { font: 700 .8rem var(--mono); letter-spacing: .06em; text-transform: uppercase; }
.state.pass { color: var(--pass); } .state.fail { color: var(--fail); } .state.pending { color: var(--muted); }
.question { color: var(--muted); }
ul.verdicts { list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; }
li.v { display: grid; grid-template-columns: minmax(0, 1fr) auto auto 3.5em; gap: 14px;
  align-items: baseline; padding: 4px 0; border-bottom: 1px solid var(--rule);
  font-variant-numeric: tabular-nums; }
li.v .num, li.v .bound { font-family: var(--mono); font-size: .9rem; }
li.v .bound { color: var(--muted); }
li.v .tag { font: 700 .72rem var(--mono); text-transform: uppercase; text-align: right; }
li.v.pass .tag { color: var(--pass); } li.v.fail .tag { color: var(--fail); }
.tw { overflow-x: auto; }
table { border-collapse: collapse; font: .85rem var(--mono); font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 4px 14px 4px 0; border-bottom: 1px solid var(--rule); }
th { color: var(--muted); font-weight: 600; }
.note { color: var(--muted); font-size: .9rem; }
figure { margin: 0; display: grid; gap: 8px; min-width: 0; }
.plate { background: var(--plate); border: 1px solid var(--rule); border-radius: 4px; padding: 8px; }
.plate img { display: block; width: 100%; height: auto; }
figcaption { color: var(--muted); font-size: .9rem; max-width: 80ch; }
.figs { display: grid; gap: 28px; }
@media (max-width: 560px) { li.v { grid-template-columns: minmax(0, 1fr) auto; }
  li.v .bound, li.v .tag { grid-column: 2; } h1 { font-size: 1.6rem; } }
"""


def page(
    run: Run, checks: list[dict], figs: dict, outline: dict, seeds_named: list[str]
) -> str:
    cfg = run.cfg
    tcfg = cfg.training
    passed = sum(1 for c in checks if c["pass"])
    chips = "".join(
        f'<span class="chip {"pass" if c["pass"] else ("pending" if c.get("pending") else "fail")}">'
        f"{html.escape(c['name'])}</span>"
        for c in checks
    )
    sections = []
    for c in checks:
        state = "pass" if c["pass"] else ("pending" if c.get("pending") else "fail")
        verdicts = "".join(_verdict_html(v) for v in c["verdicts"])
        sections.append(
            f'<section class="check" id="{c["id"]}"><div class="head"><h3>{html.escape(c["name"])}</h3>'
            f'<span class="state {state}">{state}</span></div>'
            f'<p class="question">{html.escape(c["question"])}</p>'
            + (f'<ul class="verdicts">{verdicts}</ul>' if verdicts else "")
            + (f'<p class="note">{html.escape(c["pending"])}</p>' if c.get("pending") else "")
            + _table_html(c.get("table"))
            + "</section>"
        )
    a = run.audit["verdicts"]
    outline_rows = (
        [{"label": "worst frame IoU", **a["outline"]}]
        + (
            [{"label": "drop from the warm-up fit", **a["outline_drop"]}]
            if "outline_drop" in a
            else []
        )
        + [{"label": "worst nose error (channel widths)", **a["nose"]}]
    )
    figures = "".join(
        [
            _figure(
                figs.get("outline"),
                "The outline at every camera frame (black measured, red model), "
                "with the nose position and the projected area over time.",
            ),
            _figure(
                figs.get("velocity"), "Speed with streamlines; the black line is the outline."
            ),
            _figure(
                figs.get("temperature"),
                "Temperature. The interface and the vapour sit at "
                "saturation; the liquid starts above it.",
            ),
            _figure(
                figs.get("pressure"),
                "Pressure against the bubble's vapour pressure: uniform "
                "inside, the Young-Laplace jump at the interface, the liquid outside.",
            ),
            _figure(
                figs.get("source"),
                "Where vapour is made: the film under the footprint and the "
                "conduction flux at the outline.",
            ),
            _figure(
                figs.get("mirror"),
                "Check 3 in detail: each field split into the part the "
                "channel's symmetry allows and the part it forbids.",
            ),
            _figure(
                figs.get("scan"),
                "Checks 1 and 4 in time: the physics at instants across the "
                "event, between the camera frames as well as at them.",
            ),
            _figure(
                figs.get("jump"),
                "Check 5 in detail: the shape change the jump condition asks "
                "for along every frame's outline. Green holds within the pixel tolerance.",
            ),
            _figure(
                figs.get("losses"),
                "The training record: every loss term, the gradient norm and "
                "the outline guard.",
            ),
        ]
    )
    stage = (
        "two stages: the outline frozen at the warm-up's fit, then the fields solved"
        if bool(getattr(tcfg, "freeze_geometry", False))
        else "coupled training"
    )
    seeds_text = ", ".join(seeds_named) if seeds_named else "none yet"
    return f"""<title>{html.escape(run.name)} Run Report</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{STYLE}</style>
<main>
<header>
  <span class="eyebrow">Decoupled stage gate · Series-1 · FC-72</span>
  <h1>{html.escape(run.name)}</h1>
  <p>Seed {int(tcfg.seed)}, {int(run.state["done"])} steps, {stage}. A frozen-shape solution
  passes this gate before the coupled stage: the equations hold everywhere, the seeds agree,
  the fields are symmetric, mass and heat close, and the shape the jump condition asks for stays
  within the camera's resolution.</p>
  <div class="meta"><span>{passed} of {len(checks)} checks pass</span>
  <span>nose {outline["nose_speed_model"]:.0f} mm/s, measured {outline["nose_speed_measured"]:.0f}</span>
  <span>IoU {min(outline["iou"]):.3f} to {max(outline["iou"]):.3f}</span>
  <span>other seeds: {html.escape(seeds_text)}</span></div>
  <div class="summary">{chips}</div>
</header>
<section class="check" id="outline"><div class="head"><h3>The outline (the measurement)</h3>
<span class="state {"pass" if all(v["pass"] for v in outline_rows) else "fail"}">{"pass" if all(v["pass"] for v in outline_rows) else "fail"}</span></div>
<p class="question">With the shape frozen after the warm-up this holds by construction; it is shown so the
rest is read against the bubble the physics was given.</p>
<ul class="verdicts">{"".join(_verdict_html(v) for v in outline_rows)}</ul></section>
<h2>The five checks</h2>
{"".join(sections)}
<h2>Figures</h2>
<div class="figs">{figures}</div>
</main>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="the run reported, then its other seeds")
    parser.add_argument(
        "--reaudit",
        action="store_true",
        help="re-run evaluate + audit on the reported run first",
    )
    args = parser.parse_args(argv)
    if args.reaudit:
        reaudit(args.runs[0])  # the seeds contribute only their field snapshots
    runs = [Run(name) for name in args.runs]
    primary = runs[0]
    out = primary.dir / "report"
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, str | None] = {}
    figs["outline"], outline = fig_outline(primary, figdir)
    figs.update(fig_fields(primary, figdir))
    figs["mirror"] = fig_mirror(primary, figdir)
    figs["scan"] = fig_scan(primary, figdir)
    figs["jump"] = fig_jump(primary, figdir)
    figs["losses"] = fig_losses(runs, figdir)
    seeds = None
    if len(runs) > 1:
        snapshots = {r.name: dict(np.load(r.dir / "audit_fields.npz")) for r in runs}
        seeds = compare_runs(snapshots)
        seeds["seed_corr_min"] = float(primary.audit["thresholds"]["seed_corr_min"])
        seeds["pass"] = {
            name: entry["min_full_corr"] >= seeds["seed_corr_min"]
            for name, entry in seeds["fields"].items()
        }
    checks = gate(primary, seeds)
    (out / "gate.json").write_text(
        json.dumps(
            {
                "run": primary.name,
                "seeds": [r.name for r in runs[1:]],
                "checks": checks,
                "outline": outline,
            },
            indent=2,
            allow_nan=False,
            default=float,
        )
    )
    (out / "index.html").write_text(
        page(primary, checks, figs, outline, [r.name for r in runs[1:]]), encoding="utf-8"
    )
    print(f"wrote {out / 'index.html'}")
    for c in checks:
        print(
            f"  {'PASS' if c['pass'] else ('PENDING' if c.get('pending') else 'FAIL')}  {c['name']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
