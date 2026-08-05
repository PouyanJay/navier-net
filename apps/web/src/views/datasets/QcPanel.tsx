import * as d3 from "d3";
import { useEffect, useId, useRef, useState } from "react";

import { ChartFrame } from "../../components/ChartFrame";
import { ViewCanvas } from "../../components";
import {
  attachCrosshair,
  makeTip,
  type Readout,
} from "../../components/charts/crosshair";
import type { QcData, QcKinematics } from "../../lib/api";

/**
 * One chart at a time, at the card's full width.
 *
 * Two-up was tried and does not work here: the checks have very different
 * natural heights, so one panel ends up a tall column beside a short box with
 * dead space under it, and a spatial plot of a 1700µm channel loses too much
 * at half width. A viewBox scales its type along with everything else, so the
 * system has to be sized for where it is actually drawn.
 */
const WIDTH = 920;
// Room on the left and bottom for tick labels *and* an axis title under them.
const MARGIN = { top: 18, right: 20, bottom: 54, left: 74 };
const INNER_W = WIDTH - MARGIN.left - MARGIN.right;

type G = d3.Selection<SVGGElement, unknown, null, undefined>;
type Dots = d3.Selection<SVGCircleElement, number, SVGGElement, unknown>;
type Linear = d3.ScaleLinear<number, number>;

type Check = "kinematics" | "interface";

/** The headline number a check produces, and what says whether to trust it. */
interface Finding {
  value: string;
  unit: string;
  note: string;
}

interface CheckSpec {
  id: Check;
  label: string;
  sub: string;
  finding: (qc: QcData) => Finding | null;
}

const CHECKS: CheckSpec[] = [
  {
    id: "kinematics",
    label: "Growth kinematics",
    sub: "L(t), its fit, and the residual",
    finding: (qc) =>
      qc.kinematics.t_ms.length > 1
        ? {
            value: qc.kinematics.fit_slope_mm_s.toFixed(0),
            unit: "mm·s⁻¹ nose speed",
            note: `R² ${fitR2(qc.kinematics).toFixed(3)}`,
          }
        : null,
  },
  {
    id: "interface",
    label: "Interface evolution",
    sub: "bubble silhouettes, first frame to last",
    finding: (qc) => {
      const frames = qc.interface.frames;
      if (frames.length === 0) return null;
      return {
        value: `${frames.length}`,
        unit: "frames overlaid",
        note: `${ms(frames[0].t_ms)} → ${ms(frames[frames.length - 1].t_ms)} ms`,
      };
    },
  },
];

interface QcChecksProps {
  /** The QC data once the tensors exist, or null before / while they build. */
  qc: QcData | null;
  /** Whether preprocessing has run, to word the empty state correctly. */
  processed: boolean;
}

/** Milliseconds without trailing zeros: 1.5 stays 1.5, 2.0 becomes 2. */
function ms(value: number): string {
  return `${Number(value.toFixed(2))}`;
}

/** A signed micrometre residual, so a positive one is unambiguous. */
function signedUm(value: number): string {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(1)} µm`;
}

/** The active check's data in export-friendly long format. */
function qcRows(qc: QcData, check: Check): Record<string, unknown>[] {
  if (check === "kinematics")
    return qc.kinematics.t_ms.map((t, i) => ({
      t_ms: t,
      length_um: qc.kinematics.length_um[i],
      fit_slope_mm_s: qc.kinematics.fit_slope_mm_s,
    }));
  return qc.interface.frames.flatMap((frame) =>
    frame.rings.flatMap((ring, ringIndex) =>
      ring.map(([xStar, yStar]) => ({
        camera_frame: frame.camera_frame,
        t_ms: frame.t_ms,
        ring: ringIndex,
        x_star: xStar,
        y_star: yStar,
      })),
    ),
  );
}

/** The section's identity and, on the right, whatever the open check found. */
function QcHeader({ finding }: { finding: Finding | null }) {
  return (
    <div className="qc-sub-hd">
      <div className="qc-sub-title">
        <h3>Preprocessing QC</h3>
        <span className="sub">computed from the training tensors</span>
      </div>
      {finding && (
        <p className="qc-finding">
          <span className="qc-finding-v mono">{finding.value}</span>
          <span className="qc-finding-u mono">{finding.unit}</span>
          <span className="qc-finding-r mono">{finding.note}</span>
        </p>
      )}
    </div>
  );
}

export function QcChecks({ qc, processed }: QcChecksProps) {
  const [check, setCheck] = useState<Check>("kinematics");
  const pickerId = useId();

  if (!qc) {
    return (
      <section className="qc-sub" aria-label="Preprocessing QC">
        <QcHeader finding={null} />
        <p className="state-note" role="status">
          {processed
            ? "Building the QC checks from the tensors…"
            : "Run preprocessing to compute the QC checks."}
        </p>
      </section>
    );
  }

  const active = CHECKS.find((c) => c.id === check)!;

  /* A select rather than a segmented control: it takes one slot however many
     checks there are. It rides in the chart's own toolbar because it changes
     what is plotted, and because a card with two stacked control strips reads
     as an accident. */
  const picker = (
    <div className="qc-pick">
      {/* The caption sits outside the label: inside it, it would join the
          select's accessible name and describe the wrong thing. */}
      <label className="sr-only" htmlFor={pickerId}>
        Preprocessing check
      </label>
      <select
        id={pickerId}
        value={check}
        onChange={(event) => setCheck(event.target.value as Check)}
      >
        {CHECKS.map((c) => (
          <option key={c.id} value={c.id}>
            {c.label}
          </option>
        ))}
      </select>
      <span className="qc-pick-sub">{active.sub}</span>
    </div>
  );

  return (
    <section className="qc-sub" aria-label="Preprocessing QC">
      <QcHeader finding={active.finding(qc)} />

      <ChartFrame
        name={`${qc.dataset}-qc-${check}`}
        title={active.label}
        rows={qcRows(qc, check)}
        controls={picker}
        render={() => (
          <ViewCanvas>
            {check === "kinematics" && <KinematicsChart qc={qc} />}
            {check === "interface" && <InterfaceChart qc={qc} />}
          </ViewCanvas>
        )}
      />
    </section>
  );
}

// ── Shared axis furniture ────────────────────────────────────────────────────

interface AxisSpec {
  /** Quantity and unit, e.g. "t (ms)". Always both: a bare number is unreadable. */
  title: string;
  ticks?: number;
  format?: (value: number) => string;
}

/** Recessive gridlines across the plot, on the y ticks only. */
function drawGrid(g: G, y: Linear, ticks: number): void {
  g.append("g")
    .attr("class", "chart-grid")
    .selectAll("line")
    .data(y.ticks(ticks))
    .join("line")
    .attr("x1", 0)
    .attr("x2", INNER_W)
    .attr("y1", (d) => y(d))
    .attr("y2", (d) => y(d));
}

/**
 * Tick labels plus a titled axis on both edges.
 *
 * Every chart on the dark canvas goes through here, so no chart can end up
 * with bare numbers and no statement of what they measure.
 */
function drawAxes(
  g: G,
  x: Linear,
  y: Linear,
  innerH: number,
  xAxis: AxisSpec,
  yAxis: AxisSpec,
): void {
  const xTicks = x.ticks(xAxis.ticks ?? 6);
  const yTicks = y.ticks(yAxis.ticks ?? 5);
  const xFormat = xAxis.format ?? d3.format("~s");
  const yFormat = yAxis.format ?? d3.format("~s");

  g.append("g")
    .attr("class", "chart-axis")
    .selectAll("text")
    .data(yTicks)
    .join("text")
    .attr("x", -10)
    .attr("y", (d) => y(d))
    .attr("dy", "0.32em")
    .attr("text-anchor", "end")
    .text(yFormat);

  g.append("g")
    .attr("class", "chart-axis")
    .selectAll("text")
    .data(xTicks)
    .join("text")
    .attr("x", (d) => x(d))
    .attr("y", innerH + 20)
    .attr("text-anchor", "middle")
    .text(xFormat);

  g.append("text")
    .attr("class", "chart-axis-title")
    .attr("x", INNER_W / 2)
    .attr("y", innerH + 44)
    .attr("text-anchor", "middle")
    .text(xAxis.title);

  g.append("text")
    .attr("class", "chart-axis-title")
    .attr(
      "transform",
      `translate(${-MARGIN.left + 16},${innerH / 2}) rotate(-90)`,
    )
    .attr("text-anchor", "middle")
    .text(yAxis.title);
}

// ── Growth kinematics ────────────────────────────────────────────────────────

/* The fit gets a residual strip beneath it, sharing its x axis. A line that
   looks straight is not evidence that it IS straight; the residual is. */
const KIN_RESID_H = 56;
const KIN_GAP = 18;
const KIN_HEIGHT = 320;
const KIN_INNER_H =
  KIN_HEIGHT - MARGIN.top - MARGIN.bottom - KIN_RESID_H - KIN_GAP;
const KIN_PLOT_H = KIN_INNER_H + KIN_GAP + KIN_RESID_H;

/** measured − fit, per frame, in µm. */
function residuals(kin: QcKinematics): number[] {
  return kin.length_um.map(
    (L, i) => L - (kin.fit_slope_mm_s * kin.t_ms[i] + kin.fit_intercept_um),
  );
}

/** Coefficient of determination for the straight-line fit. */
function fitR2(kin: QcKinematics): number {
  const mean = d3.mean(kin.length_um) ?? 0;
  const ssTot = d3.sum(kin.length_um, (L) => (L - mean) ** 2);
  const ssRes = d3.sum(residuals(kin), (r) => r ** 2);
  return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
}

/** The residual strip: a zero line, a +/-1 sigma band, and the scatter. */
function drawResiduals(g: G, x: Linear, kin: QcKinematics): Dots {
  const res = residuals(kin);
  const spread = Math.max(d3.max(res, Math.abs) ?? 1, 1e-6);
  const top = KIN_INNER_H + KIN_GAP;
  const y = d3
    .scaleLinear()
    .domain([-spread, spread])
    .range([top + KIN_RESID_H, top]);
  const sigma = Math.sqrt(d3.mean(res, (r) => r ** 2) ?? 0);

  g.append("rect")
    .attr("class", "qc-resid-band")
    .attr("x", 0)
    .attr("y", y(sigma))
    .attr("width", INNER_W)
    .attr("height", Math.max(1, y(-sigma) - y(sigma)));
  g.append("line")
    .attr("class", "qc-resid-zero")
    .attr("x1", 0)
    .attr("x2", INNER_W)
    .attr("y1", y(0))
    .attr("y2", y(0));
  g.append("text")
    .attr("class", "qc-resid-label")
    .attr("x", 0)
    .attr("y", top - 5)
    .text(`residual (µm) · measured − fit · ±${sigma.toFixed(1)} rms`);
  return g
    .append("g")
    .selectAll<SVGCircleElement, number>("circle")
    .data(res)
    .join("circle")
    .attr("class", "qc-resid-dot")
    .attr("cx", (_, i) => x(kin.t_ms[i]))
    .attr("cy", (d) => y(d))
    .attr("r", 2.6);
}

function kinScales(kin: QcKinematics): { x: Linear; y: Linear } {
  const x = d3
    .scaleLinear()
    .domain(d3.extent(kin.t_ms) as [number, number])
    .range([0, INNER_W]);
  const y = d3
    .scaleLinear()
    .domain(d3.extent(kin.length_um) as [number, number])
    .nice()
    .range([KIN_INNER_H, 0]);
  return { x, y };
}

function drawFitLine(g: G, x: Linear, y: Linear, kin: QcKinematics): void {
  const [t0, t1] = x.domain();
  const fitY = (t: number) => kin.fit_slope_mm_s * t + kin.fit_intercept_um;
  g.append("line")
    .attr("class", "chart-line qc-fit")
    .attr("x1", x(t0))
    .attr("y1", y(fitY(t0)))
    .attr("x2", x(t1))
    .attr("y2", y(fitY(t1)));
  // Top left, where a rising series leaves room. Ending the label at the right
  // edge put it on top of the last frame, the one furthest along the fit.
  g.append("text")
    .attr("class", "qc-fit-label")
    .attr("x", 0)
    .attr("y", 12)
    .text(`fit dL/dt = ${kin.fit_slope_mm_s.toFixed(0)} mm/s`);
}

function drawMeasuredSeries(
  g: G,
  x: Linear,
  y: Linear,
  kin: QcKinematics,
): Dots {
  const line = d3
    .line<number>()
    .x((_, i) => x(kin.t_ms[i]))
    .y((d) => y(d));
  g.append("path")
    .attr("class", "chart-line qc-measured")
    .attr("d", line(kin.length_um));
  return g
    .append("g")
    .selectAll<SVGCircleElement, number>("circle")
    .data(kin.length_um)
    .join("circle")
    .attr("class", "qc-dot")
    .attr("cx", (_, i) => x(kin.t_ms[i]))
    .attr("cy", (d) => y(d))
    .attr("r", 4);
}

/** Measured bubble length per frame with the linear growth fit. */
function KinematicsChart({ qc }: { qc: QcData }) {
  const ref = useRef<SVGSVGElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const liveRef = useRef<HTMLParagraphElement>(null);
  const kin = qc.kinematics;

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    if (kin.t_ms.length === 0) return;
    const g = svg
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
    const { x, y } = kinScales(kin);
    drawGrid(g, y, 5);
    drawAxes(
      g,
      x,
      y,
      // Below the residual strip: the two panels share one x axis, which is
      // what lets a reader carry a frame's position from the fit to its error.
      KIN_PLOT_H,
      { title: "t (ms), from the first frame", format: (d) => `${d}` },
      { title: "L (µm), streamwise bubble length" },
    );
    drawFitLine(g, x, y, kin);
    const dots = drawMeasuredSeries(g, x, y, kin);
    const residDots = drawResiduals(g, x, kin);

    // The readout carries measured, fit and residual together: the whole point
    // of the check is the gap between the first two, which is the third.
    const res = residuals(kin);
    const at = (i: number): Readout | null => {
      if (i < 0 || i >= kin.t_ms.length) return null;
      const fitted = kin.fit_slope_mm_s * kin.t_ms[i] + kin.fit_intercept_um;
      return {
        xPix: x(kin.t_ms[i]),
        index: i,
        title: `frame ${i + 1} · t = ${ms(kin.t_ms[i])} ms`,
        rows: [
          {
            text: `measured  ${kin.length_um[i].toFixed(0)} µm`,
            swatchClass: "measured",
          },
          { text: `fit       ${fitted.toFixed(0)} µm`, swatchClass: "fit" },
          { text: `residual  ${signedUm(res[i])}` },
        ],
      };
    };
    const bisect = d3.bisector((t: number) => t).center;

    return attachCrosshair({
      svg,
      g,
      tipEl: tipRef.current,
      width: WIDTH,
      margin: MARGIN,
      innerWidth: INNER_W,
      innerHeight: KIN_PLOT_H,
      readout: (px) => at(bisect(kin.t_ms, x.invert(px))),
      keyboard: { count: kin.t_ms.length, at, live: liveRef.current },
      onActive: (active) => {
        dots.classed("hot", (_, i) => i === active?.index);
        residDots.classed("hot", (_, i) => i === active?.index);
      },
    });
  }, [kin]);

  return (
    <div className="chart-wrap">
      <svg
        ref={ref}
        viewBox={`0 0 ${WIDTH} ${KIN_HEIGHT}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Bubble length in micrometres against time in milliseconds, with a linear fit of ${kin.fit_slope_mm_s.toFixed(0)} millimetres per second. Focus the chart and use the arrow keys to read each frame.`}
      />
      <div ref={tipRef} className="chart-tip" style={{ display: "none" }} />
      <p ref={liveRef} className="sr-only" role="status" aria-live="polite" />
    </div>
  );
}

// ── Interface evolution ──────────────────────────────────────────────────────

/* Time ramp: one hue, dim to bright, so ordering reads without a key. Both
   ends are canvas tokens — the canvas is dark in both themes, so a chrome
   accent here washes the newest frames out to near-white in the light one. */
const RAMP_FROM = "--console-dim";
const RAMP_TO = "--canvas-accent";
const LEGEND_W = 132;
const LEGEND_STOPS = 24;

type Frames = QcData["interface"]["frames"];
type Paths = d3.Selection<SVGPathElement, Frames[number], SVGGElement, unknown>;

function tokenColor(node: Element, name: string, fallback: string): string {
  const value = getComputedStyle(node).getPropertyValue(name).trim();
  return value || fallback;
}

function drawSilhouettes(
  g: G,
  x: Linear,
  y: Linear,
  frames: Frames,
  ramp: (t: number) => string,
): Paths {
  const path = d3
    .line<number[]>()
    .x((p) => x(p[0]))
    .y((p) => y(p[1]));
  return (
    g
      .append("g")
      .attr("class", "qc-frames")
      .selectAll<SVGPathElement, Frames[number]>("path")
      .data(frames)
      .join("path")
      .attr("class", "qc-silhouette")
      // One path per frame, every ring in it, so an even-odd fill keeps holes.
      .attr("d", (frame) =>
        frame.rings.map((ring) => `${path(ring) ?? ""}Z`).join(" "),
      )
      .attr("fill", (_, order) =>
        ramp(frames.length > 1 ? order / (frames.length - 1) : 1),
      )
      .attr("stroke", (_, order) =>
        ramp(frames.length > 1 ? order / (frames.length - 1) : 1),
      )
  );
}

function drawTimeLegend(g: G, ramp: (t: number) => string, frames: Frames) {
  if (frames.length < 2) return;
  const legend = g
    .append("g")
    .attr("transform", `translate(${INNER_W - LEGEND_W},${-8})`);
  legend
    .selectAll("rect")
    .data(d3.range(LEGEND_STOPS))
    .join("rect")
    .attr("x", (i) => (i * LEGEND_W) / LEGEND_STOPS)
    .attr("y", 0)
    .attr("width", LEGEND_W / LEGEND_STOPS + 0.5)
    .attr("height", 6)
    .attr("fill", (i) => ramp(i / (LEGEND_STOPS - 1)));
  legend
    .append("text")
    .attr("class", "chart-axis-title")
    .attr("x", 0)
    .attr("y", -4)
    .text(`${ms(frames[0].t_ms)} ms`);
  legend
    .append("text")
    .attr("class", "chart-axis-title")
    .attr("x", LEGEND_W)
    .attr("y", -4)
    .attr("text-anchor", "end")
    .text(`${ms(frames[frames.length - 1].t_ms)} ms`);
}

/** Every point of every ring in a frame, so its extent can be measured. */
function framePoints(frame: Frames[number]): number[][] {
  return frame.rings.flat();
}

/** Bubble silhouettes frame by frame, oldest dim to newest bright. */
function InterfaceChart({ qc }: { qc: QcData }) {
  const ref = useRef<SVGSVGElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const liveRef = useRef<HTMLParagraphElement>(null);
  const { x_range, y_range, x_pin_star, l_ref_um, frames } = qc.interface;
  // Equal x/y aspect: the channel's shape is part of what is being checked.
  const innerH = Math.max(
    120,
    Math.round(
      (INNER_W * (y_range[1] - y_range[0])) / (x_range[1] - x_range[0]),
    ),
  );
  const height = innerH + MARGIN.top + MARGIN.bottom;

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    const node = ref.current;
    if (!node) return;
    const g = svg
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
    // Axes are in µm; the tensors' x* is a nondimensional working coordinate.
    const toUm = (v: number) => v * l_ref_um;
    const x = d3.scaleLinear().domain(x_range.map(toUm)).range([0, INNER_W]);
    const y = d3.scaleLinear().domain(y_range.map(toUm)).range([innerH, 0]);
    const xStar = d3.scaleLinear().domain(x_range).range([0, INNER_W]);
    const yStar = d3.scaleLinear().domain(y_range).range([innerH, 0]);
    const ramp = d3.interpolateRgb(
      tokenColor(node, RAMP_FROM, "#586a8a"),
      tokenColor(node, RAMP_TO, "#93c5fd"),
    );

    drawGrid(g, y, 4);
    const paths = drawSilhouettes(g, xStar, yStar, frames, ramp);
    g.append("line")
      .attr("class", "qc-pin")
      .attr("x1", x(toUm(x_pin_star)))
      .attr("x2", x(toUm(x_pin_star)))
      .attr("y1", 0)
      .attr("y2", innerH)
      .append("title")
      .text("Pinned nucleation cavity");
    g.append("text")
      .attr("class", "qc-annotation")
      .attr("x", x(toUm(x_pin_star)) + 6)
      .attr("y", 12)
      .text("pinned cavity");
    drawAxes(
      g,
      x,
      y,
      innerH,
      { title: "x (µm), downstream", format: (d) => `${d}` },
      { title: "y (µm), across channel", ticks: 4, format: (d) => `${d}` },
    );
    drawTimeLegend(g, ramp, frames);

    /* This plot has no x to scan: the frames are stacked outlines, so the
       probe picks a whole frame and the others recede behind it. */
    const tip = makeTip(tipRef.current, node, { width: WIDTH, margin: MARGIN });
    const at = (i: number): Readout | null => {
      const frame = frames[i];
      if (!frame) return null;
      const points = framePoints(frame);
      const noseStar = d3.max(points, (p) => p[0]) ?? 0;
      const span =
        (d3.max(points, (p) => p[1]) ?? 0) - (d3.min(points, (p) => p[1]) ?? 0);
      return {
        /* Parked at the plot's top-left corner as one line, not floated over
           the outline. This plot is wide and shallow and the bubble grows
           left to right, so a box anywhere near the pointer covers the nose,
           which is the part being read. */
        xPix: 0,
        yPix: 0,
        index: i,
        title: `frame ${frame.index + 1} · t = ${ms(frame.t_ms)} ms`,
        rows: [
          { text: `nose x   ${toUm(noseStar).toFixed(0)} µm` },
          { text: `spanwise ${toUm(span).toFixed(0)} µm` },
        ],
      };
    };
    const focus = (i: number | null) => {
      const active = i === null ? null : at(i);
      paths
        .classed("hot", (_, k) => k === active?.index)
        .classed("cool", () => active !== null);
      if (!active) {
        tip.hide();
        if (liveRef.current) liveRef.current.textContent = "";
        return;
      }
      tip.show(active);
      if (liveRef.current)
        liveRef.current.textContent = `${active.title}. ${active.rows
          .map((row) => row.text)
          .join(". ")}`;
    };

    paths
      .on("pointerenter", (_, frame) => focus(frames.indexOf(frame)))
      .on("pointerleave", () => focus(null));

    let index = 0;
    svg
      .attr("tabindex", 0)
      .on("focus", () => focus(index))
      .on("blur", () => focus(null))
      .on("keydown", (event: KeyboardEvent) => {
        if (event.key === "Escape") return focus(null);
        const step =
          event.key === "ArrowRight" || event.key === "ArrowUp"
            ? 1
            : event.key === "ArrowLeft" || event.key === "ArrowDown"
              ? -1
              : 0;
        if (step === 0) return;
        event.preventDefault();
        index = Math.max(0, Math.min(frames.length - 1, index + step));
        focus(index);
      });

    return () => focus(null);
  }, [qc, x_range, y_range, x_pin_star, l_ref_um, frames, innerH]);

  return (
    <div className="chart-wrap">
      <svg
        ref={ref}
        viewBox={`0 0 ${WIDTH} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Bubble outline for ${frames.length} frames from ${ms(frames[0]?.t_ms ?? 0)} to ${ms(frames[frames.length - 1]?.t_ms ?? 0)} milliseconds, later frames drawn brighter, on axes in micrometres. Focus the chart and use the arrow keys to read each frame.`}
      />
      <div
        ref={tipRef}
        className="chart-tip qc-frame-tip"
        style={{ display: "none" }}
      />
      <p ref={liveRef} className="sr-only" role="status" aria-live="polite" />
    </div>
  );
}
