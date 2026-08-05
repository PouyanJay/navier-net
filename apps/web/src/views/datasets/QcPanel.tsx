import * as d3 from "d3";
import { useEffect, useRef } from "react";

import { ChartFrame } from "../../components/ChartFrame";
import { ViewCanvas } from "../../components";
import type { QcData, QcKinematics } from "../../lib/api";

/**
 * The charts sit two-up, so the coordinate system is sized for half a card
 * rather than a whole one.
 *
 * This matters more than it looks: a viewBox scales its type along with
 * everything else, so a 920-wide system rendered into 357px of card would set
 * its 10px tick labels at 3.9px. Narrowing the box is what keeps them readable
 * at the width they are actually drawn.
 */
const WIDTH = 470;
// Room on the left and bottom for tick labels *and* an axis title under them.
const MARGIN = { top: 16, right: 14, bottom: 44, left: 52 };
const INNER_W = WIDTH - MARGIN.left - MARGIN.right;

type G = d3.Selection<SVGGElement, unknown, null, undefined>;
type Linear = d3.ScaleLinear<number, number>;

type Check = "kinematics" | "interface";

interface QcChecksProps {
  /** The QC data once the tensors exist, or null before / while they build. */
  qc: QcData | null;
  /** Whether preprocessing has run, to word the empty state correctly. */
  processed: boolean;
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

export function QcChecks({ qc, processed }: QcChecksProps) {
  if (!qc) {
    return (
      <section className="qc-sub" aria-label="Preprocessing QC">
        <div className="qc-sub-hd">
          <div className="qc-sub-title">
            <h3>Preprocessing QC</h3>
            <span className="sub">computed from the training tensors</span>
          </div>
        </div>
        <p className="state-note" role="status">
          {processed
            ? "Building the QC checks from the tensors…"
            : "Run preprocessing to compute the QC checks."}
        </p>
      </section>
    );
  }

  return (
    <section className="qc-sub" aria-label="Preprocessing QC">
      {/* Both checks, side by side and neither hidden. There is no switch left
          to press: two is few enough to show, which is the whole reason the
          third view came out. */}
      <div className="qc-pair">
        <div className="qc-one">
          <div className="qc-sub-hd">
            <div className="qc-sub-title">
              <h3>Growth kinematics</h3>
              <span className="sub">L(t), its fit, and the residual</span>
            </div>
          </div>
          {qc.kinematics.t_ms.length > 1 && (
            <p className="qc-finding">
              <span className="qc-finding-v mono">
                {qc.kinematics.fit_slope_mm_s.toFixed(0)}
              </span>
              <span className="qc-finding-u mono">mm·s⁻¹ nose speed</span>
              <span className="qc-finding-r mono">
                R² {fitR2(qc.kinematics).toFixed(3)}
              </span>
            </p>
          )}
          <ChartFrame
            name={`${qc.dataset}-qc-kinematics`}
            title="Growth kinematics"
            rows={qcRows(qc, "kinematics")}
            render={() => (
              <ViewCanvas>
                <KinematicsChart qc={qc} />
              </ViewCanvas>
            )}
          />
        </div>

        <div className="qc-one">
          <div className="qc-sub-hd">
            <div className="qc-sub-title">
              <h3>Interface evolution</h3>
              <span className="sub">
                bubble silhouettes, first frame to last
              </span>
            </div>
          </div>
          <ChartFrame
            name={`${qc.dataset}-qc-interface`}
            title="Interface evolution"
            rows={qcRows(qc, "interface")}
            render={() => (
              <ViewCanvas>
                <InterfaceChart qc={qc} />
              </ViewCanvas>
            )}
          />
        </div>
      </div>
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
  const xTicks = x.ticks(xAxis.ticks ?? 4);
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
function drawResiduals(g: G, x: Linear, kin: QcKinematics): void {
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
  g.append("g")
    .selectAll("circle")
    .data(res)
    .join("circle")
    .attr("class", "qc-resid-dot")
    .attr("cx", (_, i) => x(kin.t_ms[i]))
    .attr("cy", (d) => y(d))
    .attr("r", 2.6)
    .append("title")
    .text(
      (d, i) =>
        `t = ${kin.t_ms[i]} ms · ${d >= 0 ? "+" : ""}${d.toFixed(1)} µm`,
    );
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
  g.append("text")
    .attr("class", "qc-fit-label")
    .attr("x", INNER_W - 4)
    .attr("y", 12)
    .attr("text-anchor", "end")
    .text(`fit dL/dt = ${kin.fit_slope_mm_s.toFixed(0)} mm/s`);
}

function drawMeasuredSeries(
  g: G,
  x: Linear,
  y: Linear,
  kin: QcKinematics,
): void {
  const line = d3
    .line<number>()
    .x((_, i) => x(kin.t_ms[i]))
    .y((d) => y(d));
  g.append("path")
    .attr("class", "chart-line qc-measured")
    .attr("d", line(kin.length_um));
  g.append("g")
    .selectAll("circle")
    .data(kin.length_um)
    .join("circle")
    .attr("class", "qc-dot")
    .attr("cx", (_, i) => x(kin.t_ms[i]))
    .attr("cy", (d) => y(d))
    .attr("r", 4)
    .append("title")
    .text((d, i) => `t = ${kin.t_ms[i]} ms · L = ${d.toFixed(0)} µm`);
}

/** Measured bubble length per frame with the linear growth fit. */
function KinematicsChart({ qc }: { qc: QcData }) {
  const ref = useRef<SVGSVGElement>(null);
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
      KIN_INNER_H + KIN_GAP + KIN_RESID_H,
      { title: "t (ms), from the first frame", format: (d) => `${d}` },
      { title: "L (µm), streamwise bubble length" },
    );
    drawFitLine(g, x, y, kin);
    drawMeasuredSeries(g, x, y, kin);
    drawResiduals(g, x, kin);
  }, [kin]);

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${WIDTH} ${KIN_HEIGHT}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={`Bubble length in micrometres against time in milliseconds, with a linear fit of ${kin.fit_slope_mm_s.toFixed(0)} millimetres per second.`}
    />
  );
}

// ── Interface evolution ──────────────────────────────────────────────────────

/** Time ramp: one hue, dim to bright, so ordering reads without a key. */
const RAMP_FROM = "--console-dim";
const RAMP_TO = "--acc2";
const LEGEND_W = 132;
const LEGEND_STOPS = 24;

function tokenColor(node: Element, name: string, fallback: string): string {
  const value = getComputedStyle(node).getPropertyValue(name).trim();
  return value || fallback;
}

function drawSilhouettes(
  g: G,
  x: Linear,
  y: Linear,
  qc: QcData,
  ramp: (t: number) => string,
): void {
  const frames = qc.interface.frames;
  const path = d3
    .line<number[]>()
    .x((p) => x(p[0]))
    .y((p) => y(p[1]));
  frames.forEach((frame, order) => {
    const shade = ramp(frames.length > 1 ? order / (frames.length - 1) : 1);
    // One path per frame, every ring in it, so an even-odd fill keeps holes.
    const d = frame.rings.map((ring) => `${path(ring) ?? ""}Z`).join(" ");
    g.append("path")
      .attr("class", "qc-silhouette")
      .attr("d", d)
      .attr("fill", shade)
      .attr("stroke", shade)
      .append("title")
      .text(`t = ${frame.t_ms} ms (frame ${frame.index + 1})`);
  });
}

function drawTimeLegend(
  g: G,
  ramp: (t: number) => string,
  frames: QcData["interface"]["frames"],
): void {
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
    .text(`${frames[0].t_ms} ms`);
  legend
    .append("text")
    .attr("class", "chart-axis-title")
    .attr("x", LEGEND_W)
    .attr("y", -4)
    .attr("text-anchor", "end")
    .text(`${frames[frames.length - 1].t_ms} ms`);
}

/** Bubble silhouettes frame by frame, oldest dim to newest bright. */
function InterfaceChart({ qc }: { qc: QcData }) {
  const ref = useRef<SVGSVGElement>(null);
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
    const ramp = d3.interpolateRgb(
      tokenColor(node, RAMP_FROM, "#586a8a"),
      tokenColor(node, RAMP_TO, "#93c5fd"),
    );

    drawGrid(g, y, 4);
    drawSilhouettes(
      g,
      d3.scaleLinear().domain(x_range).range([0, INNER_W]),
      d3.scaleLinear().domain(y_range).range([innerH, 0]),
      qc,
      ramp,
    );
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
  }, [qc, x_range, y_range, x_pin_star, l_ref_um, frames, innerH]);

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${WIDTH} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={`Bubble outline for ${frames.length} frames from ${frames[0]?.t_ms ?? 0} to ${frames[frames.length - 1]?.t_ms ?? 0} milliseconds, later frames drawn brighter, on axes in micrometres.`}
    />
  );
}
