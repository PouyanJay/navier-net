import * as d3 from "d3";
import { useEffect, useRef, useState } from "react";

import { Panel, ViewCanvas } from "../../components";
import type { QcData, QcKinematics } from "../../lib/api";

const WIDTH = 920;
// Room on the left and bottom for tick labels *and* an axis title under them.
const MARGIN = { top: 18, right: 20, bottom: 54, left: 74 };
const INNER_W = WIDTH - MARGIN.left - MARGIN.right;

type G = d3.Selection<SVGGElement, unknown, null, undefined>;
type Linear = d3.ScaleLinear<number, number>;

type Check = "kinematics" | "interface" | "sdf";

const CHECKS: { id: Check; label: string; sub: string }[] = [
  { id: "kinematics", label: "Growth kinematics", sub: "L(t) + fit" },
  { id: "interface", label: "Interface evolution", sub: "bubble silhouettes" },
  { id: "sdf", label: "Signed distance", sub: "mid frame" },
];

/** The three preprocessing checks as interactive charts behind one switch. */
export function QcPanel({ qc }: { qc: QcData }) {
  const [check, setCheck] = useState<Check>("kinematics");
  return (
    <Panel
      title="Preprocessing QC"
      subtitle="computed from the training tensors; inspect before solving"
      actions={
        <div className="seg" role="tablist" aria-label="QC check">
          {CHECKS.map((c) => (
            <button
              key={c.id}
              type="button"
              role="tab"
              aria-selected={check === c.id}
              className={check === c.id ? "segb on" : "segb"}
              onClick={() => setCheck(c.id)}
            >
              {c.label}
              <span>{c.sub}</span>
            </button>
          ))}
        </div>
      }
    >
      <ViewCanvas>
        {check === "kinematics" && <KinematicsChart qc={qc} />}
        {check === "interface" && <InterfaceChart qc={qc} />}
        {check === "sdf" && <SdfChart qc={qc} />}
      </ViewCanvas>
    </Panel>
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

const KIN_HEIGHT = 320;
const KIN_INNER_H = KIN_HEIGHT - MARGIN.top - MARGIN.bottom;

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
      KIN_INNER_H,
      { title: "t (ms), from the first frame", format: (d) => `${d}` },
      { title: "L (µm), streamwise bubble length" },
    );
    drawFitLine(g, x, y, kin);
    drawMeasuredSeries(g, x, y, kin);
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

// ── Signed distance field ────────────────────────────────────────────────────

/** Diverging heatmap of the mid-frame signed distance field. */
function SdfChart({ qc }: { qc: QcData }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { values, t_ms, x_range, y_range } = qc.sdf;
  const l_ref_um = qc.interface.l_ref_um;
  const rows = values.length;
  const cols = rows > 0 ? values[0].length : 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rows === 0) return;
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const image = ctx.createImageData(cols, rows);
    // Matplotlib's RdBu, the colormap the pipeline's own QC figure uses.
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const v = Math.max(-1, Math.min(1, values[r][c]));
        const { r: red, g, b } = d3.rgb(d3.interpolateRdBu((v + 1) / 2));
        const at = (r * cols + c) * 4;
        image.data[at] = red;
        image.data[at + 1] = g;
        image.data[at + 2] = b;
        image.data[at + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [values, rows, cols]);

  const um = (v: number) => Math.round(v * l_ref_um);

  return (
    <figure className="qc-sdf">
      <span className="qc-sdf-ytitle mono">y (µm)</span>
      <span className="qc-sdf-ymax mono">{um(y_range[1])}</span>
      <span className="qc-sdf-ymin mono">{um(y_range[0])}</span>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`Signed distance field at t = ${t_ms} ms across ${um(x_range[0])} to ${um(x_range[1])} µm downstream; red inside the vapour, blue outside.`}
      />
      <span className="qc-sdf-xmin mono">{um(x_range[0])}</span>
      <span className="qc-sdf-xmax mono">{um(x_range[1])}</span>
      <figcaption className="qc-sdf-xtitle mono">x (µm), downstream</figcaption>
      <div className="qc-sdf-legend mono" aria-hidden="true">
        <span>−1 vapour</span>
        <span className="qc-sdf-bar" />
        <span>+1 liquid</span>
      </div>
    </figure>
  );
}
