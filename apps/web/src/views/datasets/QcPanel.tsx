import * as d3 from "d3";
import { useEffect, useRef, useState } from "react";

import { Panel, ViewCanvas } from "../../components";
import type { QcData, QcKinematics } from "../../lib/api";

const WIDTH = 920;
const MARGIN = { top: 16, right: 20, bottom: 34, left: 52 };
const INNER_W = WIDTH - MARGIN.left - MARGIN.right;

type G = d3.Selection<SVGGElement, unknown, null, undefined>;
type Linear = d3.ScaleLinear<number, number>;

type Check = "kinematics" | "interface" | "sdf";

const CHECKS: { id: Check; label: string; sub: string }[] = [
  { id: "kinematics", label: "Growth kinematics", sub: "L(t) + fit" },
  { id: "interface", label: "Interface evolution", sub: "α = 0.5 contours" },
  { id: "sdf", label: "Signed distance", sub: "mid frame" },
];

/** The three preprocessing checks as interactive charts behind one switch. */
export function QcPanel({ qc }: { qc: QcData }) {
  const [check, setCheck] = useState<Check>("kinematics");
  return (
    <Panel
      title="Preprocessing QC"
      subtitle="computed from the training tensors — inspect before solving"
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

// ── Growth kinematics ────────────────────────────────────────────────────────

const KIN_HEIGHT = 300;
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

function drawKinGridAndAxes(g: G, x: Linear, y: Linear): void {
  g.append("g")
    .attr("class", "chart-grid")
    .selectAll("line")
    .data(y.ticks(5))
    .join("line")
    .attr("x1", 0)
    .attr("x2", INNER_W)
    .attr("y1", (d) => y(d))
    .attr("y2", (d) => y(d));
  g.append("g")
    .attr("class", "chart-axis")
    .selectAll("text")
    .data(y.ticks(5))
    .join("text")
    .attr("x", -8)
    .attr("y", (d) => y(d))
    .attr("dy", "0.32em")
    .attr("text-anchor", "end")
    .text((d) => d3.format("~s")(d));
  g.append("g")
    .attr("class", "chart-axis")
    .selectAll("text")
    .data(x.ticks(6))
    .join("text")
    .attr("x", (d) => x(d))
    .attr("y", KIN_INNER_H + 22)
    .attr("text-anchor", "middle")
    .text((d) => `${d} ms`);
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
    .attr("class", "chart-axis qc-fit-label")
    .attr("x", INNER_W - 4)
    .attr("y", 14)
    .attr("text-anchor", "end")
    .text(`fit dL/dt = ${kin.fit_slope_mm_s.toFixed(0)} mm/s`);
}

function drawMeasuredSeries(g: G, x: Linear, y: Linear, kin: QcKinematics): void {
  const line = d3
    .line<number>()
    .x((_, i) => x(kin.t_ms[i]))
    .y((d) => y(d));
  g.append("path").attr("class", "chart-line qc-measured").attr("d", line(kin.length_um));
  g.append("g")
    .selectAll("circle")
    .data(kin.length_um)
    .join("circle")
    .attr("class", "qc-dot")
    .attr("cx", (_, i) => x(kin.t_ms[i]))
    .attr("cy", (d) => y(d))
    .attr("r", 3.5)
    .append("title")
    .text((d, i) => `t = ${kin.t_ms[i]} ms — L = ${d.toFixed(0)} µm`);
}

/** Measured bubble length per frame with the linear growth fit. */
function KinematicsChart({ qc }: { qc: QcData }) {
  const ref = useRef<SVGSVGElement>(null);
  const kin = qc.kinematics;

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    if (kin.t_ms.length === 0) return;
    const g = svg.append("g").attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
    const { x, y } = kinScales(kin);
    drawKinGridAndAxes(g, x, y);
    drawFitLine(g, x, y, kin);
    drawMeasuredSeries(g, x, y, kin);
  }, [kin]);

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${WIDTH} ${KIN_HEIGHT}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={`Bubble length per frame with linear fit dL/dt = ${kin.fit_slope_mm_s.toFixed(0)} mm/s.`}
    />
  );
}

// ── Interface evolution ──────────────────────────────────────────────────────

function drawPin(g: G, x: Linear, xPin: number, innerH: number): void {
  g.append("line")
    .attr("class", "qc-pin")
    .attr("x1", x(xPin))
    .attr("x2", x(xPin))
    .attr("y1", 0)
    .attr("y2", innerH);
}

function drawContourFrames(g: G, x: Linear, y: Linear, qc: QcData): void {
  const frames = qc.interface.frames;
  const tone = d3
    .scaleLinear()
    .domain([0, Math.max(1, frames.length - 1)])
    .range([0.35, 1]);
  const path = d3.line<number[]>().x((p) => x(p[0])).y((p) => y(p[1]));
  for (const [order, frame] of frames.entries()) {
    const group = g.append("g").attr("class", "chart-line qc-contour");
    group.attr("opacity", tone(order));
    for (const contour of frame.contours) {
      group
        .append("path")
        .attr("d", path(contour) ?? "")
        .append("title")
        .text(`t = ${frame.t_ms} ms`);
    }
  }
}

/** The α = 0.5 interface polylines, every second frame, oldest to newest. */
function InterfaceChart({ qc }: { qc: QcData }) {
  const ref = useRef<SVGSVGElement>(null);
  const { x_range, y_range, x_pin_star, frames } = qc.interface;
  // Equal x*/y* aspect: height follows the domain's shape.
  const innerH = Math.max(
    120,
    Math.round((INNER_W * (y_range[1] - y_range[0])) / (x_range[1] - x_range[0])),
  );
  const height = innerH + MARGIN.top + MARGIN.bottom;

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
    const x = d3.scaleLinear().domain(x_range).range([0, INNER_W]);
    const y = d3.scaleLinear().domain(y_range).range([innerH, 0]);
    drawPin(g, x, x_pin_star, innerH);
    drawContourFrames(g, x, y, qc);
    g.append("text")
      .attr("class", "chart-axis")
      .attr("x", INNER_W / 2)
      .attr("y", innerH + 24)
      .attr("text-anchor", "middle")
      .text("x* (downstream) — later frames brighter · dotted line: pinned cavity");
  }, [qc, x_range, y_range, x_pin_star, innerH]);

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${WIDTH} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={`Interface contours for ${frames.length} frames; later frames drawn brighter.`}
    />
  );
}

// ── Signed distance field ────────────────────────────────────────────────────

/** Diverging heatmap of the mid-frame signed distance field. */
function SdfChart({ qc }: { qc: QcData }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { values, t_ms } = qc.sdf;
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

  return (
    <div className="qc-sdf">
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`Signed distance field at t = ${t_ms} ms; red inside the vapour, blue outside.`}
      />
      <div className="qc-sdf-legend mono" aria-hidden="true">
        <span>−1 vapour</span>
        <span className="qc-sdf-bar" />
        <span>+1 liquid</span>
      </div>
    </div>
  );
}
