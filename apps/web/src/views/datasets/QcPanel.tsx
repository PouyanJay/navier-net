import * as d3 from "d3";
import { useEffect, useRef, useState } from "react";

import { Panel, ViewCanvas } from "../../components";
import type { QcData } from "../../lib/api";

const WIDTH = 920;
const MARGIN = { top: 16, right: 20, bottom: 34, left: 52 };

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

/** Measured bubble length per frame with the linear growth fit. */
function KinematicsChart({ qc }: { qc: QcData }) {
  const ref = useRef<SVGSVGElement>(null);
  const { t_ms, length_um, fit_slope_mm_s, fit_intercept_um } = qc.kinematics;
  const height = 300;

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    if (t_ms.length === 0) return;
    const innerW = WIDTH - MARGIN.left - MARGIN.right;
    const innerH = height - MARGIN.top - MARGIN.bottom;
    const g = svg.append("g").attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    const x = d3.scaleLinear().domain(d3.extent(t_ms) as [number, number]).range([0, innerW]);
    const yExtent = d3.extent(length_um) as [number, number];
    const y = d3.scaleLinear().domain(yExtent).nice().range([innerH, 0]);

    g.append("g")
      .attr("class", "chart-grid")
      .selectAll("line")
      .data(y.ticks(5))
      .join("line")
      .attr("x1", 0)
      .attr("x2", innerW)
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
      .attr("y", innerH + 22)
      .attr("text-anchor", "middle")
      .text((d) => `${d} ms`);

    // The dashed fit underneath, then the measured series on top.
    const [t0, t1] = x.domain();
    const fitY = (t: number) => fit_slope_mm_s * t + fit_intercept_um;
    g.append("line")
      .attr("class", "chart-line qc-fit")
      .attr("x1", x(t0))
      .attr("y1", y(fitY(t0)))
      .attr("x2", x(t1))
      .attr("y2", y(fitY(t1)));

    const line = d3
      .line<number>()
      .x((_, i) => x(t_ms[i]))
      .y((d) => y(d));
    g.append("path").attr("class", "chart-line qc-measured").attr("d", line(length_um));
    g.append("g")
      .selectAll("circle")
      .data(length_um)
      .join("circle")
      .attr("class", "qc-dot")
      .attr("cx", (_, i) => x(t_ms[i]))
      .attr("cy", (d) => y(d))
      .attr("r", 3.5)
      .append("title")
      .text((d, i) => `t = ${t_ms[i]} ms — L = ${d.toFixed(0)} µm`);

    g.append("text")
      .attr("class", "chart-axis qc-fit-label")
      .attr("x", innerW - 4)
      .attr("y", 14)
      .attr("text-anchor", "end")
      .text(`fit dL/dt = ${fit_slope_mm_s.toFixed(0)} mm/s`);
  }, [t_ms, length_um, fit_slope_mm_s, fit_intercept_um]);

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${WIDTH} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={`Bubble length per frame with linear fit dL/dt = ${fit_slope_mm_s.toFixed(0)} mm/s.`}
    />
  );
}

/** The α = 0.5 interface polylines, every second frame, oldest to newest. */
function InterfaceChart({ qc }: { qc: QcData }) {
  const ref = useRef<SVGSVGElement>(null);
  const { frames, x_range, y_range, x_pin_star } = qc.interface;
  // Equal x*/y* aspect: height follows the domain's shape.
  const innerW = WIDTH - MARGIN.left - MARGIN.right;
  const innerH = Math.max(
    120,
    Math.round((innerW * (y_range[1] - y_range[0])) / (x_range[1] - x_range[0])),
  );
  const height = innerH + MARGIN.top + MARGIN.bottom;

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
    const x = d3.scaleLinear().domain(x_range).range([0, innerW]);
    const y = d3.scaleLinear().domain(y_range).range([innerH, 0]);
    const tone = d3
      .scaleLinear()
      .domain([0, Math.max(1, frames.length - 1)])
      .range([0.35, 1]);

    g.append("line")
      .attr("class", "qc-pin")
      .attr("x1", x(x_pin_star))
      .attr("x2", x(x_pin_star))
      .attr("y1", 0)
      .attr("y2", innerH);

    for (const [order, frame] of frames.entries()) {
      const group = g.append("g").attr("class", "chart-line qc-contour");
      group.attr("opacity", tone(order));
      for (const contour of frame.contours) {
        group
          .append("path")
          .attr(
            "d",
            d3.line<number[]>().x((p) => x(p[0])).y((p) => y(p[1]))(contour) ?? "",
          )
          .append("title")
          .text(`t = ${frame.t_ms} ms`);
      }
    }

    g.append("text")
      .attr("class", "chart-axis")
      .attr("x", innerW / 2)
      .attr("y", innerH + 24)
      .attr("text-anchor", "middle")
      .text("x* (downstream) — later frames brighter · dotted line: pinned cavity");
  }, [frames, x_range, y_range, x_pin_star, innerW, innerH]);

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
