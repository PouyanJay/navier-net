import * as d3 from "d3";
import { useEffect, useRef } from "react";

export interface FramePoint {
  frame: number;
  iou: number;
  holdout: boolean;
}

const WIDTH = 640;
const HEIGHT = 260;
const MARGIN = { top: 12, right: 12, bottom: 28, left: 40 };
const INNER_W = WIDTH - MARGIN.left - MARGIN.right;
const INNER_H = HEIGHT - MARGIN.top - MARGIN.bottom;

type G = d3.Selection<SVGGElement, unknown, null, undefined>;
type XScale = d3.ScaleBand<number>;
type YScale = d3.ScaleLinear<number, number>;

function makeScales(data: FramePoint[]): { x: XScale; y: YScale } {
  const x = d3
    .scaleBand<number>()
    .domain(data.map((d) => d.frame))
    .range([0, INNER_W])
    .padding(0.25);
  const minIou = d3.min(data, (d) => d.iou) ?? 0.9;
  const yMin = Math.min(0.9, Math.floor(minIou * 20) / 20);
  const y = d3.scaleLinear().domain([yMin, 1]).range([INNER_H, 0]);
  return { x, y };
}

function drawGridAndYAxis(g: G, y: YScale): void {
  const ticks = y.ticks(5);
  g.append("g")
    .attr("class", "chart-grid")
    .selectAll("line")
    .data(ticks)
    .join("line")
    .attr("x1", 0)
    .attr("x2", INNER_W)
    .attr("y1", (d) => y(d))
    .attr("y2", (d) => y(d));
  g.append("g")
    .attr("class", "chart-axis")
    .selectAll("text")
    .data(ticks)
    .join("text")
    .attr("x", -8)
    .attr("y", (d) => y(d))
    .attr("dy", "0.32em")
    .attr("text-anchor", "end")
    .text((d) => d.toFixed(2));
}

function drawBars(g: G, x: XScale, y: YScale, data: FramePoint[]): void {
  g.append("g")
    .selectAll("rect")
    .data(data)
    .join("rect")
    .attr("class", (d) => (d.holdout ? "chart-bar holdout" : "chart-bar"))
    .attr("x", (d) => x(d.frame) ?? 0)
    .attr("width", x.bandwidth())
    .attr("y", (d) => y(d.iou))
    .attr("height", (d) => INNER_H - y(d.iou))
    .attr("rx", 2);
}

function drawXAxisAndBaseline(g: G, x: XScale, data: FramePoint[]): void {
  g.append("g")
    .attr("class", "chart-axis")
    .selectAll("text")
    .data(data)
    .join("text")
    .attr("x", (d) => (x(d.frame) ?? 0) + x.bandwidth() / 2)
    .attr("y", INNER_H + 18)
    .attr("text-anchor", "middle")
    .text((d) => d.frame);
  g.append("line")
    .attr("class", "chart-baseline")
    .attr("x1", 0)
    .attr("x2", INNER_W)
    .attr("y1", INNER_H)
    .attr("y2", INNER_H);
}

/**
 * IoU-per-frame bar chart. D3 owns geometry; color comes from CSS token classes
 * (.chart-bar / .holdout / .chart-grid / .chart-axis) so it reads correctly on
 * the always-dark view canvas in both themes.
 */
export function IouChart({ data }: { data: FramePoint[] }) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    if (data.length === 0) return;

    const g = svg.append("g").attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
    const { x, y } = makeScales(data);
    drawGridAndYAxis(g, y);
    drawBars(g, x, y, data);
    drawXAxisAndBaseline(g, x, data);
  }, [data]);

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Intersection-over-union per frame; the holdout frame is highlighted."
    />
  );
}
