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

    const innerW = WIDTH - MARGIN.left - MARGIN.right;
    const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
    const g = svg.append("g").attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    const x = d3
      .scaleBand<number>()
      .domain(data.map((d) => d.frame))
      .range([0, innerW])
      .padding(0.25);

    const minIou = d3.min(data, (d) => d.iou) ?? 0.9;
    const yMin = Math.min(0.9, Math.floor(minIou * 20) / 20);
    const y = d3.scaleLinear().domain([yMin, 1]).range([innerH, 0]);

    // Horizontal gridlines + y ticks.
    const yTicks = y.ticks(5);
    const grid = g.append("g").attr("class", "chart-grid");
    grid
      .selectAll("line")
      .data(yTicks)
      .join("line")
      .attr("x1", 0)
      .attr("x2", innerW)
      .attr("y1", (d) => y(d))
      .attr("y2", (d) => y(d));

    const yAxis = g.append("g").attr("class", "chart-axis");
    yAxis
      .selectAll("text")
      .data(yTicks)
      .join("text")
      .attr("x", -8)
      .attr("y", (d) => y(d))
      .attr("dy", "0.32em")
      .attr("text-anchor", "end")
      .text((d) => d.toFixed(2));

    // Bars.
    g.append("g")
      .selectAll("rect")
      .data(data)
      .join("rect")
      .attr("class", (d) => (d.holdout ? "chart-bar holdout" : "chart-bar"))
      .attr("x", (d) => x(d.frame) ?? 0)
      .attr("width", x.bandwidth())
      .attr("y", (d) => y(d.iou))
      .attr("height", (d) => innerH - y(d.iou))
      .attr("rx", 2);

    // X labels (frame numbers).
    const xAxis = g.append("g").attr("class", "chart-axis");
    xAxis
      .selectAll("text")
      .data(data)
      .join("text")
      .attr("x", (d) => (x(d.frame) ?? 0) + x.bandwidth() / 2)
      .attr("y", innerH + 18)
      .attr("text-anchor", "middle")
      .text((d) => d.frame);

    g.append("line")
      .attr("class", "chart-baseline")
      .attr("x1", 0)
      .attr("x2", innerW)
      .attr("y1", innerH)
      .attr("y2", innerH);
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
