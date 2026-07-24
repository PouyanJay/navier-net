import * as d3 from "d3";
import { useEffect, useRef } from "react";

import { attachCrosshair } from "./crosshair";

export interface ComparePoint {
  x: number;
  y: number;
}

export interface CompareSeries {
  id: string;
  points: ComparePoint[];
}

const WIDTH = 640;
const HEIGHT = 220;
const MARGIN = { top: 14, right: 12, bottom: 26, left: 48 };
const INNER_W = WIDTH - MARGIN.left - MARGIN.right;
const INNER_H = HEIGHT - MARGIN.top - MARGIN.bottom;

// Log-scale floor for loss values that can approach zero.
const FLOOR = 1e-12;

interface CompareChartProps {
  series: CompareSeries[];
  logY?: boolean;
  xLabel: string;
  ariaLabel: string;
  yFormat?: (value: number) => string;
}

type G = d3.Selection<SVGGElement, unknown, null, undefined>;

function makeScales(series: CompareSeries[], logY: boolean) {
  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const ys = series.flatMap((s) =>
    s.points.map((p) => (logY ? Math.max(p.y, FLOOR) : p.y)),
  );
  const x = d3
    .scaleLinear()
    .domain([Math.min(...xs), Math.max(...xs)])
    .range([0, INNER_W]);
  const y = (logY ? d3.scaleLog() : d3.scaleLinear())
    .domain([Math.min(...ys), Math.max(...ys)])
    .range([INNER_H, 0])
    .nice();
  return { x, y };
}

function drawAxes(
  g: G,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleContinuousNumeric<number, number>,
  logY: boolean,
) {
  const yTicks = logY
    ? y.ticks(4).filter((t) => Number.isInteger(Math.log10(t)))
    : y.ticks(4);
  g.append("g")
    .attr("class", "chart-grid")
    .selectAll("line")
    .data(yTicks)
    .join("line")
    .attr("x1", 0)
    .attr("x2", INNER_W)
    .attr("y1", (d) => y(d))
    .attr("y2", (d) => y(d));
  g.append("g")
    .attr("class", "chart-axis")
    .selectAll("text")
    .data(yTicks)
    .join("text")
    .attr("x", -8)
    .attr("y", (d) => y(d))
    .attr("dy", "0.32em")
    .attr("text-anchor", "end")
    .text((d) => (logY ? `1e${Math.round(Math.log10(d))}` : String(d)));
  g.append("g")
    .attr("class", "chart-axis")
    .selectAll("text")
    .data(x.ticks(5))
    .join("text")
    .attr("x", (d) => x(d))
    .attr("y", INNER_H + 18)
    .attr("text-anchor", "middle")
    .text((d) => d);
}

/**
 * Interactive multi-series line chart for run comparison. Series colors come
 * from CSS token classes (`.series-N`); a pointer crosshair reads out every
 * series' value at the nearest x. D3 owns geometry only.
 */
export function CompareChart({
  series,
  logY = false,
  xLabel,
  ariaLabel,
  yFormat = (v) => v.toPrecision(3),
}: CompareChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    const drawable = series.filter((s) => s.points.length > 0);
    if (drawable.length === 0) return;

    const g = svg
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
    const { x, y } = makeScales(drawable, logY);
    drawAxes(g, x, y, logY);

    drawable.forEach((s, i) => {
      const line = d3
        .line<ComparePoint>()
        .x((p) => x(p.x))
        .y((p) => y(logY ? Math.max(p.y, FLOOR) : p.y));
      g.append("path")
        .attr("class", `chart-line series-${i % 4}`)
        .attr("d", line(s.points) ?? "");
    });

    // Crosshair + tooltip: nearest-x readout across every series.
    const hide = attachCrosshair({
      svg,
      g,
      tipEl: tipRef.current,
      width: WIDTH,
      margin: MARGIN,
      innerWidth: INNER_W,
      innerHeight: INNER_H,
      readout: (px) => {
        const xValue = x.invert(px);
        const nearest = drawable.map((s) => {
          const idx = d3
            .bisector((p: ComparePoint) => p.x)
            .center(s.points, xValue);
          return { id: s.id, point: s.points[idx] };
        });
        const anchor = nearest[0]?.point;
        if (!anchor) return null;
        return {
          xPix: x(anchor.x),
          title: `${xLabel} ${anchor.x}`,
          rows: nearest.map((entry, i) => ({
            text: `${entry.id}  ${yFormat(entry.point.y)}`,
            swatchClass: `series-${i % 4}`,
          })),
        };
      },
    });
    return hide;
  }, [series, logY, xLabel, yFormat]);

  return (
    <div className="chart-wrap">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={ariaLabel}
      />
      <div ref={tipRef} className="chart-tip" style={{ display: "none" }} />
    </div>
  );
}
