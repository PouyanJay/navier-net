import * as d3 from "d3";
import { useEffect, useRef } from "react";

import type { LossRecord } from "../../lib/api";
import { attachCrosshair } from "./crosshair";

const WIDTH = 640;
const HEIGHT = 210;
const MARGIN = { top: 14, right: 12, bottom: 24, left: 44 };
const INNER_W = WIDTH - MARGIN.left - MARGIN.right;
const INNER_H = HEIGHT - MARGIN.top - MARGIN.bottom;

/** The plotted loss terms, in legend order. Colors come from CSS classes. */
const SERIES = ["data", "vof", "div"] as const;
type SeriesName = (typeof SERIES)[number];

// Log-scale floor: losses are positive but can dip arbitrarily close to zero.
const FLOOR = 1e-12;

type G = d3.Selection<SVGGElement, unknown, null, undefined>;
type XScale = d3.ScaleLinear<number, number>;
type YScale = d3.ScaleLogarithmic<number, number>;

function makeScales(records: LossRecord[]): { x: XScale; y: YScale } {
  const steps = records.map((r) => r.step);
  const values = records.flatMap((r) =>
    SERIES.map((s) => Math.max(r[s], FLOOR)),
  );
  const x = d3
    .scaleLinear()
    .domain([Math.min(...steps), Math.max(...steps)])
    .range([0, INNER_W]);
  const y = d3
    .scaleLog()
    .domain([Math.min(...values), Math.max(...values)])
    .range([INNER_H, 0])
    .nice();
  return { x, y };
}

function drawGridAndYAxis(g: G, y: YScale): void {
  const ticks = y.ticks(4).filter((t) => Number.isInteger(Math.log10(t)));
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
    .text((d) => `1e${Math.round(Math.log10(d))}`);
}

function drawXAxis(g: G, x: XScale): void {
  g.append("g")
    .attr("class", "chart-axis")
    .selectAll("text")
    .data(x.ticks(5))
    .join("text")
    .attr("x", (d) => x(d))
    .attr("y", INNER_H + 16)
    .attr("text-anchor", "middle")
    .text((d) => d);
}

function drawRebalanceMarkers(g: G, x: XScale, markers: number[]): void {
  g.append("g")
    .selectAll("line")
    .data(markers)
    .join("line")
    .attr("class", "chart-marker")
    .attr("x1", (d) => x(d))
    .attr("x2", (d) => x(d))
    .attr("y1", 0)
    .attr("y2", INNER_H);
}

function drawSeries(g: G, x: XScale, y: YScale, records: LossRecord[]): void {
  for (const name of SERIES) {
    const line = d3
      .line<LossRecord>()
      .x((r) => x(r.step))
      .y((r) => y(Math.max(r[name], FLOOR)));
    g.append("path")
      .attr("class", `chart-line ${name}`)
      .attr("d", line(records) ?? "");
  }
}

function drawLegend(g: G): void {
  const legend = g
    .append("g")
    .attr("class", "chart-legend")
    .attr("transform", `translate(${INNER_W - SERIES.length * 52}, -8)`);
  SERIES.forEach((name: SeriesName, i) => {
    const item = legend
      .append("g")
      .attr("transform", `translate(${i * 52}, 0)`);
    item
      .append("rect")
      .attr("class", `chart-swatch ${name}`)
      .attr("width", 10)
      .attr("height", 3);
    item.append("text").attr("x", 14).attr("y", 4).text(name);
  });
}

interface LossChartProps {
  records: LossRecord[];
  /** Steps where gradient-norm rebalancing fired (drawn as dashed markers). */
  rebalanceSteps?: number[];
}

/** Every loss term shown in the crosshair readout (chart lines stay the
 * headline three, matching the mockup; src/bc appear in the readout only). */
const READOUT_TERMS = ["data", "vof", "div", "src", "bc"] as const;

/**
 * Live multi-series loss history on a log₁₀ axis with a crosshair readout of
 * all five terms. D3 owns geometry only; the series colors come from CSS token
 * classes so both themes read correctly on the always-dark view canvas.
 */
export function LossChart({ records, rebalanceSteps = [] }: LossChartProps) {
  const ref = useRef<SVGSVGElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    if (records.length < 2) return; // a line needs two points

    const g = svg
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
    const { x, y } = makeScales(records);
    const [minStep, maxStep] = x.domain();
    drawGridAndYAxis(g, y);
    drawXAxis(g, x);
    drawRebalanceMarkers(
      g,
      x,
      rebalanceSteps.filter((s) => s >= minStep && s <= maxStep),
    );
    drawSeries(g, x, y, records);
    drawLegend(g);

    // Crosshair: every loss term at the nearest logged step.
    const bisect = d3.bisector((r: LossRecord) => r.step).center;
    const hide = attachCrosshair({
      svg,
      g,
      tipEl: tipRef.current,
      width: WIDTH,
      margin: MARGIN,
      innerWidth: INNER_W,
      innerHeight: INNER_H,
      readout: (px) => {
        const record = records[bisect(records, x.invert(px))];
        if (!record) return null;
        return {
          xPix: x(record.step),
          title: `step ${record.step}`,
          rows: READOUT_TERMS.map((term) => ({
            text: `${term}  ${record[term].toExponential(2)}`,
            swatchClass: (SERIES as readonly string[]).includes(term)
              ? term
              : undefined,
          })),
        };
      },
    });
    return hide;
  }, [records, rebalanceSteps]);

  return (
    <div className="chart-wrap">
      <svg
        ref={ref}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Training loss per term over steps, on a logarithmic axis."
      />
      <div ref={tipRef} className="chart-tip" style={{ display: "none" }} />
    </div>
  );
}
