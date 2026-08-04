import * as d3 from "d3";
import { useEffect, useRef } from "react";

import { attachCrosshair } from "./crosshair";

export interface ComparePoint {
  x: number;
  /** Null is a real answer: "nothing was measured here". Lines break across it
   * rather than drawing a straight segment over the gap, which would show a
   * value the data never claimed. */
  y: number | null;
}

/**
 * Palette slots named by the meaning the design system gives each colour, so a
 * caller whose colour carries meaning does not have to know that amber happens
 * to be the fourth entry. Omitting `slot` keeps the positional palette, which is
 * what comparing N runs against each other wants.
 */
export const SERIES_SLOT = {
  primary: 0,
  measured: 1,
  alternate: 2,
  heldout: 3,
} as const;

export interface CompareSeries {
  id: string;
  points: ComparePoint[];
  /** Draw discrete circles instead of a connected line (e.g. the measured
   * camera instants against a continuous reconstruction). */
  markers?: boolean;
  /** Palette slot; defaults to the series' position. Set it from
   * {@link SERIES_SLOT} when the colour means something. */
  slot?: number;
}

/** A named span of the x axis, drawn as a labelled band behind the series --
 * for an axis whose regions mean something (the front's four segments). */
export interface CompareBand {
  start: number;
  end: number;
  label: string;
  /** Marks a span where the data is deliberately absent, so the gap in the
   * series reads as a statement rather than as missing data. */
  muted?: boolean;
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
  /** Labelled spans of the x axis, drawn behind the series. */
  bands?: CompareBand[];
  xLabel: string;
  /** Axis caption drawn top-left (what the y numbers are, with unit). */
  yLabel?: string;
  ariaLabel: string;
  yFormat?: (value: number) => string;
}

type G = d3.Selection<SVGGElement, unknown, null, undefined>;

/**
 * Pair a time axis with a value series, skipping instants where either is null
 * (a gap in the data, not a zero). What every artifact-backed chart needs to
 * turn two parallel arrays into points.
 */
export function toComparePoints(
  t: (number | null)[],
  values: (number | null)[],
): ComparePoint[] {
  const points: ComparePoint[] = [];
  t.forEach((time, i) => {
    const value = values[i];
    if (time != null && value != null) points.push({ x: time, y: value });
  });
  return points;
}

/** The points a scale, a marker or a readout can actually use. */
function defined(points: ComparePoint[]) {
  return points.filter((p): p is { x: number; y: number } => p.y != null);
}

function makeScales(series: CompareSeries[], logY: boolean) {
  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const ys = series.flatMap((s) =>
    defined(s.points).map((p) => (logY ? Math.max(p.y, FLOOR) : p.y)),
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
  bands,
  xLabel,
  yLabel,
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
    // Bands sit behind everything: they name regions of the axis, they are not
    // data.
    (bands ?? []).forEach((band) => {
      const left = x(band.start);
      g.append("rect")
        .attr("class", "chart-band" + (band.muted ? " muted" : ""))
        .attr("x", left)
        .attr("y", 0)
        .attr("width", Math.max(0, x(band.end) - left))
        .attr("height", INNER_H);
      g.append("text")
        .attr("class", "chart-axis chart-band-label")
        .attr("x", (left + x(band.end)) / 2)
        .attr("y", 10)
        .attr("text-anchor", "middle")
        .text(band.label);
    });
    drawAxes(g, x, y, logY);
    if (yLabel)
      g.append("text")
        .attr("class", "chart-axis chart-ylabel")
        .attr("x", 0)
        .attr("y", -4)
        .text(yLabel);

    const paletteClass = (s: CompareSeries, i: number) =>
      `series-${(s.slot ?? i) % 4}`;

    // Lines first, then every marker series on top of them, so discrete
    // samples always read as whole circles rather than notching the curves.
    drawable.forEach((s, i) => {
      if (s.markers) return;
      const line = d3
        .line<ComparePoint>()
        // A null y breaks the line instead of being interpolated across.
        .defined((p) => p.y != null)
        .x((p) => x(p.x))
        .y((p) => y(logY ? Math.max(p.y ?? FLOOR, FLOOR) : (p.y ?? 0)));
      g.append("path")
        .attr("class", `chart-line ${paletteClass(s, i)}`)
        .attr("d", line(s.points) ?? "");
    });
    drawable.forEach((s, i) => {
      if (!s.markers) return;
      g.append("g")
        .selectAll("circle")
        .data(defined(s.points))
        .join("circle")
        .attr("class", `chart-sample ${paletteClass(s, i)}`)
        .attr("cx", (p) => x(p.x))
        .attr("cy", (p) => y(logY ? Math.max(p.y, FLOOR) : p.y))
        .attr("r", 5)
        // The chart scales down with its column; keep the ring stroke crisp
        // instead of letting it thin into an antialiased crescent.
        .attr("vector-effect", "non-scaling-stroke")
        .attr("shape-rendering", "geometricPrecision");
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
        const anchor = nearest.find((entry) => entry.point != null)?.point;
        if (!anchor) return null;
        return {
          xPix: x(anchor.x),
          title: `${xLabel} ${anchor.x}`,
          // A series with nothing at this x reads "not measured" rather than
          // borrowing a neighbour's number.
          rows: nearest.map((entry, i) => ({
            text: `${entry.id}  ${
              entry.point?.y == null ? "not measured" : yFormat(entry.point.y)
            }`,
            swatchClass: paletteClass(drawable[i], i),
          })),
        };
      },
    });
    return hide;
  }, [series, logY, bands, xLabel, yLabel, yFormat]);

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
