import * as d3 from "d3";

export interface TipRow {
  text: string;
  swatchClass?: string;
}

export interface Readout {
  /** Crosshair x position in inner-chart pixels. */
  xPix: number;
  title: string;
  rows: TipRow[];
}

interface CrosshairOptions {
  svg: d3.Selection<SVGSVGElement | null, unknown, null, undefined>;
  g: d3.Selection<SVGGElement, unknown, null, undefined>;
  tipEl: HTMLDivElement | null;
  width: number;
  margin: { top: number; left: number };
  innerWidth: number;
  innerHeight: number;
  /** Map a pointer x (inner-chart pixels) to the nearest datum's readout. */
  readout: (px: number) => Readout | null;
}

/**
 * Shared pointer crosshair + tooltip for the interactive charts: a dashed
 * cursor line plus a positioned readout of the nearest datum. Returns the
 * hide() cleanup for the caller's effect teardown.
 */
export function attachCrosshair(options: CrosshairOptions): () => void {
  const { svg, g, tipEl, width, margin, innerWidth, innerHeight, readout } = options;
  const crosshair = g
    .append("line")
    .attr("class", "chart-cursor")
    .attr("y1", 0)
    .attr("y2", innerHeight)
    .style("display", "none");
  const tip = d3.select(tipEl);

  const show = (event: PointerEvent) => {
    const [px] = d3.pointer(event, g.node());
    const result = readout(px);
    if (!result) return;
    crosshair.style("display", null).attr("x1", result.xPix).attr("x2", result.xPix);
    tip.style("display", "block").text("");
    tip.append("div").attr("class", "tip-x").text(result.title);
    for (const row of result.rows) {
      const line = tip.append("div").attr("class", "tip-row");
      line.append("i").attr("class", `tip-swatch ${row.swatchClass ?? ""}`.trim());
      line.append("span").text(row.text);
    }
    const bounds = (svg.node() as SVGSVGElement).getBoundingClientRect();
    const scale = bounds.width / width;
    tip
      .style("left", `${(margin.left + result.xPix) * scale + 12}px`)
      .style("top", `${margin.top * scale + 8}px`);
  };
  const hide = () => {
    crosshair.style("display", "none");
    tip.style("display", "none");
  };

  svg
    .append("rect")
    .attr("class", "chart-hover-capture")
    .attr("x", margin.left)
    .attr("y", margin.top)
    .attr("width", innerWidth)
    .attr("height", innerHeight)
    .on("pointermove", show)
    .on("pointerleave", hide);

  return hide;
}
