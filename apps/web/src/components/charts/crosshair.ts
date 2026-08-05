import * as d3 from "d3";

export interface TipRow {
  text: string;
  swatchClass?: string;
}

export interface Readout {
  /** Anchor x, in the chart's inner coordinates. */
  xPix: number;
  /** Anchor y, in the chart's inner coordinates. Defaults to the plot top. */
  yPix?: number;
  title: string;
  rows: TipRow[];
  /** Which datum this is, so the caller can emphasise the mark drawn for it. */
  index?: number;
}

interface TipGeometry {
  /** The svg's viewBox width, to convert its units into rendered pixels. */
  width: number;
  margin: { top: number; left: number };
}

export interface Tip {
  show(readout: Readout): void;
  hide(): void;
}

/** How far the tip sits from its anchor, in rendered pixels. */
const TIP_OFFSET = 12;

/**
 * The floating readout the interactive charts share.
 *
 * It is an HTML element beside the svg rather than a node inside it, so its
 * type is set in the page's own pixels instead of being scaled by the viewBox
 * along with the plot.
 */
export function makeTip(
  tipEl: HTMLDivElement | null,
  svgNode: SVGSVGElement | null,
  geometry: TipGeometry,
): Tip {
  const tip = d3.select(tipEl);
  return {
    show(readout) {
      // null, not "block": an inline display would beat the stylesheet, and a
      // chart that lays its readout out differently sets that in CSS.
      tip.style("display", null).text("");
      tip.append("div").attr("class", "tip-x").text(readout.title);
      for (const row of readout.rows) {
        const line = tip.append("div").attr("class", "tip-row");
        line
          .append("i")
          .attr("class", `tip-swatch ${row.swatchClass ?? ""}`.trim());
        line.append("span").text(row.text);
      }

      const bounds = svgNode?.getBoundingClientRect();
      const scale = bounds?.width ? bounds.width / geometry.width : 1;
      const anchor = (geometry.margin.left + readout.xPix) * scale;
      // Flip to the anchor's other side rather than run off the plot: a
      // readout for the last frame is exactly the one most likely to overflow.
      const tipWidth = tipEl?.offsetWidth ?? 0;
      const overflows =
        !!bounds &&
        bounds.width > 0 &&
        anchor + TIP_OFFSET + tipWidth > bounds.width;
      tip
        .style(
          "left",
          `${overflows ? anchor - TIP_OFFSET - tipWidth : anchor + TIP_OFFSET}px`,
        )
        .style(
          "top",
          `${(geometry.margin.top + (readout.yPix ?? 0)) * scale + 8}px`,
        );
    },
    hide() {
      tip.style("display", "none");
    },
  };
}

/**
 * Announce the focused datum to a screen reader.
 *
 * A crosshair is a pointer affordance and says nothing to anyone not using
 * one, so the same readout is written into a live region as a sentence.
 */
function announce(live: HTMLElement | null, readout: Readout | null): void {
  if (!live) return;
  live.textContent = readout
    ? `${readout.title}. ${readout.rows.map((row) => row.text).join(". ")}`
    : "";
}

/** Arrow-key stepping over the data, so the chart is not pointer-only. */
interface KeyboardSpec {
  count: number;
  at: (index: number) => Readout | null;
  live: HTMLElement | null;
}

interface CrosshairOptions {
  svg: d3.Selection<SVGSVGElement | null, unknown, null, undefined>;
  g: d3.Selection<SVGGElement, unknown, null, undefined>;
  tipEl: HTMLDivElement | null;
  width: number;
  margin: { top: number; left: number };
  innerWidth: number;
  innerHeight: number;
  /** Map a pointer x (inner-chart units) to the nearest datum's readout. */
  readout: (px: number) => Readout | null;
  /** Supply this to make the chart keyboard-navigable. */
  keyboard?: KeyboardSpec;
  /** Fires when the active datum changes, to emphasise the mark drawn for it. */
  onActive?: (readout: Readout | null) => void;
}

/** Move `index` by the pressed key, or return null if the key is not ours. */
function nextIndex(key: string, index: number, count: number): number | null {
  if (key === "ArrowRight" || key === "ArrowUp") return index + 1;
  if (key === "ArrowLeft" || key === "ArrowDown") return index - 1;
  if (key === "Home") return 0;
  if (key === "End") return count - 1;
  return null;
}

/**
 * Shared pointer crosshair and tooltip: a dashed cursor line plus a positioned
 * readout of the nearest datum, optionally steppable from the keyboard.
 * Returns the hide() cleanup for the caller's effect teardown.
 */
export function attachCrosshair(options: CrosshairOptions): () => void {
  const { svg, g, tipEl, width, margin, innerWidth, innerHeight } = options;
  const { readout, keyboard, onActive } = options;

  const crosshair = g
    .append("line")
    .attr("class", "chart-cursor")
    .attr("y1", 0)
    .attr("y2", innerHeight)
    .style("display", "none");
  const tip = makeTip(tipEl, svg.node(), { width, margin });

  const paint = (result: Readout | null) => {
    if (!result) return;
    crosshair
      .style("display", null)
      .attr("x1", result.xPix)
      .attr("x2", result.xPix);
    tip.show(result);
    onActive?.(result);
  };
  const hide = () => {
    crosshair.style("display", "none");
    tip.hide();
    onActive?.(null);
    announce(keyboard?.live ?? null, null);
  };

  svg
    .append("rect")
    .attr("class", "chart-hover-capture")
    .attr("x", margin.left)
    .attr("y", margin.top)
    .attr("width", innerWidth)
    .attr("height", innerHeight)
    .on("pointermove", (event: PointerEvent) => {
      const [px] = d3.pointer(event, g.node());
      paint(readout(px));
    })
    .on("pointerleave", hide);

  if (keyboard && keyboard.count > 0) {
    let index = 0;
    const step = (to: number) => {
      index = Math.max(0, Math.min(keyboard.count - 1, to));
      const result = keyboard.at(index);
      paint(result);
      announce(keyboard.live, result);
    };
    svg
      .attr("tabindex", 0)
      .on("focus", () => step(index))
      .on("blur", hide)
      .on("keydown", (event: KeyboardEvent) => {
        if (event.key === "Escape") {
          hide();
          return;
        }
        const to = nextIndex(event.key, index, keyboard.count);
        if (to === null) return;
        event.preventDefault();
        step(to);
      });
  }

  return hide;
}
