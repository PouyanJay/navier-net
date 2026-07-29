import { useEffect, useRef, useState, type ReactNode } from "react";

import {
  canvasToPng,
  downloadBlob,
  downloadText,
  rowsToCsv,
  serializeSvg,
  svgTextToPng,
} from "../lib/chartExport";
import { useToast } from "./Toast";

interface ChartFrameProps {
  /** File stem for every download ("nose-position" → nose-position.png…). */
  name: string;
  /** Modal heading when expanded. */
  title: string;
  /** The charted data, for CSV/JSON export; omit to hide the data buttons. */
  rows?: Record<string, unknown>[];
  /** JSON payload override (defaults to the rows). */
  json?: unknown;
  /** Raster charts (canvas) have no meaningful SVG export. */
  raster?: boolean;
  /** Renders the chart — called for the inline view AND the expanded modal,
   * so the modal gets a live instance, not a stale copy. */
  render: (expanded: boolean) => ReactNode;
}

/**
 * The frame every chart lives in: the chart itself plus its standard controls
 * — expand into a modal, download as high-res PNG / standalone SVG, and take
 * the underlying data as CSV / JSON. One component so every graph in the app
 * offers the same affordances.
 */
export function ChartFrame({
  name,
  title,
  rows,
  json,
  raster = false,
  render,
}: ChartFrameProps) {
  const host = useRef<HTMLDivElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);
  const toast = useToast();

  useEffect(() => {
    if (expanded) dialog.current?.focus();
  }, [expanded]);

  const canvasBackground = () =>
    getComputedStyle(document.documentElement)
      .getPropertyValue("--view")
      .trim() || "#11131a";

  const exportImage = async (kind: "png" | "svg") => {
    try {
      const svg = host.current?.querySelector("svg");
      const canvas = host.current?.querySelector("canvas");
      if (svg) {
        const text = serializeSvg(svg, canvasBackground());
        if (kind === "svg") {
          downloadText(`${name}.svg`, text, "image/svg+xml");
          return;
        }
        const box = svg.viewBox.baseVal;
        downloadBlob(
          `${name}.png`,
          await svgTextToPng(text, box.width, box.height),
        );
        return;
      }
      if (canvas && kind === "png") {
        downloadBlob(`${name}.png`, await canvasToPng(canvas));
        return;
      }
      toast("Nothing to export", "the chart has not rendered yet", "err");
    } catch (exc: unknown) {
      toast(
        "Export failed",
        exc instanceof Error ? exc.message : String(exc),
        "err",
      );
    }
  };

  const exportData = (kind: "csv" | "json") => {
    if (!rows) return;
    if (kind === "csv")
      downloadText(`${name}.csv`, rowsToCsv(rows), "text/csv");
    else
      downloadText(
        `${name}.json`,
        JSON.stringify(json ?? rows, null, 1),
        "application/json",
      );
  };

  const toolbar = (
    <div className="cf-bar" role="group" aria-label={`${title} chart actions`}>
      <button
        type="button"
        className="cf-btn"
        onClick={() => setExpanded(true)}
        title="Expand the chart"
      >
        Expand
      </button>
      <button
        type="button"
        className="cf-btn"
        onClick={() => void exportImage("png")}
        title="Download as high-resolution PNG"
      >
        PNG
      </button>
      {!raster && (
        <button
          type="button"
          className="cf-btn"
          onClick={() => void exportImage("svg")}
          title="Download as standalone SVG"
        >
          SVG
        </button>
      )}
      {rows && (
        <>
          <button
            type="button"
            className="cf-btn"
            onClick={() => exportData("csv")}
            title="Download the charted data as CSV"
          >
            CSV
          </button>
          <button
            type="button"
            className="cf-btn"
            onClick={() => exportData("json")}
            title="Download the charted data as JSON"
          >
            JSON
          </button>
        </>
      )}
    </div>
  );

  return (
    <div className="chart-frame">
      {toolbar}
      <div ref={host}>{render(false)}</div>
      {expanded && (
        <div
          className="modal-ov"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setExpanded(false);
          }}
        >
          <div
            ref={dialog}
            tabIndex={-1}
            className="modal chart-modal"
            role="dialog"
            aria-modal="true"
            aria-label={title}
            onKeyDown={(event) => {
              if (event.key === "Escape") setExpanded(false);
            }}
          >
            <div className="hd">
              <h2>{title}</h2>
              <button
                type="button"
                className="cf-btn"
                onClick={() => setExpanded(false)}
              >
                Close
              </button>
            </div>
            <div className="chart-modal-body">{render(true)}</div>
          </div>
        </div>
      )}
    </div>
  );
}
