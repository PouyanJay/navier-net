import { useEffect, useRef, useState, type ReactNode } from "react";

import {
  canvasToPng,
  downloadBlob,
  downloadText,
  rowsToCsv,
  serializeSvg,
  svgTextToPng,
} from "../lib/chartExport";
import { DownloadIcon, ExpandIcon, HugeiconsIcon } from "./icons";
import { MenuButton, type MenuAction } from "./MenuButton";
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
  /**
   * The chart's own controls, led into the toolbar row ahead of the export
   * actions. A control that changes what is plotted belongs beside the chart
   * it changes, and sharing this row keeps a card to one strip of controls
   * rather than two stacked ones.
   */
  controls?: ReactNode;
  /**
   * What the chart found, taken to the far end of the toolbar row. The
   * controls cluster on the left and the reading closes the row, so the number
   * lands directly above the plot that evidences it.
   */
  trailing?: ReactNode;
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
  controls,
  trailing,
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

  /* One menu rather than four naked buttons. The exports are a single idea —
     "take a copy of this" — and spelling each format out in the toolbar gave
     that idea more width than the control that changes what is plotted. */
  const downloads: MenuAction[] = [
    {
      id: "png",
      label: "PNG image",
      hint: "high resolution",
      onSelect: () => void exportImage("png"),
    },
    ...(raster
      ? []
      : [
          {
            id: "svg",
            label: "SVG image",
            hint: "vector, standalone",
            onSelect: () => void exportImage("svg"),
          },
        ]),
    ...(rows
      ? [
          {
            id: "csv",
            label: "CSV data",
            hint: "the plotted values",
            onSelect: () => exportData("csv"),
          },
          {
            id: "json",
            label: "JSON data",
            hint: "the plotted values",
            onSelect: () => exportData("json"),
          },
        ]
      : []),
  ];

  const toolbar = (
    <div className="cf-bar">
      <div className="cf-lead">
        {controls}
        <MenuButton
          label={`Download ${title}`}
          text="Download"
          icon={
            <HugeiconsIcon icon={DownloadIcon} size={14} aria-hidden="true" />
          }
          actions={downloads}
        />
      </div>
      {trailing}
    </div>
  );

  return (
    <div className="chart-frame">
      {toolbar}
      <div className="cf-plot" ref={host}>
        {render(false)}
        {/* On the plot, not in the toolbar: it acts on the picture, and the
            picture is what you are pointing at when you want it bigger. */}
        <button
          type="button"
          className="cf-expand"
          aria-label={`Expand ${title}`}
          title={`Expand ${title}`}
          onClick={() => setExpanded(true)}
        >
          <HugeiconsIcon icon={ExpandIcon} size={15} aria-hidden="true" />
        </button>
      </div>
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
