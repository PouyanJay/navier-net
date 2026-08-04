import type { ReactNode } from "react";

import { ViewCanvas } from "../../components";
import { ChartFrame } from "../../components/ChartFrame";

interface ChartCardProps {
  /** Heading above the chart. */
  title: string;
  /** The y quantity's unit, typeset in mono beside the heading. */
  unit: string;
  /** File stem for every download ("nose-speed" → nose-speed.png…). */
  name: string;
  /** The charted data, for CSV/JSON export; omit to hide the data buttons. */
  rows?: Record<string, unknown>[];
  /** Canvas-drawn charts have no meaningful SVG export. */
  raster?: boolean;
  /** Called for the inline view AND the expanded modal, so the modal renders a
   * live instance rather than a stale copy. */
  render: (expanded: boolean) => ReactNode;
}

/**
 * One titled chart on the dark view canvas, with its unit and the standard
 * export controls. The shape every kinematics chart shares -- extracted so the
 * heading, the unit's typography and the canvas cannot drift between panels.
 */
export function ChartCard({
  title,
  unit,
  name,
  rows,
  raster,
  render,
}: ChartCardProps) {
  return (
    <div className="kin-chart">
      <div className="kin-chart-head">
        <h3>{title}</h3>
        <span className="kin-unit">{unit}</span>
      </div>
      <ChartFrame
        name={name}
        title={title}
        rows={rows}
        raster={raster}
        render={(expanded) => <ViewCanvas>{render(expanded)}</ViewCanvas>}
      />
    </div>
  );
}
