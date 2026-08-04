import { useEffect, useState } from "react";

import { Callout, Panel } from "../../components";
import {
  CompareChart,
  toComparePoints,
} from "../../components/charts/CompareChart";
import { api, ApiError, type Trajectory } from "../../lib/api";
import { errorMessage } from "../../lib/errors";
import { ChartCard } from "./ChartCard";

type Load =
  | { status: "loading" }
  | { status: "unavailable" }
  | { status: "error"; message: string }
  | { status: "ready"; trajectory: Trajectory };

interface KinematicsPanelProps {
  runId: string;
  /** Kinematics dataset scope (joint runs); null = the run's own (single). */
  dataset?: string | null;
}

/** One quantity's chart: the PINN's continuous curve vs camera instants. */
function QuantityChart({
  title,
  unit,
  trajectory,
  field,
  exportName,
}: {
  title: string;
  unit: string;
  trajectory: Trajectory;
  field: "nose_um" | "area_um2";
  exportName: string;
}) {
  const rows = [
    ...toComparePoints(trajectory.t_ms, trajectory[field]).map((point) => ({
      series: "pinn",
      t_ms: point.x,
      [field]: point.y,
    })),
    ...toComparePoints(
      trajectory.measured.t_ms,
      trajectory.measured[field],
    ).map((point) => ({ series: "measured", t_ms: point.x, [field]: point.y })),
  ];
  return (
    <ChartCard
      title={title}
      unit={unit}
      name={exportName}
      rows={rows}
      render={() => (
        <CompareChart
          series={[
            {
              id: "PINN",
              points: toComparePoints(trajectory.t_ms, trajectory[field]),
            },
            {
              id: "measured",
              points: toComparePoints(
                trajectory.measured.t_ms,
                trajectory.measured[field],
              ),
              markers: true,
            },
          ]}
          xLabel="t (ms)"
          yLabel={`${title.toLowerCase()} · ${unit}`}
          ariaLabel={`${title}: continuous PINN curve versus measured camera instants (circles).`}
          yFormat={(v) => v.toFixed(0)}
        />
      )}
    />
  );
}

/** Nose position and vapor area over time, side by side (the trajectories
 * figure as interactive charts; one axis per quantity, never dual-axis). */
export function KinematicsPanel({ runId, dataset }: KinematicsPanelProps) {
  const [load, setLoad] = useState<Load>({ status: "loading" });

  useEffect(() => {
    let alive = true;
    setLoad({ status: "loading" });
    api
      .getTrajectory(runId, dataset ?? undefined)
      .then((trajectory) => alive && setLoad({ status: "ready", trajectory }))
      .catch((err) => {
        if (!alive) return;
        // A 404 means the evaluate stage hasn't produced kinematics yet.
        setLoad(
          err instanceof ApiError && err.status === 404
            ? { status: "unavailable" }
            : { status: "error", message: errorMessage(err) },
        );
      });
    return () => {
      alive = false;
    };
  }, [runId, dataset]);

  return (
    <Panel
      title="Growth kinematics"
      subtitle="continuous reconstruction vs camera instants"
    >
      {load.status === "loading" && (
        <p className="state-note" role="status">
          Loading kinematics…
        </p>
      )}
      {load.status === "unavailable" && (
        <p className="state-note">
          No kinematics recorded; re-run the evaluate stage to produce them.
        </p>
      )}
      {load.status === "error" && (
        <Callout tone="error" title="Could not load kinematics">
          {load.message}
        </Callout>
      )}
      {load.status === "ready" && (
        <>
          <div className="kin-grid">
            <QuantityChart
              title="Nose position"
              unit="µm"
              trajectory={load.trajectory}
              field="nose_um"
              exportName={`${runId}${dataset ? `-${dataset}` : ""}-nose`}
            />
            <QuantityChart
              title="Vapor area"
              unit="µm²"
              trajectory={load.trajectory}
              field="area_um2"
              exportName={`${runId}${dataset ? `-${dataset}` : ""}-area`}
            />
          </div>
          <p className="figcap">
            <b>Figure 3.</b> The reconstruction is continuous between camera
            instants; the nose's mean slope over the steady growth is the
            inferred nose speed the physics checks compare against.
          </p>
        </>
      )}
    </Panel>
  );
}
