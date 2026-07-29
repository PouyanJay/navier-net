import { useEffect, useState } from "react";

import { Callout, Panel, ViewCanvas } from "../../components";
import { ChartFrame } from "../../components/ChartFrame";
import {
  CompareChart,
  type ComparePoint,
} from "../../components/charts/CompareChart";
import {
  api,
  ApiError,
  type KinematicsSeries,
  type Trajectory,
} from "../../lib/api";
import { errorMessage } from "../../lib/errors";

type Load =
  | { status: "loading" }
  | { status: "unavailable" }
  | { status: "error"; message: string }
  | { status: "ready"; trajectory: Trajectory };

/** Pair times with values, skipping instants where either is null (a gap). */
function toSeries(
  t: KinematicsSeries,
  values: KinematicsSeries,
): ComparePoint[] {
  const points: ComparePoint[] = [];
  t.forEach((time, i) => {
    const value = values[i];
    if (time != null && value != null) points.push({ x: time, y: value });
  });
  return points;
}

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
    ...toSeries(trajectory.t_ms, trajectory[field]).map((point) => ({
      series: "pinn",
      t_ms: point.x,
      [field]: point.y,
    })),
    ...toSeries(trajectory.measured.t_ms, trajectory.measured[field]).map(
      (point) => ({ series: "measured", t_ms: point.x, [field]: point.y }),
    ),
  ];
  return (
    <div className="kin-chart">
      <div className="kin-chart-head">
        <h3>{title}</h3>
        <span className="kin-unit">{unit}</span>
      </div>
      <ChartFrame
        name={exportName}
        title={title}
        rows={rows}
        render={() => (
          <ViewCanvas>
            <CompareChart
              series={[
                {
                  id: "PINN",
                  points: toSeries(trajectory.t_ms, trajectory[field]),
                },
                {
                  id: "measured",
                  points: toSeries(
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
          </ViewCanvas>
        )}
      />
    </div>
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
