import { useEffect, useState } from "react";

import { Callout, Panel, SelectField, ViewCanvas } from "../../components";
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

type Quantity = "nose" | "area";

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

/** Nose position / vapor area over time: the PINN's continuous curve vs the
 * measured camera instants: the trajectories figure, as an interactive chart. */
export function GrowthKinematics({ runId }: { runId: string }) {
  const [load, setLoad] = useState<Load>({ status: "loading" });
  const [quantity, setQuantity] = useState<Quantity>("nose");

  useEffect(() => {
    let alive = true;
    setLoad({ status: "loading" });
    api
      .getTrajectory(runId)
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
  }, [runId]);

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
          <div className="compare-chart-head">
            <h3 className="compare-h">
              {quantity === "nose" ? "Nose position (µm)" : "Vapor area (µm²)"}
            </h3>
            <SelectField
              label="Quantity"
              value={quantity}
              onChange={(value) => setQuantity(value as Quantity)}
              options={[
                { value: "nose", label: "nose position · µm" },
                { value: "area", label: "vapor area · µm²" },
              ]}
            />
          </div>
          <ViewCanvas>
            <CompareChart
              series={[
                {
                  id: "PINN",
                  points: toSeries(
                    load.trajectory.t_ms,
                    quantity === "nose"
                      ? load.trajectory.nose_um
                      : load.trajectory.area_um2,
                  ),
                },
                {
                  id: "measured",
                  points: toSeries(
                    load.trajectory.measured.t_ms,
                    quantity === "nose"
                      ? load.trajectory.measured.nose_um
                      : load.trajectory.measured.area_um2,
                  ),
                },
              ]}
              xLabel="t (ms)"
              ariaLabel="Growth kinematics: PINN reconstruction versus measured camera instants."
              yFormat={(v) => v.toFixed(0)}
            />
          </ViewCanvas>
        </>
      )}
    </Panel>
  );
}
