import { useEffect, useState } from "react";

import { Panel, SelectField, ViewCanvas } from "../../components";
import { CompareChart } from "../../components/charts/CompareChart";
import { api, type Trajectory } from "../../lib/api";
import { errorMessage } from "../../lib/errors";

type Load =
  | { status: "loading" }
  | { status: "unavailable" }
  | { status: "error"; message: string }
  | { status: "ready"; trajectory: Trajectory };

type Quantity = "nose" | "area";

const toSeries = (t: number[], values: number[]) =>
  t.map((time, i) => ({ x: time, y: values[i] }));

/** Nose position / vapor area over time: the PINN's continuous curve vs the
 * measured camera instants — the trajectories figure, as an interactive chart. */
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
        const message = errorMessage(err);
        setLoad(
          message.includes("no trajectory")
            ? { status: "unavailable" }
            : { status: "error", message },
        );
      });
    return () => {
      alive = false;
    };
  }, [runId]);

  return (
    <Panel title="Growth kinematics" subtitle="continuous reconstruction vs camera instants">
      {load.status === "loading" && (
        <p className="state-note" role="status">
          Loading kinematics…
        </p>
      )}
      {load.status === "unavailable" && (
        <p className="state-note">
          No kinematics recorded — re-run the evaluate stage to produce them.
        </p>
      )}
      {load.status === "error" && (
        <p className="state-note error" role="alert">
          Could not load kinematics: {load.message}
        </p>
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
                    quantity === "nose" ? load.trajectory.nose_um : load.trajectory.area_um2,
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
