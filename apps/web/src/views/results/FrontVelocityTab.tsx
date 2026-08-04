import { Callout, Panel } from "../../components";
import {
  CompareChart,
  type ComparePoint,
} from "../../components/charts/CompareChart";
import type { KinematicsSeries, NoseSpeed } from "../../lib/api";
import { ChartCard } from "./ChartCard";
import { useFrontVelocity } from "./useFrontVelocity";

/** µm/ms is the axis unit (it makes each speed chart the visible slope of the
 * position chart above it); this is the SI value the tooltip adds. */
const M_PER_S_PER_UM_PER_MS = 1e-3;

/** Both units on one readout: the axis' µm/ms, then the SI equivalent. */
export function speedReadout(value: number): string {
  const si = value * M_PER_S_PER_UM_PER_MS;
  return `${value.toFixed(2)} µm/ms (${si.toFixed(3)} m/s)`;
}

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

function NoseSpeedChart({ nose, name }: { nose: NoseSpeed; name: string }) {
  const predicted = toSeries(nose.t_ms, nose.v_um_per_ms);
  return (
    <ChartCard
      title="Nose speed"
      unit="µm/ms"
      name={name}
      rows={predicted.map((point) => ({
        series: "pinn",
        t_ms: point.x,
        v_um_per_ms: point.y,
        v_m_per_s: point.y * M_PER_S_PER_UM_PER_MS,
      }))}
      render={() => (
        <CompareChart
          series={[{ id: "PINN", points: predicted }]}
          xLabel="t (ms)"
          yLabel="nose speed · µm/ms"
          ariaLabel="Nose speed over time, from the continuous reconstruction."
          yFormat={speedReadout}
        />
      )}
    />
  );
}

interface FrontVelocityTabProps {
  runId: string;
  /** The viewing condition of a joint run; null = the run's own (single). */
  dataset?: string | null;
}

/**
 * How fast the interface moved, and where on it.
 *
 * Every other output in this view describes the interface's SHAPE. This one
 * describes its RATE -- the quantity that carries a front-geometry solution past
 * the last supervised frame, and the one a reader otherwise has to recover by
 * differentiating a position curve by eye.
 */
export function FrontVelocityTab({ runId, dataset }: FrontVelocityTabProps) {
  const { report, error, loading } = useFrontVelocity(runId, dataset);
  const exportStem = `${runId}${dataset ? `-${dataset}` : ""}`;

  if (loading)
    return (
      <Panel title="Front velocity" subtitle="how fast the interface moved">
        <p className="state-note" role="status">
          Loading front kinematics…
        </p>
      </Panel>
    );

  if (error)
    return (
      <Callout tone="error" title="Could not load the front velocity">
        {error}
      </Callout>
    );

  if (!report)
    return (
      <Panel title="Front velocity" subtitle="how fast the interface moved">
        <p className="state-note">
          No front kinematics recorded for this run — re-run the evaluate stage
          to measure them.
        </p>
      </Panel>
    );

  return (
    <Panel
      title="Front velocity"
      subtitle="how fast the interface moved · the rate, not the shape"
    >
      <div className="kin-grid">
        <NoseSpeedChart
          nose={report.nose_speed}
          name={`${exportStem}-nose-speed`}
        />
      </div>
    </Panel>
  );
}
