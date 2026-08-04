import { Callout, Panel } from "../../components";
import {
  CompareChart,
  SERIES_SLOT,
  type ComparePoint,
  type CompareSeries,
} from "../../components/charts/CompareChart";
import type { KinematicsSeries, NoseSpeed } from "../../lib/api";
import { ChartCard } from "./ChartCard";
import { useFrontVelocity } from "./useFrontVelocity";

/** µm/ms is the axis unit -- it makes each speed chart the visible slope of the
 * position chart above it. This is the SI value the tooltip and export add. */
const M_PER_S_PER_UM_PER_MS = 1e-3;

/** Three significant figures, the register the physics tiles already use. Zero
 * is written as itself rather than as "0.00". */
const sigFigs = (value: number) => (value === 0 ? "0" : value.toPrecision(3));

/** Both units on one readout: the axis' µm/ms, then the SI equivalent.
 *
 * Significant figures rather than a fixed decimal count, because these charts
 * span both the fast nose and the near-stationary flanks -- two decimals would
 * render every flank reading as a flat "0.00" and hide the very contrast the
 * profile chart exists to show. */
export function speedReadout(value: number): string {
  const si = sigFigs(value * M_PER_S_PER_UM_PER_MS);
  return `${sigFigs(value)} µm/ms (${si} m/s)`;
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

/** One exported row: both units, so the CSV needs no unit conversion to read. */
function speedRow(series: string, point: ComparePoint) {
  return {
    series,
    t_ms: point.x,
    v_um_per_ms: point.y,
    v_m_per_s: point.y * M_PER_S_PER_UM_PER_MS,
  };
}

/**
 * The measured pairs, split by whether the interval touched a held-out frame.
 *
 * Amber is this app's holdout tone, and the series' own name carries the same
 * fact -- so the distinction survives for a reader who cannot use colour.
 */
function measuredSeries(nose: NoseSpeed): CompareSeries[] {
  const points = toSeries(nose.measured.t_ms, nose.measured.v_um_per_ms);
  const { heldout } = nose.measured;
  const pick = (want: boolean) => points.filter((_, i) => heldout[i] === want);
  return [
    {
      id: "measured",
      points: pick(false),
      markers: true,
      slot: SERIES_SLOT.measured,
    },
    {
      id: "measured · held out",
      points: pick(true),
      markers: true,
      slot: SERIES_SLOT.heldout,
    },
  ];
}

function NoseSpeedChart({ nose, name }: { nose: NoseSpeed; name: string }) {
  const predicted = toSeries(nose.t_ms, nose.v_um_per_ms);
  const measured = measuredSeries(nose);
  const rows = [
    ...predicted.map((point) => speedRow("pinn", point)),
    ...measured.flatMap((s) => s.points.map((point) => speedRow(s.id, point))),
  ];
  return (
    <ChartCard
      title="Nose speed"
      unit="µm/ms"
      name={name}
      rows={rows}
      render={() => (
        <CompareChart
          series={[
            { id: "PINN", points: predicted, slot: SERIES_SLOT.primary },
            ...measured,
          ]}
          xLabel="t (ms)"
          yLabel="nose speed · µm/ms"
          ariaLabel={
            "Nose speed over time: the continuous reconstruction as a line, " +
            "and one circle per consecutive camera-frame pair. Pairs spanning " +
            "a held-out frame are drawn separately."
          }
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
      <p className="figcap">
        <b>Figure 1.</b> The reconstruction's nose speed is continuous; each
        measured circle is one finite difference between consecutive camera
        frames, plotted at the midpoint of the interval it measures. An interval
        spanning a held-out frame is drawn apart and labelled — nothing trained
        on it, so it is the model's rate where it was never shown the answer.
      </p>
    </Panel>
  );
}
