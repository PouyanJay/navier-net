import { Callout, Panel } from "../../components";
import {
  CompareChart,
  SERIES_SLOT,
  toComparePoints,
  type ComparePoint,
  type CompareSeries,
} from "../../components/charts/CompareChart";
import type {
  ApexVelocity,
  KinematicsSeries,
  MeasuredSpeed,
} from "../../lib/api";
import { ChartCard } from "./ChartCard";
import { FrontProfilePanel } from "./FrontProfilePanel";
import type { RunCapability } from "./runCapability";
import { StateNote } from "./StateNote";
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

/** One exported row: both units, so the CSV needs no unit conversion to read.
 * A null value stays null in both columns rather than becoming a zero. */
function speedRow(series: string, point: ComparePoint) {
  return {
    series,
    t_ms: point.x,
    v_um_per_ms: point.y,
    v_m_per_s: point.y == null ? null : point.y * M_PER_S_PER_UM_PER_MS,
  };
}

/**
 * The measured pairs, split by whether the interval touched a held-out frame.
 *
 * Amber is this app's holdout tone, and the series' own name carries the same
 * fact -- so the distinction survives for a reader who cannot use colour.
 */
function measuredSeries(measured: MeasuredSpeed): CompareSeries[] {
  const points = toComparePoints(measured.t_ms, measured.v_um_per_ms);
  const pick = (want: boolean) =>
    points.filter((_, i) => measured.heldout[i] === want);
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

interface SpeedChartProps {
  title: string;
  /** File stem for the chart's downloads. */
  name: string;
  /** What the y numbers are, for the screen-reader description. */
  quantity: string;
  model: { t_ms: KinematicsSeries; v_um_per_ms: KinematicsSeries };
  measured: MeasuredSpeed;
}

/**
 * One speed against time: the model's continuous curve, and one circle per
 * consecutive camera-frame pair. The shape the nose and both apex components
 * share.
 */
function SpeedChart({
  title,
  name,
  quantity,
  model,
  measured,
}: SpeedChartProps) {
  const predicted = toComparePoints(model.t_ms, model.v_um_per_ms);
  const camera = measuredSeries(measured);
  const rows = [
    ...predicted.map((point) => speedRow("pinn", point)),
    ...camera.flatMap((s) => s.points.map((point) => speedRow(s.id, point))),
  ];
  return (
    <ChartCard
      title={title}
      unit="µm/ms"
      name={name}
      rows={rows}
      render={() => (
        <CompareChart
          series={[
            { id: "PINN", points: predicted, slot: SERIES_SLOT.primary },
            ...camera,
          ]}
          xLabel="t (ms)"
          yLabel={`${quantity} · µm/ms`}
          ariaLabel={
            `${title}: the continuous reconstruction as a line, and one circle ` +
            "per consecutive camera-frame pair. Pairs spanning a held-out " +
            "frame are drawn separately."
          }
          yFormat={speedReadout}
        />
      )}
    />
  );
}

/** The apex's two components, each against time. Same unit on both, but one
 * axis per quantity -- never a dual axis. */
function ApexCharts({ apex, stem }: { apex: ApexVelocity; stem: string }) {
  const component = (axis: "x" | "y") => ({
    model: {
      t_ms: apex.t_ms,
      v_um_per_ms: axis === "x" ? apex.vx_um_per_ms : apex.vy_um_per_ms,
    },
    measured: {
      t_ms: apex.measured.t_ms,
      v_um_per_ms:
        axis === "x" ? apex.measured.vx_um_per_ms : apex.measured.vy_um_per_ms,
      heldout: apex.measured.heldout,
    },
  });
  return (
    <>
      <SpeedChart
        title="Apex velocity · along x"
        quantity="apex vₓ"
        name={`${stem}-apex-vx`}
        {...component("x")}
      />
      <SpeedChart
        title="Apex velocity · across y"
        quantity="apex v_y"
        name={`${stem}-apex-vy`}
        {...component("y")}
      />
    </>
  );
}

interface FrontVelocityTabProps {
  /** Whether the run asked for the measurement at all, and the inferred nose
   * speed it has regardless. */
  capability: RunCapability;
  noseSpeed: number | null;
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
export function FrontVelocityTab({
  runId,
  dataset,
  capability,
  noseSpeed,
}: FrontVelocityTabProps) {
  const { report, error, loading } = useFrontVelocity(runId, dataset);
  const stem = `${runId}${dataset ? `-${dataset}` : ""}`;

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

  // Two different situations wore one empty state: a run that never measured
  // the front's velocity, and one that did but whose report is missing. The
  // advice differs, so the states do.
  if (!report) {
    const asked = capability.frontVelocityRequested;
    return (
      <Panel
        title="Front velocity"
        subtitle={
          asked ? "measured · report missing" : "not measured by this run"
        }
      >
        {asked ? (
          <StateNote tone="caution" title="The report is missing.">
            This run enabled the measured front velocity, so the evaluate stage
            should have written a normal-speed profile. Re-run evaluate to
            produce it.
          </StateNote>
        ) : (
          <StateNote
            title={
              asked === false
                ? "This run did not measure the front's velocity."
                : "No front-velocity report, and no config to say whether one was asked for."
            }
          >
            {asked === false ? (
              <>
                The <span className="mono">Measured front velocity</span> option
                was off when it was launched, so no normal-speed profile was
                recorded. Enable it in the Solver and re-run to get one.
              </>
            ) : (
              <>
                Re-running the evaluate stage on a run launched from the Solver
                records it.
              </>
            )}
          </StateNote>
        )}
        {noseSpeed != null && (
          <p className="state-note">
            The nose&apos;s own rate is known without it, from the reconstructed
            front: <b className="mono">{noseSpeed.toFixed(1)} mm·s⁻¹</b>.
          </p>
        )}
      </Panel>
    );
  }

  return (
    <>
      <Panel
        title="Front velocity"
        subtitle="how fast the interface moved · the rate, not the shape"
      >
        <div className="kin-grid">
          <SpeedChart
            title="Nose speed"
            quantity="nose speed"
            name={`${stem}-nose-speed`}
            model={report.nose_speed}
            measured={report.nose_speed.measured}
          />
          {report.apex && <ApexCharts apex={report.apex} stem={stem} />}
        </div>
        <p className="figcap">
          <b>Figure 1.</b> The reconstruction's speed is continuous; each
          measured circle is one finite difference between consecutive camera
          frames, plotted at the midpoint of the interval it measures. An
          interval spanning a held-out frame is drawn apart and labelled:
          nothing trained on it, so it is the model's rate where it was never
          shown the answer.
        </p>
      </Panel>

      {report.profile && (
        <FrontProfilePanel
          profile={report.profile}
          stem={stem}
          readout={speedReadout}
        />
      )}

      {!report.apex && (
        <Panel
          title="Along the front"
          subtitle="the apex, the profile and the kymograph"
        >
          <p className="state-note">
            This run was trained without an explicit front
            (model.front_geometry), so there is no parameterised interface to
            read a per-point speed from and no apex to track, only the nose,
            above, which is measured off the predicted mask. Enable Front
            geometry in the Solver to measure the rest.
          </p>
        </Panel>
      )}
    </>
  );
}
