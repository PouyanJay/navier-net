import { Panel, Stat, Table, type Column, type Tone } from "../../components";
import { ChartFrame } from "../../components/ChartFrame";
import {
  CompareChart,
  type CompareSeries,
} from "../../components/charts/CompareChart";
import type { PhysicsDiagnostics, PhysicsFrame } from "../../lib/api";

/** Below this the Young-Laplace jump is being satisfied; above it the pressure
 * field is not tracking the interface's own curvature. The threshold the R4
 * physics gate is stated against. */
const JUMP_TARGET = 0.1;

/** A residual that ends above this fraction of where it started is not being
 * solved -- the R3 baseline's momentum term sat flat at ~4.7 for 1500 steps. */
const CONVERGED_RATIO = 0.5;

function jumpTone(value: number | null): Tone {
  if (value == null) return "default";
  if (value <= JUMP_TARGET) return "green";
  return value <= 3 * JUMP_TARGET ? "amber" : "red";
}

/** How close the model's neck is to the measured one, as a tone. A neck that is
 * absent entirely (the R3 failure) reads red however good the IoU is. */
function neckTone(model: number | null, measured: number | null): Tone {
  if (model == null || measured == null || measured <= 0) return "default";
  const ratio = model / measured;
  if (ratio >= 0.75) return "green";
  return ratio >= 0.25 ? "amber" : "red";
}

function fmt(value: number | null, digits = 3): string {
  return value == null ? "—" : value.toFixed(digits);
}

function pct(value: number | null): string {
  return value == null ? "—" : (value * 100).toFixed(1);
}

interface ConvergenceRow {
  term: string;
  first: number;
  last: number;
  ratio: number | null;
}

const NECK_COLUMNS: Column<PhysicsFrame>[] = [
  { header: "Frame", cell: (row) => row.frame, num: true },
  {
    header: "Depth · model",
    cell: (row) => fmt(row.neck_depth_model),
    num: true,
  },
  {
    header: "Depth · measured",
    cell: (row) => fmt(row.neck_depth_measured),
    num: true,
  },
  {
    header: "Waist · model",
    cell: (row) => fmt(row.neck_location_model, 2),
    num: true,
  },
  {
    header: "Waist · measured",
    cell: (row) => fmt(row.neck_location_measured, 2),
    num: true,
  },
];

const CONVERGENCE_COLUMNS: Column<ConvergenceRow>[] = [
  { header: "Equation", cell: (row) => row.term },
  { header: "First", cell: (row) => row.first.toPrecision(3), num: true },
  { header: "Last", cell: (row) => row.last.toPrecision(3), num: true },
  {
    header: "Ratio",
    cell: (row) => (row.ratio == null ? "—" : row.ratio.toPrecision(3)),
    num: true,
  },
  { header: "Status", cell: (row) => convergenceLabel(row.ratio) },
];

/** A term with no ratio is not "not solved" — it had nothing to form a ratio
 * from (its first window averaged exactly zero), which is a different fact. */
function convergenceLabel(ratio: number | null): string {
  if (ratio == null) return "not measurable";
  return ratio < CONVERGED_RATIO ? "converging" : "not solved";
}

interface InterfacePhysicsPanelProps {
  physics: PhysicsDiagnostics | null;
  /** Whether the run has an explicit front at all — decides which empty state. */
  frontGeometry: boolean;
}

/**
 * Whether the solution obeys the physics, not just the pixels.
 *
 * The R3 baseline scored IoU 0.929 on its trained frames while its Young-Laplace
 * jump was 85 % wrong and its bubble carried no neck at all against a measured
 * 0.474 — an overlap metric cannot see either. These are the numbers that can,
 * and they sit beside IoU rather than behind it.
 */
export function InterfacePhysicsPanel({
  physics,
  frontGeometry,
}: InterfacePhysicsPanelProps) {
  if (!physics) {
    return (
      <Panel title="Interface physics" subtitle="the conditions at the front">
        <p className="state-note">
          {frontGeometry
            ? "No physics diagnostics recorded for this run — re-run the evaluate stage to measure them."
            : "This run was trained without an explicit front (model.front_geometry), so there is no interface to impose the conditions on and nothing to measure. Enable Front geometry in the Solver to score them."}
        </p>
      </Panel>
    );
  }

  const frames = physics.per_frame;
  const neckSeries: CompareSeries[] = [
    {
      id: "measured",
      points: frames.map((f) => ({ x: f.frame, y: f.neck_depth_measured })),
      markers: true,
    },
    {
      id: "model",
      points: frames.map((f) => ({ x: f.frame, y: f.neck_depth_model })),
    },
  ];
  const convergence: ConvergenceRow[] = Object.entries(
    physics.residual_convergence,
  ).map(([term, record]) => ({ term, ...record }));

  return (
    <>
      <Panel
        title="Interface physics"
        subtitle="the conditions at the front · IoU cannot see these"
      >
        <div className="physics-stats">
          <Stat
            label="Young–Laplace · nose"
            value={pct(physics.laplace_error_nose)}
            unit="%"
            tone={jumpTone(physics.laplace_error_nose)}
            hint={`jump error · target below ${JUMP_TARGET * 100} %`}
          />
          <Stat
            label="Young–Laplace · front"
            value={pct(physics.laplace_error_front)}
            unit="%"
            tone={jumpTone(physics.laplace_error_front)}
            hint="against the capillary pressure's variation"
          />
          <Stat
            label="Axial capillary gradient"
            value={fmt(physics.axial_capillary_gradient, 3)}
            hint="the drainage drive · 0 means no neck can form"
          />
          <Stat
            label="Neck depth · model"
            value={fmt(physics.neck_depth_model, 3)}
            tone={neckTone(
              physics.neck_depth_model,
              physics.neck_depth_measured,
            )}
            hint={`measured ${fmt(physics.neck_depth_measured, 3)} at u = ${fmt(
              physics.neck_location_measured,
              2,
            )}`}
          />
        </div>

        {frames.length > 0 && (
          <ChartFrame
            name="neck-depth"
            title="Neck depth per frame — model vs measured masks"
            rows={frames.map((f) => ({
              frame: f.frame,
              neck_depth_model: f.neck_depth_model,
              neck_depth_measured: f.neck_depth_measured,
              neck_location_model: f.neck_location_model,
              neck_location_measured: f.neck_location_measured,
            }))}
            render={() => (
              <CompareChart
                series={neckSeries}
                xLabel="frame"
                yLabel="neck depth (fraction of the shoulder)"
                ariaLabel="Neck depth per frame, model against the measured masks"
                yFormat={(v) => v.toFixed(2)}
              />
            )}
          />
        )}

        <Table<PhysicsFrame>
          caption="Neck per frame: how far the bubble's waist collapses below its shoulder, and where along the bubble it sits."
          columns={NECK_COLUMNS}
          rows={frames}
          rowKey={(row) => String(row.frame)}
        />
      </Panel>

      <Panel
        title="Residual convergence"
        subtitle="did each equation actually get solved"
      >
        {convergence.length === 0 ? (
          <p className="state-note">
            No physics residuals were active in this run — the Stage-B equations
            are off, so there is nothing to converge.
          </p>
        ) : (
          <>
            <p className="note">
              The <em>value</em> of a residual says little on its own. What
              matters is whether it fell over its active window: the ratio below
              is its mean over the last fifth against the first fifth, excluding
              the Stage-B warm-up. At or above 1, that equation is not being
              solved at all.
            </p>
            <Table<ConvergenceRow>
              caption="Each physics residual's mean at the start and end of its active window."
              columns={CONVERGENCE_COLUMNS}
              rows={convergence}
              rowKey={(row) => row.term}
              rowTone={(row) =>
                row.ratio != null && row.ratio < CONVERGED_RATIO
                  ? undefined
                  : "amber"
              }
            />
          </>
        )}
      </Panel>
    </>
  );
}
