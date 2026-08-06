import { Panel, ViewCanvas } from "../../components";
import { ChartFrame } from "../../components/ChartFrame";
import {
  CompareChart,
  type CompareSeries,
} from "../../components/charts/CompareChart";
import { ResidualBars } from "../../components/charts/ResidualBars";
import { api, type LossRecord } from "../../lib/api";
import { useApiResource } from "./useApiResource";

/**
 * The loss terms, in families, with a display label for each.
 *
 * Derived from the record rather than declared: the trainer writes whichever
 * terms the run's own physics turned on, and a hardcoded list of five meant the
 * whole sharp-interface family never appeared — including, on the runs in this
 * repository, the largest residual in the run by three decades.
 */
const TERM_LABELS: Record<string, { family: Family; label: string }> = {
  data: { family: "core", label: "data · α supervision" },
  vof: { family: "core", label: "vof · transport" },
  div: { family: "core", label: "div · continuity" },
  src: { family: "core", label: "src · off-interface" },
  bc: { family: "core", label: "bc · inlet/walls" },
  darcy: { family: "interface", label: "darcy · depth-averaged momentum" },
  laplace: { family: "interface", label: "laplace · Young–Laplace jump" },
  kinematic: { family: "interface", label: "kinematic · front condition" },
  mom: { family: "interface", label: "mom · 2-D momentum" },
  energy: { family: "extra", label: "energy · wall heating" },
  evap: { family: "extra", label: "evap · phase-change closure" },
  fv_normal: { family: "extra", label: "fv_normal · measured front speed" },
  fv_apex: { family: "extra", label: "fv_apex · nose displacement" },
};

type Family = "core" | "interface" | "extra";

/** The five terms the tab charted before it read the record. Marked on the bars
 * so what this chart ADDED reads without a legend claiming it. */
const PREVIOUSLY_CHARTED = new Set(["data", "vof", "div", "src", "bc"]);

/** Keys that are not loss terms. */
const NOT_A_TERM = new Set(["step", "lr"]);

interface TermSeries {
  key: string;
  label: string;
  family: Family;
  points: { x: number; y: number }[];
  final: number;
}

/** Every term this run actually recorded, largest final residual first. */
function termsOf(history: LossRecord[]): TermSeries[] {
  const keys = new Set<string>();
  for (const record of history) {
    for (const key of Object.keys(record)) {
      if (
        !NOT_A_TERM.has(key) &&
        typeof record[key as keyof LossRecord] === "number"
      )
        keys.add(key);
    }
  }
  return [...keys]
    .map((key) => {
      const points = history
        .map((record) => ({
          x: record.step,
          y: record[key as keyof LossRecord] as number,
        }))
        .filter((point) => typeof point.y === "number");
      const meta = TERM_LABELS[key] ?? {
        family: "extra" as Family,
        label: key,
      };
      return {
        key,
        label: meta.label,
        family: meta.family,
        points,
        final: points.length ? points[points.length - 1].y : 0,
      };
    })
    .filter((term) => term.points.length > 0)
    .sort((a, b) => b.final - a.final);
}

/** The cadence the history was logged at, so a four-point line says it is one. */
function cadence(history: LossRecord[]): string {
  if (history.length < 3) return `${history.length} records`;
  const step = history[2].step - history[1].step;
  return `${history.length} records · every ${step.toLocaleString()} steps`;
}

interface TrainingTabProps {
  runId: string;
}

/** The optimisation's own story: every loss term the run recorded, log-scaled,
 * with the one that dominates the objective named. */
export function TrainingTab({ runId }: TrainingTabProps) {
  const historyQ = useApiResource<LossRecord[]>(
    runId,
    (id) => api.getLossHistory(id),
    { nullOn404: true },
  );
  const history = historyQ.data;
  const terms = history ? termsOf(history) : [];
  const series: CompareSeries[] = terms.map((term) => ({
    id: term.key,
    points: term.points,
  }));
  const dominant = terms[0];
  const runnerUp = terms[1];
  const dataTerm = terms.find((term) => term.key === "data");

  return (
    <Panel
      title="Training diagnostics"
      subtitle={
        history && history.length > 0
          ? `${terms.length} terms · ${cadence(history)}`
          : "loss per term · log₁₀ · from ckpt.pt history"
      }
    >
      {historyQ.loading && (
        <p className="state-note" role="status">
          Loading loss history…
        </p>
      )}
      {!historyQ.loading && (!history || history.length === 0) && (
        <p className="state-note">
          No loss history recorded; the run has not trained on this server yet.
        </p>
      )}
      {history && history.length > 0 && (
        <>
          <div className="train-charts">
            <ChartFrame
              name={`${runId}-loss-history`}
              title="Training loss per term"
              rows={history as unknown as Record<string, unknown>[]}
              render={() => (
                <ViewCanvas>
                  <CompareChart
                    series={series}
                    logY
                    xLabel="step"
                    yLabel="residual · log scale"
                    ariaLabel={`Every recorded loss term over optimisation steps, log scale. ${
                      dominant
                        ? `${dominant.key} ends highest at ${dominant.final.toExponential(1)}.`
                        : ""
                    }`}
                    yFormat={(v) => v.toExponential(1)}
                  />
                </ViewCanvas>
              )}
            />
            {/* Where the objective's weight sits at the last step: the curves
              answer "did it converge", this answers "on what". */}
            <ChartFrame
              name={`${runId}-final-residuals`}
              title="Final residual per term"
              rows={terms.map((term) => ({
                term: term.key,
                final: term.final,
              }))}
              render={() => (
                <ViewCanvas>
                  <ResidualBars
                    bars={terms.map((term) => ({
                      key: term.key,
                      value: term.final,
                      family: term.family,
                      known: PREVIOUSLY_CHARTED.has(term.key),
                    }))}
                    ariaLabel={`Final residual per term, log scale. ${
                      dominant ? `${dominant.key} is largest.` : ""
                    }`}
                  />
                </ViewCanvas>
              )}
            />
          </div>
          <div className="legend">
            {terms.map((term, i) => (
              <span className={`li fam-${term.family}`} key={term.key}>
                <i className={`sw-line series-${i % 5}`} /> {term.label}
              </span>
            ))}
          </div>
          {dominant && runnerUp && (
            <p className="train-dominant">
              <b>{dominant.key} dominates the objective</b> at{" "}
              {dominant.final.toExponential(2)}:{" "}
              {(dominant.final / runnerUp.final).toFixed(0)}× {runnerUp.key}
              {dataTerm && dataTerm.key !== dominant.key
                ? ` and ${(dominant.final / dataTerm.final).toFixed(0)}× the data term`
                : ""}
              . This is the residual to re-weight.
            </p>
          )}
          <p className="train-final">
            final ·{" "}
            {terms
              .map((term) => `${term.key} ${term.final.toExponential(1)}`)
              .join(" · ")}
            {typeof history[history.length - 1].lr === "number" &&
              ` · lr ${history[history.length - 1].lr!.toExponential(1)}`}
          </p>
          <p className="figcap">
            <b>Figure 8.</b> Every loss term the optimiser balanced, on the same
            log axis. A term that plateaus early is the one to re-weight; a term
            that grows is the one that diverged.
          </p>
        </>
      )}
    </Panel>
  );
}
