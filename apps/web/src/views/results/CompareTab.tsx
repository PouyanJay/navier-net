import { useEffect, useState } from "react";

import { Panel, ViewCanvas } from "../../components";
import { ChartFrame } from "../../components/ChartFrame";
import {
  CompareChart,
  type CompareSeries,
} from "../../components/charts/CompareChart";
import {
  api,
  type LossRecord,
  type RunMetrics,
  type RunSummary,
} from "../../lib/api";
import { isTrainedRun, MAX_COMPARED } from "../../lib/runs";
import { fmtIou, runDisplayName } from "./format";

interface RunFacts {
  metrics: RunMetrics | null;
  loss: LossRecord[];
}

interface CompareTabProps {
  runs: RunSummary[];
  /** The run open in the page; preselected. */
  currentId: string;
  /** Further runs the rail picked, preselected alongside `currentId`. */
  alsoCompare?: string[];
  datasetLabels: Map<string, string>;
}

interface MetricRow {
  label: string;
  value: (
    facts: RunFacts | undefined,
    run: RunSummary,
  ) => number | string | null;
  /** Higher is better → the best cell reads green. */
  best?: boolean;
  format?: (value: number) => string;
}

const ROWS: MetricRow[] = [
  // What differs first: a comparison of two numbers means nothing until the
  // configurations behind them are on the same screen.
  {
    label: "Recipe",
    value: (_, run) =>
      run.recipe == null
        ? "not recorded"
        : run.recipe.length === 0
          ? "recommended"
          : run.recipe.join(" · "),
  },
  { label: "Seed", value: (_, run) => run.seed ?? "not recorded" },
  {
    label: "Steps",
    value: (_, run) => run.steps,
    format: (v) => v.toLocaleString(),
  },
  {
    label: "Frames evaluated",
    value: (_, run) => run.n_frames ?? null,
    format: (v) => String(v),
  },
  {
    label: "Conditions trained",
    value: (_, run) =>
      (run.datasets?.length ?? 1) - (run.heldout_datasets?.length ?? 0),
    format: (v) => String(v),
  },
  // Then the scores, on the same ladder the run list leads with. The retired
  // holdout-frame row is gone: it could only ever print a dash.
  {
    label: "Validation IoU · A",
    value: (facts, run) =>
      facts?.metrics?.val_iou_mean ??
      facts?.metrics?.iou_val ??
      run.val_iou_mean ??
      run.iou_val ??
      null,
    best: true,
  },
  {
    label: "Transfer IoU · B",
    value: (facts) => facts?.metrics?.transfer?.mean ?? null,
    best: true,
  },
  {
    label: "Mean IoU",
    value: (facts, run) => facts?.metrics?.iou_mean ?? run.iou_mean ?? null,
    best: true,
  },
];

/** What the compared runs have in common, when that makes their gap readable.
 * Two runs that differ in one knob attribute their difference to it; two that
 * differ in several attribute it to nothing. */
function differing(runs: RunSummary[]): string | null {
  if (runs.length !== 2) return null;
  const [a, b] = runs;
  const sameSteps = a.steps === b.steps;
  const sameSeed = a.seed === b.seed;
  const recipeA = (a.recipe ?? []).join("+");
  const recipeB = (b.recipe ?? []).join("+");
  if (a.recipe == null || b.recipe == null) return null;
  if (recipeA === recipeB) {
    return sameSeed
      ? null
      : "Same recipe, different seed: the gap below is this configuration's own spread.";
  }
  return sameSteps && sameSeed
    ? "One knob apart. Everything but the recipe matches, so the gap below is attributable to it."
    : null;
}

/** Side-by-side metrics and overlaid data-loss for up to four trained runs. */
export function CompareTab({
  runs,
  currentId,
  alsoCompare,
  datasetLabels,
}: CompareTabProps) {
  const trained = runs.filter(isTrainedRun);
  const [picked, setPicked] = useState<Set<string>>(() => {
    const first = trained.find((run) => run.id === currentId) ?? trained[0];
    // A comparison assembled in the rail arrives whole; otherwise the tab opens
    // on the run being read plus whichever other one is at hand.
    const rest = (alsoCompare ?? []).filter((id) =>
      trained.some((run) => run.id === id && run.id !== first?.id),
    );
    const fallback = rest.length
      ? []
      : [trained.find((run) => run.id !== first?.id)?.id];
    return new Set(
      [first?.id, ...rest, ...fallback]
        .filter(Boolean)
        .slice(0, MAX_COMPARED) as string[],
    );
  });
  const [facts, setFacts] = useState<Map<string, RunFacts>>(new Map());

  useEffect(() => {
    let mounted = true;
    const missing = [...picked].filter((id) => !facts.has(id));
    if (missing.length === 0) return;
    Promise.all(
      missing.map(async (id) => {
        const [detail, loss] = await Promise.all([
          api.getRun(id).catch(() => null),
          api.getLossHistory(id).catch(() => [] as LossRecord[]),
        ]);
        return [id, { metrics: detail?.metrics ?? null, loss }] as const;
      }),
    ).then((entries) => {
      if (mounted) setFacts((current) => new Map([...current, ...entries]));
    });
    return () => {
      mounted = false;
    };
  }, [picked, facts]);

  if (trained.length < 2)
    return (
      <Panel title="Compare runs" subtitle="up to 4 · same project">
        <div className="res-empty">
          <b>Nothing to compare yet</b>
          Comparison needs at least two trained runs in this project.
        </div>
      </Panel>
    );

  const selected = trained.filter((run) => picked.has(run.id));
  const togglePick = (id: string) =>
    setPicked((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        if (next.size > 1) next.delete(id);
      } else if (next.size < MAX_COMPARED) next.add(id);
      return next;
    });

  const lossSeries: CompareSeries[] = selected.map((run) => ({
    id: run.id,
    points: (facts.get(run.id)?.loss ?? []).map((record) => ({
      x: record.step,
      y: record.data,
    })),
  }));

  return (
    <Panel title="Compare runs" subtitle="up to 4 · same project">
      <div className="cmp-picks" role="group" aria-label="Runs to compare">
        {trained.map((run) => {
          const index = selected.findIndex((s) => s.id === run.id);
          return (
            <button
              key={run.id}
              type="button"
              className="cmp-pick"
              aria-pressed={picked.has(run.id)}
              onClick={() => togglePick(run.id)}
            >
              <span
                className={
                  "cmp-swatch" + (index >= 0 ? ` series-${index}` : "")
                }
                aria-hidden="true"
              />
              {runDisplayName(run, datasetLabels)}
            </button>
          );
        })}
      </div>

      {differing(selected) && (
        <p className="note">
          <b>{differing(selected)}</b>
        </p>
      )}
      <div className="cmp-table-wrap">
        <table className="cmp-table">
          <thead>
            <tr>
              <th scope="col">Metric</th>
              {selected.map((run) => (
                <th scope="col" key={run.id}>
                  {runDisplayName(run, datasetLabels)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => {
              const values = selected.map((run) =>
                row.value(facts.get(run.id), run),
              );
              const numbers = values.filter(
                (v): v is number => typeof v === "number",
              );
              const best =
                row.best && numbers.length > 1 ? Math.max(...numbers) : null;
              return (
                <tr key={row.label}>
                  <th scope="row">{row.label}</th>
                  {values.map((value, i) => (
                    <td
                      key={selected[i].id}
                      className={
                        typeof value === "number" && value === best
                          ? "best"
                          : undefined
                      }
                    >
                      {value == null
                        ? "—"
                        : typeof value === "number"
                          ? (row.format ?? fmtIou)(value)
                          : value}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {lossSeries.some((series) => series.points.length > 0) && (
        <>
          <ChartFrame
            name="compared-runs-data-loss"
            title="Data loss of the compared runs"
            rows={lossSeries.flatMap((series) =>
              series.points.map((point) => ({
                run: series.id,
                step: point.x,
                data_loss: point.y,
              })),
            )}
            render={() => (
              <ViewCanvas>
                <CompareChart
                  series={lossSeries}
                  logY
                  xLabel="step"
                  yLabel="data loss · log scale"
                  ariaLabel="Data loss of the compared runs over optimisation steps."
                  yFormat={(v) => v.toExponential(1)}
                />
              </ViewCanvas>
            )}
          />
          <p className="figcap">
            <b>Figure 9.</b> The supervision (data) loss of each compared run,
            log-scaled; best IoU per metric row reads green in the table.
          </p>
        </>
      )}
    </Panel>
  );
}
