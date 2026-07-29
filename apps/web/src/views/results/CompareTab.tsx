import { useEffect, useState } from "react";

import { Panel, ViewCanvas } from "../../components";
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
import { isTrainedRun } from "../../lib/runs";
import { fmtIou } from "./format";

/** Comparing more runs than chart series colors would un-name the lines. */
const MAX_COMPARED = 4;

interface RunFacts {
  metrics: RunMetrics | null;
  loss: LossRecord[];
}

interface CompareTabProps {
  runs: RunSummary[];
  /** The run open in the page; preselected. */
  currentId: string;
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
  {
    label: "In-distribution IoU · A",
    value: (facts) =>
      facts?.metrics?.val_iou_mean ?? facts?.metrics?.iou_val ?? null,
    best: true,
  },
  {
    label: "Transfer IoU · B",
    value: (facts) => facts?.metrics?.transfer?.mean ?? null,
    best: true,
  },
  {
    label: "Mean IoU",
    value: (facts) => facts?.metrics?.iou_mean ?? null,
    best: true,
  },
  {
    label: "Holdout-frame IoU",
    value: (facts) => facts?.metrics?.iou_holdout ?? null,
    best: true,
  },
  {
    label: "Steps",
    value: (_, run) => run.steps,
    format: (v) => v.toLocaleString(),
  },
  {
    label: "Conditions trained",
    value: (_, run) =>
      (run.datasets?.length ?? 1) - (run.heldout_datasets?.length ?? 0),
    format: (v) => String(v),
  },
];

/** Side-by-side metrics and overlaid data-loss for up to four trained runs. */
export function CompareTab({ runs, currentId }: CompareTabProps) {
  const trained = runs.filter(isTrainedRun);
  const [picked, setPicked] = useState<Set<string>>(() => {
    const first = trained.find((run) => run.id === currentId) ?? trained[0];
    const second = trained.find((run) => run.id !== first?.id);
    return new Set([first?.id, second?.id].filter(Boolean) as string[]);
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
              {run.id}
            </button>
          );
        })}
      </div>

      <div className="cmp-table-wrap">
        <table className="cmp-table">
          <thead>
            <tr>
              <th scope="col">Metric</th>
              {selected.map((run) => (
                <th scope="col" key={run.id}>
                  {run.id}
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
          <ViewCanvas>
            <CompareChart
              series={lossSeries}
              logY
              xLabel="step"
              ariaLabel="Data loss of the compared runs over optimisation steps."
              yFormat={(v) => v.toExponential(1)}
            />
          </ViewCanvas>
          <p className="figcap">
            <b>Figure 9.</b> The supervision (data) loss of each compared run,
            log-scaled; best IoU per metric row reads green in the table.
          </p>
        </>
      )}
    </Panel>
  );
}
