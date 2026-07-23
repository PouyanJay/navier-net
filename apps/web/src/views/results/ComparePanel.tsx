import { useState } from "react";

import { Panel, SelectField, Table, ViewCanvas, type Column } from "../../components";
import { CompareChart, type CompareSeries } from "../../components/charts/CompareChart";
import type { RunSummary } from "../../lib/api";
import { MAX_COMPARED, useComparison, type ComparedRun } from "./useComparison";

const LOSS_TERMS = [
  { value: "data", label: "data · α supervision" },
  { value: "vof", label: "vof · interface transport" },
  { value: "div", label: "div · continuity" },
  { value: "src", label: "src · dilatation penalty" },
  { value: "bc", label: "bc · boundary" },
] as const;

type LossTerm = (typeof LOSS_TERMS)[number]["value"];

interface MetricsRow {
  id: string;
  steps: string;
  iouMean: string;
  iouHoldout: string;
  nose: string;
}

const METRICS_COLUMNS: Column<MetricsRow>[] = [
  { header: "Run", cell: (row) => row.id },
  { header: "Steps", cell: (row) => row.steps, num: true },
  { header: "IoU mean", cell: (row) => row.iouMean, num: true },
  { header: "IoU holdout", cell: (row) => row.iouHoldout, num: true },
  { header: "Nose speed (mm/s)", cell: (row) => row.nose, num: true },
];

const fmt = (value: number | null | undefined, digits = 3) =>
  value == null ? "—" : value.toFixed(digits);

function lossSeries(runs: ComparedRun[], term: LossTerm): CompareSeries[] {
  return runs.map((run) => ({
    id: run.detail.id,
    points: run.history.map((record) => ({ x: record.step, y: record[term] })),
  }));
}

function iouSeries(runs: ComparedRun[]): CompareSeries[] {
  return runs.map((run) => ({
    id: run.detail.id,
    points: Object.entries(run.detail.metrics?.iou_per_frame ?? {}).map(([frame, iou]) => ({
      x: Number(frame),
      y: iou,
    })),
  }));
}

function metricsRows(runs: ComparedRun[]): MetricsRow[] {
  return runs.map((run) => ({
    id: run.detail.id,
    steps: run.detail.steps != null ? String(run.detail.steps) : "—",
    iouMean: fmt(run.detail.metrics?.iou_mean),
    iouHoldout: fmt(run.detail.metrics?.iou_holdout),
    nose: fmt(run.detail.metrics?.nose_speed_mm_s, 0),
  }));
}

/** Side-by-side run comparison: pick up to four runs, read them together. */
export function ComparePanel({ candidates }: { candidates: RunSummary[] }) {
  const comparison = useComparison(candidates);
  const [term, setTerm] = useState<LossTerm>("data");
  const { selected, toggle, runs, error } = comparison;

  return (
    <Panel
      title="Compare runs"
      subtitle={`up to ${MAX_COMPARED} · sweep children group by id`}
    >
      <div className="compare-picker" role="group" aria-label="Runs to compare">
        {candidates.map((run) => {
          const index = selected.indexOf(run.id);
          return (
            <button
              key={run.id}
              type="button"
              className="compare-choice"
              aria-pressed={index >= 0}
              data-series={index >= 0 ? index % 4 : undefined}
              onClick={() => toggle(run.id)}
            >
              {run.id}
            </button>
          );
        })}
      </div>
      {error && (
        <p className="state-note error" role="alert">
          Could not load comparison data: {error}
        </p>
      )}
      {selected.length < 2 && (
        <p className="state-note">Select at least two runs to compare them.</p>
      )}
      {runs === null && (
        <p className="state-note" role="status">
          Loading comparison…
        </p>
      )}
      {runs !== null && selected.length >= 2 && (
        <div className="compare-grids">
          <div>
            <div className="compare-chart-head">
              <h3 className="compare-h">Loss history</h3>
              <SelectField
                label="Loss term"
                value={term}
                onChange={(value) => setTerm(value as LossTerm)}
                options={LOSS_TERMS.map((option) => ({ ...option }))}
              />
            </div>
            <ViewCanvas>
              <CompareChart
                series={lossSeries(runs, term)}
                logY
                xLabel="step"
                ariaLabel="Loss history per run, log scale, overlaid for comparison."
                yFormat={(v) => v.toExponential(2)}
              />
            </ViewCanvas>
          </div>
          <div>
            <h3 className="compare-h">IoU per frame</h3>
            <ViewCanvas>
              <CompareChart
                series={iouSeries(runs)}
                xLabel="frame"
                ariaLabel="Intersection-over-union per frame, per run."
              />
            </ViewCanvas>
          </div>
          <Table columns={METRICS_COLUMNS} rows={metricsRows(runs)} rowKey={(row) => row.id} />
        </div>
      )}
    </Panel>
  );
}
