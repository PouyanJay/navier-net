import { Panel, ViewCanvas } from "../../components";
import {
  CompareChart,
  type CompareSeries,
} from "../../components/charts/CompareChart";
import { api, type LossRecord } from "../../lib/api";
import { useApiResource } from "./useApiResource";

/** Loss terms in display order; only the ones the run recorded chart. */
const LOSS_TERMS: { key: keyof LossRecord; label: string }[] = [
  { key: "data", label: "data · α supervision" },
  { key: "vof", label: "vof · transport" },
  { key: "div", label: "div · continuity" },
  { key: "src", label: "src · off-interface" },
  { key: "bc", label: "bc · inlet/walls" },
];

interface TrainingTabProps {
  runId: string;
}

/** The optimisation's own story: every loss term over the run, log-scaled. */
export function TrainingTab({ runId }: TrainingTabProps) {
  const historyQ = useApiResource<LossRecord[]>(
    runId,
    (id) => api.getLossHistory(id),
    { nullOn404: true },
  );
  const history = historyQ.data;

  const terms = LOSS_TERMS.filter(({ key }) =>
    history?.some((record) => typeof record[key] === "number"),
  );
  const series: CompareSeries[] = terms.map(({ key }) => ({
    id: String(key),
    points: (history ?? [])
      .filter((record) => typeof record[key] === "number")
      .map((record) => ({ x: record.step, y: record[key] as number })),
  }));
  const last = history?.[history.length - 1];

  return (
    <Panel
      title="Training diagnostics"
      subtitle="loss per term · log₁₀ · from ckpt.pt history"
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
          <ViewCanvas>
            <CompareChart
              series={series}
              logY
              xLabel="step"
              yLabel="residual · log scale"
              ariaLabel="Training loss per term over optimisation steps, log scale."
              yFormat={(v) => v.toExponential(1)}
            />
          </ViewCanvas>
          <div className="legend">
            {terms.map(({ key, label }, i) => (
              <span className="li" key={String(key)}>
                <i className={`sw-line series-${i}`} /> {label}
              </span>
            ))}
          </div>
          {last && (
            <p className="train-final">
              final ·{" "}
              {terms
                .map(({ key }) => {
                  const value = last[key];
                  return `${String(key)} ${typeof value === "number" ? value.toExponential(1) : "—"}`;
                })
                .join(" · ")}
              {/* CLI-era checkpoints record no lr; never crash on it. */}
              {typeof last.lr === "number" &&
                ` · lr ${last.lr.toExponential(1)}`}
            </p>
          )}
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
