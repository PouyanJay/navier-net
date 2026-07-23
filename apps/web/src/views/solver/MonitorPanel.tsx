import { Meter, Panel, Stat } from "../../components";
import type { LossRecord, RunJobStatus } from "../../lib/api";

const formatLoss = (value: number | undefined) =>
  value === undefined ? "—" : value.toExponential(2);

interface MonitorPanelProps {
  status: RunJobStatus | null;
  latest: LossRecord | null;
  holdoutIou: number | null;
}

/** Live run stats — step progress, the headline loss terms, holdout IoU. */
export function MonitorPanel({ status, latest, holdoutIou }: MonitorPanelProps) {
  const total = status?.steps_total ?? 0;
  // Status events only arrive on stage changes; between them the freshest step
  // count is the latest streamed loss record's.
  const done = Math.max(status?.steps_done ?? 0, latest?.step ?? 0);
  return (
    <Panel title="Run monitor" subtitle="live from the solver">
      <div className="statrow">
        <Stat label="Step" value={done} unit={total > 0 ? `/ ${total}` : undefined} />
        <Stat label="Data loss · α" value={formatLoss(latest?.data)} />
        <Stat label="PDE residual · VOF" value={formatLoss(latest?.vof)} />
        <Stat
          label="Holdout IoU"
          value={holdoutIou != null ? holdoutIou.toFixed(3) : "—"}
          tone={holdoutIou != null ? "green" : "default"}
          hint={holdoutIou != null ? "never supervised" : "known after evaluation"}
        />
      </div>
      <Meter value={done} max={total} label="Training progress" />
    </Panel>
  );
}
