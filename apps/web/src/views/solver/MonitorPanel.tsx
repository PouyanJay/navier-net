import { Meter, Panel, Stat } from "../../components";
import type { LossRecord, RunJobStatus } from "../../lib/api";

const formatLoss = (value: number | undefined) =>
  value === undefined ? "n/a" : value.toExponential(2);

const formatIou = (value: number | null) =>
  value != null ? value.toFixed(3) : "n/a";

interface MonitorPanelProps {
  status: RunJobStatus | null;
  latest: LossRecord | null;
  /** Show the in-distribution validation IoU (a validation split is configured). */
  showVal: boolean;
  /** Show the transfer IoU (a whole series is held out of training). */
  showTransfer: boolean;
  holdoutIou: number | null;
  valIou: number | null;
  transferIou: number | null;
}

/** Live run stats: step progress, the headline loss terms, and generalization.
 * Which generalization stats appear follows the run's configuration: the
 * validation split gives an in-distribution validation IoU; a held-out series
 * gives a transfer IoU. With neither, the legacy holdout-frame IoU (if any).
 * Kept visually distinct — they answer different questions. */
export function MonitorPanel({
  status,
  latest,
  showVal,
  showTransfer,
  holdoutIou,
  valIou,
  transferIou,
}: MonitorPanelProps) {
  const total = status?.steps_total ?? 0;
  // Status events only arrive on stage changes; between them the freshest step
  // count is the latest streamed loss record's.
  const done = Math.max(status?.steps_done ?? 0, latest?.step ?? 0);
  return (
    <Panel title="Run monitor" subtitle="live from the solver">
      <div className="statrow">
        <Stat
          label="Step"
          value={done}
          unit={total > 0 ? `/ ${total}` : undefined}
        />
        <Stat label="Data loss · α" value={formatLoss(latest?.data)} />
        <Stat label="PDE residual · VOF" value={formatLoss(latest?.vof)} />
        {showVal && (
          <Stat
            label="Validation IoU"
            value={formatIou(valIou)}
            tone={valIou != null ? "green" : "default"}
            hint="held-out frames · in-distribution"
          />
        )}
        {showTransfer && (
          <Stat
            label="Transfer IoU"
            value={formatIou(transferIou)}
            tone="amber"
            hint="held-out series · never trained"
          />
        )}
        {!showVal && !showTransfer && (
          <Stat
            label="Holdout IoU"
            value={formatIou(holdoutIou)}
            tone={holdoutIou != null ? "green" : "default"}
            hint={
              holdoutIou != null ? "never supervised" : "known after evaluation"
            }
          />
        )}
      </div>
      <Meter value={done} max={total} label="Training progress" />
    </Panel>
  );
}
