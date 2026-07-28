import { Meter, Panel, Stat } from "../../components";
import type { LossRecord, RunJobStatus } from "../../lib/api";

const formatLoss = (value: number | undefined) =>
  value === undefined ? "n/a" : value.toExponential(2);

const formatIou = (value: number | null) =>
  value != null ? value.toFixed(3) : "n/a";

interface MonitorPanelProps {
  status: RunJobStatus | null;
  latest: LossRecord | null;
  holdoutIou: number | null;
  valIou: number | null;
  transferIou: number | null;
}

/** Live run stats: step progress, the headline loss terms, and generalization.
 * A single-dataset run reports the holdout-frame IoU; a joint (two-axis) run
 * reports the in-distribution validation IoU and, when a condition is held out,
 * the transfer IoU — kept visually distinct (they answer different questions). */
export function MonitorPanel({
  status,
  latest,
  holdoutIou,
  valIou,
  transferIou,
}: MonitorPanelProps) {
  const total = status?.steps_total ?? 0;
  // Status events only arrive on stage changes; between them the freshest step
  // count is the latest streamed loss record's.
  const done = Math.max(status?.steps_done ?? 0, latest?.step ?? 0);
  const joint = valIou != null || transferIou != null;
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
        {joint ? (
          <>
            <Stat
              label="Validation IoU"
              value={formatIou(valIou)}
              tone={valIou != null ? "green" : "default"}
              hint="held-out frames · in-distribution"
            />
            {transferIou != null && (
              <Stat
                label="Transfer IoU"
                value={formatIou(transferIou)}
                tone="amber"
                hint="held-out condition · never trained"
              />
            )}
          </>
        ) : (
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
