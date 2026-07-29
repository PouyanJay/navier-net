import { Button, Chip } from "../../components";
import { SelectField } from "../../components/Field";
import type { RunDetail, RunStatus, RunSummary } from "../../lib/api";
import { formatRunDate, runConditions, toYamlish } from "./format";

const STATUS_CHIP: Record<
  RunStatus,
  { tone: "green" | "accent" | "red" | "default"; label: string }
> = {
  trained: { tone: "green", label: "trained" },
  running: { tone: "accent", label: "running" },
  failed: { tone: "red", label: "failed" },
  empty: { tone: "default", label: "no checkpoint" },
};

interface RunHeaderProps {
  run: RunSummary;
  detail: RunDetail | null;
  /** Dataset id → display label (series labels are editable). */
  datasetLabels: Map<string, string>;
  /** The condition per-condition panels are viewing (joint runs). */
  viewDataset: string | null;
  onViewDataset: (dataset: string) => void;
  onResume: () => void;
  resuming: boolean;
}

/** One field of the run's pedigree line ("seed 1234"). */
function Meta({ label, value }: { label: string; value: string | number }) {
  return (
    <span>
      {label} <b>{value}</b>
    </span>
  );
}

/**
 * The persistent context above the output tabs: which run this is, what it
 * trained on, the exact config that produced it, and how to continue it.
 */
export function RunHeader({
  run,
  detail,
  datasetLabels,
  viewDataset,
  onViewDataset,
  onResume,
  resuming,
}: RunHeaderProps) {
  const status = STATUS_CHIP[run.status];
  const { all, heldout } = runConditions(run);
  const labelOf = (id: string) => datasetLabels.get(id) ?? id;
  const training = (detail?.config as Record<string, unknown> | null)?.[
    "training"
  ] as Record<string, unknown> | undefined;
  const valFraction =
    typeof training?.val_fraction === "number" ? training.val_fraction : null;
  const valSplit =
    valFraction != null && valFraction > 0
      ? `${Math.round(valFraction * 100)} % · ${String(training?.val_strategy ?? "tail")}`
      : "holdout frame only";
  const steps = detail?.steps ?? run.steps;

  return (
    <div className="card run-header" data-testid="run-header">
      <div className="run-header-top">
        <span className="run-header-id">{run.id}</span>
        <Chip tone={status.tone}>{status.label}</Chip>
        <span className="run-header-date">{formatRunDate(run.date)}</span>
        <span className="run-header-spacer" />
        {all.length > 1 && viewDataset && (
          <SelectField
            label="viewing condition"
            value={viewDataset}
            onChange={onViewDataset}
            options={all.map((id) => ({ value: id, label: labelOf(id) }))}
          />
        )}
        {run.status === "trained" && (
          <Button onClick={onResume} disabled={resuming}>
            {resuming ? "Resuming…" : "Resume training"}
          </Button>
        )}
      </div>

      <div className="run-chips">
        {all.map((id) => (
          <Chip key={id} tone={heldout.has(id) ? "amber" : "default"}>
            {labelOf(id)}
            {heldout.has(id) ? " · held out" : ""}
          </Chip>
        ))}
      </div>

      {detail && (
        <div className="run-meta">
          {steps != null && (
            <Meta label="steps" value={steps.toLocaleString()} />
          )}
          {typeof training?.seed === "number" && (
            <Meta label="seed" value={training.seed} />
          )}
          <Meta label="val split" value={valSplit} />
          {typeof training?.device === "string" && (
            <Meta label="device" value={training.device} />
          )}
        </div>
      )}

      {detail?.config && (
        <details className="cfgsnap">
          <summary>
            Config snapshot — .hydra/config.yaml{" "}
            <span className="cfgsnap-hint">reproduces this run exactly</span>
          </summary>
          <pre>{toYamlish(detail.config)}</pre>
        </details>
      )}
    </div>
  );
}
