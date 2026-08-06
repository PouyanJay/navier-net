import { useState } from "react";

import { Button, Chip } from "../../components";
import { SelectField } from "../../components/Field";
import { errorMessage } from "../../lib/errors";
import type { RunDetail, RunStatus, RunSummary } from "../../lib/api";
import {
  formatRunDate,
  runConditions,
  runDisplayName,
  toYamlish,
} from "./format";

/** Matches the API's `MAX_LABEL_LEN`, so the field stops before the 422 does. */
const MAX_LABEL_LEN = 80;

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
  /** The frames the evaluator held out, from metrics.json. */
  validationFrames: number[] | null;
  /** Where this run stands in the rail's current ranking, if it is ranked. */
  standing: string | null;
  /** Dataset id → display label (series labels are editable). */
  datasetLabels: Map<string, string>;
  /** The condition per-condition panels are viewing (joint runs). */
  viewDataset: string | null;
  onViewDataset: (dataset: string) => void;
  onResume: () => void;
  resuming: boolean;
  /** Save the run's display name; a blank string clears it back to the id. */
  onRename: (label: string) => Promise<void>;
  /** Open the delete-confirmation for this run. */
  onDelete: () => void;
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
  validationFrames,
  standing,
  datasetLabels,
  viewDataset,
  onViewDataset,
  onResume,
  resuming,
  onRename,
  onDelete,
}: RunHeaderProps) {
  const status = STATUS_CHIP[run.status];
  const displayName = runDisplayName(run, datasetLabels);
  // The draft starts from the stored label, not from the displayed name: with no
  // label the field is empty and the id shows as the placeholder, so saving an
  // untouched field is the no-op it looks like rather than pinning the id as a
  // label. Keyed on the run id below, so switching runs resets the editor.
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState(run.label ?? "");
  const [savingName, setSavingName] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);

  function openRename() {
    setDraft(run.label ?? "");
    setRenameError(null);
    setRenaming(true);
  }

  async function submitRename(event: React.FormEvent) {
    event.preventDefault();
    const cleaned = draft.trim().replace(/\s+/g, " ");
    if (cleaned === (run.label ?? "")) {
      setRenaming(false); // nothing changed
      return;
    }
    setSavingName(true);
    setRenameError(null);
    try {
      await onRename(cleaned);
      setRenaming(false);
    } catch (err) {
      setRenameError(`Could not rename the run: ${errorMessage(err)}`);
    } finally {
      setSavingName(false);
    }
  }
  const { all, heldout } = runConditions(run);
  const labelOf = (id: string) => datasetLabels.get(id) ?? id;
  const training = detail?.config?.["training"] as
    Record<string, unknown> | undefined;
  // What the run was ACTUALLY scored on, from the frames the evaluator wrote —
  // not from a config fraction, and never from the "holdout frame only" fallback
  // this used to print, which contradicted the scorecard directly below it on
  // every run since the single-frame holdout was retired.
  const frames = validationFrames ?? [];
  const valSplit = frames.length
    ? `frames ${frames.join(", ")} · never supervised`
    : "none · every frame supervised";
  const steps = detail?.steps ?? run.steps;
  const seed =
    run.seed ?? (typeof training?.seed === "number" ? training.seed : null);

  return (
    <div className="card run-header" data-testid="run-header">
      <div className="run-header-top">
        {renaming ? (
          <form className="run-rename" onSubmit={submitRename}>
            <input
              className="run-rename-input"
              type="text"
              value={draft}
              autoFocus
              maxLength={MAX_LABEL_LEN}
              placeholder={run.id}
              aria-label="Run name"
              aria-invalid={renameError ? true : undefined}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Escape") setRenaming(false);
              }}
            />
            <Button
              type="submit"
              size="sm"
              variant="primary"
              disabled={savingName}
            >
              {savingName ? "Saving…" : "Save"}
            </Button>
            <Button
              type="button"
              size="sm"
              disabled={savingName}
              onClick={() => setRenaming(false)}
            >
              Cancel
            </Button>
            {/* Blank is a real, reachable state, so say what it does. */}
            <span className="run-rename-hint">
              blank restores <code>{run.id}</code>
            </span>
          </form>
        ) : (
          <>
            <span className="run-header-id">{displayName}</span>
            <Button
              size="sm"
              onClick={openRename}
              title="Rename this run; its id and output directory never change"
            >
              Rename
            </Button>
          </>
        )}
        <Chip tone={status.tone}>{status.label}</Chip>
        {/* The series a run trained on is part of its identity, so it reads on
            the same line rather than opening a row of its own. */}
        {all.map((id) => (
          <Chip key={id} tone={heldout.has(id) ? "amber" : "default"}>
            {labelOf(id)}
            {heldout.has(id) ? " · held out" : ""}
          </Chip>
        ))}
        {standing && <span className="run-header-rank">{standing}</span>}
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
        <Button
          variant="danger"
          onClick={onDelete}
          disabled={run.status === "running"}
          title={
            run.status === "running"
              ? "Stop the run before deleting it"
              : "Delete this run and all its outputs"
          }
        >
          Delete run
        </Button>
      </div>

      {renameError && (
        <p className="run-rename-error" role="alert">
          {renameError}
        </p>
      )}

      <div className="run-meta">
        {/* Once a name replaces it, the id is only visible if the pedigree says
            it: it is the output directory, the resume target and every artifact
            URL, so a renamed run must not hide where it lives. */}
        {displayName !== run.id && <Meta label="id" value={run.id} />}
        {steps != null && <Meta label="steps" value={steps.toLocaleString()} />}
        <Meta label="validated on" value={valSplit} />
        <Meta label="seed" value={seed ?? "not recorded"} />
        {/* The recipe the run RAN, read from its own config: the ids are typed
            by hand and several in this repository disagree with it. */}
        <Meta
          label="recipe"
          value={
            run.recipe == null
              ? "config not recorded"
              : run.recipe.length === 0
                ? "recommended"
                : run.recipe.join(" · ")
          }
        />
        {typeof training?.device === "string" && (
          <Meta label="device" value={training.device} />
        )}
      </div>

      {detail?.config ? (
        <details className="cfgsnap">
          <summary>
            Config snapshot · .hydra/config.yaml{" "}
            <span className="cfgsnap-hint">reproduces this run exactly</span>
          </summary>
          <pre>{toYamlish(detail.config)}</pre>
        </details>
      ) : (
        <p className="cfgsnap-missing">
          No config snapshot: this run cannot be reproduced exactly, and its
          network cannot be rebuilt to replay.
        </p>
      )}
    </div>
  );
}
