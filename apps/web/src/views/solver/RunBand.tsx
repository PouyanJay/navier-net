import { Band, SelectField, TextField } from "../../components";
import { SeriesIcon } from "../../components/icons";
import type { DatasetSummary, RunSummary } from "../../lib/api";
import { seriesName } from "../../lib/series";
import { VAL_FRACTION_OPTIONS, type SolverFormState } from "./form";
import { SwitchRow } from "./Rows";

/** What a launch targets. The three are mutually exclusive — a resume takes its
 * config from the checkpoint, and a sweep varies the seed on one series — so
 * they are one choice, not three switches that disable each other. */
export type LaunchMode = "new" | "resume" | "sweep";

export const LAUNCH_MODES: {
  name: LaunchMode;
  label: string;
  note: string;
}[] = [
  {
    name: "new",
    label: "New run",
    note: "Trains a fresh model on the selected series",
  },
  {
    name: "resume",
    label: "Resume",
    note: "Continues a trained run from its checkpoint",
  },
  {
    name: "sweep",
    label: "Seed sweep",
    note: "One child run per seed, same configuration",
  },
];

const modeNote = (mode: LaunchMode) =>
  LAUNCH_MODES.find((option) => option.name === mode)?.note ?? "";

interface RunBandProps {
  form: SolverFormState;
  onForm: (patch: Partial<SolverFormState>) => void;
  mode: LaunchMode;
  onMode: (mode: LaunchMode) => void;
  available: DatasetSummary[];
  selected: string[];
  onToggle: (id: string) => void;
  onSelectAll: (on: boolean) => void;
  heldout: string;
  onToggleHeldout: (id: string) => void;
  resumableRuns: RunSummary[];
  resumeRunId: string;
  onResumeRunId: (id: string) => void;
  seedsText: string;
  onSeedsText: (text: string) => void;
  seedsValid: boolean;
  locked: boolean;
}

/** The run's target: which series it trains on, or which checkpoint it
 * continues, plus what it does with the frames and what it leaves behind. */
export function RunBand({
  form,
  onForm,
  mode,
  onMode,
  available,
  selected,
  onToggle,
  onSelectAll,
  heldout,
  onToggleHeldout,
  resumableRuns,
  resumeRunId,
  onResumeRunId,
  seedsText,
  onSeedsText,
  seedsValid,
  locked,
}: RunBandProps) {
  const resumeTarget = resumableRuns.find((run) => run.id === resumeRunId);
  const primary = selected[0] ?? "";
  const hint =
    mode === "resume"
      ? resumeRunId || "no checkpoint"
      : `${selected.length} of ${available.length} series`;

  return (
    <Band icon={SeriesIcon} label="Run target" hint={hint} collapsible>
      <div className="seg modes" role="radiogroup" aria-label="Launch mode">
        {LAUNCH_MODES.map((option) => {
          const on = option.name === mode;
          const unavailable =
            option.name === "resume" && resumableRuns.length === 0;
          return (
            <button
              key={option.name}
              type="button"
              role="radio"
              aria-checked={on}
              // aria-disabled, not disabled: "no trained run to resume" is the
              // only place that reason is given, and a disabled button would
              // drop out of the tab order taking it along.
              aria-disabled={unavailable || locked || undefined}
              className={on ? "segb on" : "segb"}
              onClick={() => {
                if (unavailable || locked) return;
                onMode(option.name);
              }}
            >
              {option.label}
            </button>
          );
        })}
      </div>
      <p className="rail-note">
        {resumableRuns.length === 0 && mode !== "resume"
          ? `${modeNote(mode)}. No trained run to resume yet.`
          : `${modeNote(mode)}.`}
      </p>

      {mode === "resume" ? (
        <div className="rail-stack">
          <SelectField
            label="Run to resume"
            hint={
              resumeTarget?.steps != null
                ? `ckpt.pt · step ${resumeTarget.steps}`
                : "ckpt.pt"
            }
            value={resumeRunId}
            onChange={onResumeRunId}
            options={resumableRuns.map((run) => ({
              value: run.id,
              label:
                run.steps != null ? `${run.id} · step ${run.steps}` : run.id,
            }))}
            disabled={locked}
          />
          <p className="rail-note">
            Only <b>Steps</b> applies on resume. Everything below is fixed by
            the configuration the run was started with.
          </p>
        </div>
      ) : (
        <SeriesList
          available={available}
          selected={selected}
          heldout={heldout}
          onToggle={onToggle}
          onSelectAll={onSelectAll}
          onToggleHeldout={onToggleHeldout}
          disabled={locked}
          // A sweep varies the seed on ONE series, so hold-out has no meaning
          // there and the list says which series the children will train.
          jointAllowed={mode === "new"}
        />
      )}

      {mode === "sweep" && (
        <div className="rail-stack">
          <TextField
            label="Seeds"
            hint="1-6 unique integers"
            value={seedsText}
            onChange={onSeedsText}
            placeholder="0, 1, 2"
            invalid={!seedsValid}
            disabled={locked}
          />
          <p className="rail-note">
            One child run per seed, all on{" "}
            <b className="mono">{primary || "the first selected series"}</b>,
            with this configuration held fixed.
          </p>
        </div>
      )}

      {mode !== "resume" && (
        <div className="rail-stack">
          <SelectField
            label="Validation split"
            hint="tail frames, never supervised"
            value={String(form.val_fraction)}
            onChange={(value) => onForm({ val_fraction: Number(value) })}
            options={VAL_FRACTION_OPTIONS}
            disabled={locked}
          />
        </div>
      )}

      <SwitchRow
        label="Render deliverables"
        hint="figures and growth.mp4, after evaluation"
        checked={form.render}
        onChange={(render) => onForm({ render })}
        disabled={locked || mode === "sweep"}
        reason={
          mode === "sweep"
            ? "A sweep renders nothing; compare its children on Results."
            : undefined
        }
      />
    </Band>
  );
}

interface SeriesListProps {
  available: DatasetSummary[];
  selected: string[];
  heldout: string;
  onToggle: (id: string) => void;
  onSelectAll: (on: boolean) => void;
  onToggleHeldout: (id: string) => void;
  disabled: boolean;
  jointAllowed: boolean;
}

/**
 * The project's processed series as one control: a checkbox picks whether a
 * series is in the run at all, and — once at least two are in — a per-row
 * toggle flips it train ↔ held-out. A held-out series is loaded but never
 * supervised, and is scored on every frame as the transfer test.
 */
function SeriesList({
  available,
  selected,
  heldout,
  onToggle,
  onSelectAll,
  onToggleHeldout,
  disabled,
  jointAllowed,
}: SeriesListProps) {
  if (available.length === 0) {
    return (
      <p className="rail-note">
        No preprocessed series in this project yet. Upload and preprocess one
        under Datasets &amp; conditions.
      </p>
    );
  }
  const allSelected = selected.length === available.length;
  const joint = jointAllowed && selected.length > 1;
  const trainingCount = selected.length - (heldout ? 1 : 0);

  return (
    <fieldset className="ds-select" disabled={disabled}>
      <div className="ds-select-hd">
        <legend className="rail-label">
          Series
          {joint && (
            <span className="ds-joint-tag mono">joint · {selected.length}</span>
          )}
        </legend>
        <button
          type="button"
          className="btn ghost sm"
          onClick={() => onSelectAll(!allSelected)}
          disabled={disabled}
        >
          {allSelected ? "Clear all" : "Select all"}
        </button>
      </div>
      <ul className="ds-list">
        {available.map((entry) => {
          const isSelected = selected.includes(entry.id);
          const isHeld = heldout === entry.id;
          return (
            <li key={entry.id} className={isHeld ? "ds-row held" : "ds-row"}>
              <label className="ds-item">
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => onToggle(entry.id)}
                  disabled={disabled}
                />
                <span className="mono">{seriesName(entry)}</span>
              </label>
              {joint && isSelected && (
                <button
                  type="button"
                  className={isHeld ? "ds-holdout on" : "ds-holdout"}
                  onClick={() => onToggleHeldout(entry.id)}
                  disabled={disabled}
                  aria-pressed={isHeld}
                  aria-label={`${isHeld ? "Return to training" : "Hold out"} ${seriesName(entry)}`}
                >
                  {isHeld ? "held out" : "hold out"}
                </button>
              )}
            </li>
          );
        })}
      </ul>
      {joint && (
        <p className="rail-note">
          {heldout ? (
            <>
              {trainingCount} train · 1 held out. The held-out series is never
              supervised, and is scored on every frame as a transfer test: can
              the model predict a condition it never trained on?
            </>
          ) : (
            <>
              Trains one model jointly across the selected series. Hold one out
              to keep a whole condition out of training as a transfer test.
            </>
          )}
        </p>
      )}
    </fieldset>
  );
}
