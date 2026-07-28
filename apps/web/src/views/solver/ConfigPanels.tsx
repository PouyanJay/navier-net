import type { ReactNode } from "react";

import {
  NumberField,
  Panel,
  SelectField,
  Switch,
  TextField,
} from "../../components";
import type {
  DatasetSummary,
  LossWeightsInput,
  RunSummary,
} from "../../lib/api";
import {
  FORM_BOUNDS,
  HOLDOUT_OPTIONS,
  VAL_FRACTION_OPTIONS,
  VAL_STRATEGY_OPTIONS,
  type SolverFormState,
} from "./form";

/** The numeric run-config fields, in display order. Each maps 1:1 onto a
 * `cfg.training` value; bounds come from FORM_BOUNDS by the same key. */
interface NumberSpec {
  key:
    "steps" | "lr" | "lr_halflife" | "n_data" | "n_coll" | "n_bc" | "log_every";
  label: string;
  hint?: string;
  suffix?: string;
  step: number;
  /** Steps still applies on resume; every other field is fixed by the run. */
  editableOnResume?: boolean;
}

const NUMBER_FIELDS: NumberSpec[] = [
  { key: "steps", label: "Steps", step: 100, editableOnResume: true },
  { key: "lr", label: "Learning rate", step: 0.0005 },
  {
    key: "lr_halflife",
    label: "LR schedule",
    hint: "halve every",
    suffix: "steps",
    step: 100,
  },
  { key: "n_data", label: "Data batch", suffix: "pts", step: 512 },
  { key: "n_coll", label: "Collocation batch", suffix: "pts", step: 512 },
  { key: "n_bc", label: "Boundary batch", suffix: "pts", step: 128 },
  { key: "log_every", label: "Log every", suffix: "steps", step: 10 },
];

interface RunConfigPanelProps {
  form: SolverFormState;
  onForm: (patch: Partial<SolverFormState>) => void;
  availableDatasets: DatasetSummary[];
  selectedDatasets: string[];
  onToggleDataset: (id: string) => void;
  onSelectAllDatasets: (on: boolean) => void;
  heldoutDatasets: string[];
  onToggleHeldout: (id: string) => void;
  resume: boolean;
  onResume: (on: boolean) => void;
  resumableRuns: RunSummary[];
  resumeRunId: string;
  onResumeRunId: (id: string) => void;
  sweepMode: boolean;
  onSweepMode: (on: boolean) => void;
  seedsText: string;
  onSeedsText: (text: string) => void;
  seedsValid: boolean;
  /** A run is in flight: the whole form is read-only. */
  locked: boolean;
}

/** The project's processed datasets as a checkbox group with "select all".
 * One selected trains as usual; several train one model jointly. When joint,
 * each selected row can be flipped train ↔ hold-out: a held-out condition (axis
 * B) is loaded but never supervised, scored as a transfer test. Wrapped in a
 * <fieldset> so resume disables the whole group at once. */
function DatasetMultiSelect({
  available,
  selected,
  heldout,
  allSelected,
  onToggle,
  onSelectAll,
  onToggleHeldout,
  disabled,
}: {
  available: DatasetSummary[];
  selected: string[];
  heldout: string[];
  allSelected: boolean;
  onToggle: (id: string) => void;
  onSelectAll: (on: boolean) => void;
  onToggleHeldout: (id: string) => void;
  disabled: boolean;
}) {
  if (available.length === 0) {
    return (
      <div className="ds-select">
        <span className="cfg-label">Datasets</span>
        <p className="state-note">
          No preprocessed dataset in this project yet.
        </p>
      </div>
    );
  }
  const joint = selected.length > 1;
  const trainingCount = selected.length - heldout.length;
  return (
    <fieldset className="ds-select" disabled={disabled}>
      <div className="ds-select-hd">
        <legend className="cfg-label">
          Datasets
          {joint && (
            <span className="ds-joint-tag">joint · {selected.length}</span>
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
        {available.map((d) => {
          const isSelected = selected.includes(d.id);
          const isHeld = heldout.includes(d.id);
          // Don't let the last training dataset be held out — nothing would train.
          const lockTrain = !isHeld && trainingCount <= 1;
          return (
            <li key={d.id} className={isHeld ? "ds-row held" : "ds-row"}>
              <label className="ds-item">
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => onToggle(d.id)}
                  disabled={disabled}
                />
                <span className="mono">{d.id}</span>
              </label>
              {joint && isSelected && (
                <button
                  type="button"
                  className={isHeld ? "ds-holdout on" : "ds-holdout"}
                  onClick={() => onToggleHeldout(d.id)}
                  disabled={disabled || lockTrain}
                  aria-pressed={isHeld}
                  aria-label={`${isHeld ? "Return to training" : "Hold out"} ${d.id}`}
                >
                  {isHeld ? "held out" : "train"}
                </button>
              )}
            </li>
          );
        })}
      </ul>
      {joint && (
        <p className="ds-hint">
          {heldout.length > 0 ? (
            <>
              {trainingCount} train · {heldout.length} held out. Held-out
              conditions are never supervised — scored on every frame as a
              transfer test (can the model predict a condition it never trained
              on?).
            </>
          ) : (
            <>
              Trains one model jointly across the selected datasets — transfer
              learning across their operating conditions. Flip a row to “held
              out” to keep a whole condition out of training as a transfer test.
            </>
          )}
        </p>
      )}
    </fieldset>
  );
}

/**
 * "Run configuration": every field is an input to the run, mapped 1:1 onto
 * `cfg.training`. When resuming, only Steps applies (the rest is fixed by the
 * original run's config), so everything else locks.
 */
export function RunConfigPanel({
  form,
  onForm,
  availableDatasets,
  selectedDatasets,
  onToggleDataset,
  onSelectAllDatasets,
  heldoutDatasets,
  onToggleHeldout,
  resume,
  onResume,
  resumableRuns,
  resumeRunId,
  onResumeRunId,
  sweepMode,
  onSweepMode,
  seedsText,
  onSeedsText,
  seedsValid,
  locked,
}: RunConfigPanelProps) {
  const fixedByResume = locked || resume; // fields the original run's config owns
  const resumeTarget = resumableRuns.find((run) => run.id === resumeRunId);
  const resumeHint =
    resumeTarget?.steps != null
      ? `ckpt.pt · step ${resumeTarget.steps}`
      : "ckpt.pt";

  const allSelected =
    availableDatasets.length > 0 &&
    selectedDatasets.length === availableDatasets.length;

  return (
    <Panel title="Run configuration" subtitle="inputs to this run">
      <DatasetMultiSelect
        available={availableDatasets}
        selected={selectedDatasets}
        heldout={heldoutDatasets}
        allSelected={allSelected}
        onToggle={onToggleDataset}
        onSelectAll={onSelectAllDatasets}
        onToggleHeldout={onToggleHeldout}
        disabled={fixedByResume}
      />
      <div className="cfg">
        {NUMBER_FIELDS.map((spec) => (
          <NumberField
            key={spec.key}
            label={spec.label}
            hint={spec.hint}
            value={form[spec.key]}
            onChange={(value) => onForm({ [spec.key]: value })}
            min={FORM_BOUNDS[spec.key].min}
            max={FORM_BOUNDS[spec.key].max}
            step={spec.step}
            suffix={spec.suffix}
            disabled={spec.editableOnResume ? locked : fixedByResume}
          />
        ))}
        <SelectField
          label="Holdout frame"
          hint="single-frame check"
          value={String(form.holdout_frame)}
          onChange={(value) => onForm({ holdout_frame: Number(value) })}
          options={HOLDOUT_OPTIONS}
          disabled={fixedByResume}
        />
        <SelectField
          label="Validation split"
          hint="per dataset · held from training"
          value={String(form.val_fraction)}
          onChange={(value) => onForm({ val_fraction: Number(value) })}
          options={VAL_FRACTION_OPTIONS}
          disabled={fixedByResume}
        />
        {form.val_fraction > 0 && (
          <SelectField
            label="Split strategy"
            hint="which frames"
            value={form.val_strategy}
            onChange={(value) =>
              onForm({ val_strategy: value as SolverFormState["val_strategy"] })
            }
            options={VAL_STRATEGY_OPTIONS}
            disabled={fixedByResume}
          />
        )}
      </div>
      <div className="switch-rows">
        <Switch
          label="Resume from checkpoint"
          hint={resume ? resumeHint : undefined}
          checked={resume}
          onChange={onResume}
          disabled={locked || sweepMode || resumableRuns.length === 0}
        />
        {resume && (
          <SelectField
            label="Run to resume"
            value={resumeRunId}
            onChange={onResumeRunId}
            options={resumableRuns.map((run) => ({
              value: run.id,
              label:
                run.steps != null ? `${run.id} · step ${run.steps}` : run.id,
            }))}
            disabled={locked}
          />
        )}
        <Switch
          label="Seed sweep"
          hint="same config · one child per seed"
          checked={sweepMode}
          onChange={onSweepMode}
          disabled={locked || resume}
        />
        {sweepMode && (
          <TextField
            label="Seeds"
            hint="1-6 unique integers"
            value={seedsText}
            onChange={onSeedsText}
            placeholder="0, 1, 2"
            invalid={!seedsValid}
            disabled={locked}
          />
        )}
        <Switch
          label="Render deliverables"
          hint="figures + growth.mp4 after evaluation"
          checked={form.render}
          onChange={(render) => onForm({ render })}
          disabled={locked || sweepMode}
        />
      </div>
    </Panel>
  );
}

interface LossWeightsPanelProps {
  weights: LossWeightsInput;
  rebalanceEvery: number;
  onForm: (patch: Partial<SolverFormState>) => void;
  locked: boolean;
}

/** "Loss weights": the initial per-term weights, rebalanced live by the trainer. */
export function LossWeightsPanel({
  weights,
  rebalanceEvery,
  onForm,
  locked,
}: LossWeightsPanelProps) {
  const weightField = (term: keyof LossWeightsInput, label: ReactNode) => (
    <NumberField
      label={label}
      value={weights[term]}
      onChange={(value) => onForm({ weights: { ...weights, [term]: value } })}
      min={FORM_BOUNDS.weight.min}
      max={FORM_BOUNDS.weight.max}
      step={0.1}
      disabled={locked}
    />
  );

  return (
    <Panel title="Loss weights" subtitle="initial · rebalanced live">
      <div className="cfg cfg-narrow">
        {weightField(
          "data",
          <>
            w<sub>data</sub>
          </>,
        )}
        {weightField(
          "vof",
          <>
            w<sub>VOF</sub>
          </>,
        )}
        {weightField(
          "div",
          <>
            w<sub>div</sub>
          </>,
        )}
        {weightField(
          "src",
          <>
            w<sub>src</sub>
          </>,
        )}
        {weightField(
          "bc",
          <>
            w<sub>BC</sub>
          </>,
        )}
        <NumberField
          label="Rebalance"
          hint="every"
          value={rebalanceEvery}
          onChange={(rebalance_every) => onForm({ rebalance_every })}
          min={FORM_BOUNDS.rebalance_every.min}
          max={FORM_BOUNDS.rebalance_every.max}
          step={100}
          suffix="steps"
          disabled={locked}
        />
      </div>
    </Panel>
  );
}
