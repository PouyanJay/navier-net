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
import { FORM_BOUNDS, VAL_FRACTION_OPTIONS, type SolverFormState } from "./form";

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
  heldout: string;
  onHeldout: (id: string) => void;
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

/** The project's processed datasets (series) as a checkbox group with "select
 * all". One selected trains as usual; several train one model jointly. Which
 * series to hold out, and the validation split, are separate controls below.
 * Wrapped in a <fieldset> so resume disables the whole group at once. */
function DatasetMultiSelect({
  available,
  selected,
  allSelected,
  onToggle,
  onSelectAll,
  disabled,
}: {
  available: DatasetSummary[];
  selected: string[];
  allSelected: boolean;
  onToggle: (id: string) => void;
  onSelectAll: (on: boolean) => void;
  disabled: boolean;
}) {
  if (available.length === 0) {
    return (
      <div className="ds-select">
        <span className="cfg-label">Series</span>
        <p className="state-note">
          No preprocessed series in this project yet.
        </p>
      </div>
    );
  }
  const joint = selected.length > 1;
  return (
    <fieldset className="ds-select" disabled={disabled}>
      <div className="ds-select-hd">
        <legend className="cfg-label">
          Series
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
        {available.map((d) => (
          <li key={d.id}>
            <label className="ds-item">
              <input
                type="checkbox"
                checked={selected.includes(d.id)}
                onChange={() => onToggle(d.id)}
                disabled={disabled}
              />
              <span className="mono">{d.id}</span>
            </label>
          </li>
        ))}
      </ul>
      {joint && (
        <p className="ds-hint">
          Trains one model jointly across the selected series — transfer learning
          across their operating conditions.
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
  heldout,
  onHeldout,
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

  // Hold-out (axis B) needs at least two series selected, so one can be kept out
  // while the others train. With one series it is unavailable and pinned to None.
  const canHoldOut = selectedDatasets.length > 1;
  const holdoutOptions = [
    { value: "", label: "none · train on every series" },
    ...selectedDatasets.map((id) => ({ value: id, label: `${id} · held out` })),
  ];

  return (
    <Panel title="Run configuration" subtitle="inputs to this run">
      <DatasetMultiSelect
        available={availableDatasets}
        selected={selectedDatasets}
        allSelected={allSelected}
        onToggle={onToggleDataset}
        onSelectAll={onSelectAllDatasets}
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
          label="Hold out"
          hint="a whole series · transfer test"
          value={heldout}
          onChange={onHeldout}
          options={holdoutOptions}
          disabled={fixedByResume || !canHoldOut}
        />
        <SelectField
          label="Validation split"
          hint="frames per trained series"
          value={String(form.val_fraction)}
          onChange={(value) => onForm({ val_fraction: Number(value) })}
          options={VAL_FRACTION_OPTIONS}
          disabled={fixedByResume}
        />
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
