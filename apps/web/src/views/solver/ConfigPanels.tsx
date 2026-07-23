import type { ReactNode } from "react";

import { NumberField, Panel, SelectField, Switch, type SelectOption } from "../../components";
import type { LossWeightsInput, RunSummary } from "../../lib/api";
import { FORM_BOUNDS, HOLDOUT_OPTIONS, type SolverFormState } from "./form";

interface RunConfigPanelProps {
  form: SolverFormState;
  onForm: (patch: Partial<SolverFormState>) => void;
  datasetOptions: SelectOption[];
  dataset: string;
  onDataset: (id: string) => void;
  resume: boolean;
  onResume: (on: boolean) => void;
  resumableRuns: RunSummary[];
  resumeRunId: string;
  onResumeRunId: (id: string) => void;
  /** A run is in flight: the whole form is read-only. */
  locked: boolean;
}

/**
 * "Run configuration" — every field is an input to the run, mapped 1:1 onto
 * `cfg.training`. When resuming, only Steps applies (the rest is fixed by the
 * original run's config), so everything else locks.
 */
export function RunConfigPanel({
  form,
  onForm,
  datasetOptions,
  dataset,
  onDataset,
  resume,
  onResume,
  resumableRuns,
  resumeRunId,
  onResumeRunId,
  locked,
}: RunConfigPanelProps) {
  const fixedByResume = locked || resume; // fields the original run's config owns
  const resumeTarget = resumableRuns.find((run) => run.id === resumeRunId);
  const resumeHint = resumeTarget?.steps != null ? `ckpt.pt · step ${resumeTarget.steps}` : "ckpt.pt";

  return (
    <Panel title="Run configuration" subtitle="inputs to this run">
      <div className="cfg">
        <SelectField
          label="Dataset"
          value={dataset}
          onChange={onDataset}
          options={datasetOptions}
          disabled={fixedByResume}
        />
        <NumberField
          label="Steps"
          value={form.steps}
          onChange={(steps) => onForm({ steps })}
          min={FORM_BOUNDS.steps.min}
          max={FORM_BOUNDS.steps.max}
          step={100}
          disabled={locked}
        />
        <NumberField
          label="Learning rate"
          value={form.lr}
          onChange={(lr) => onForm({ lr })}
          min={FORM_BOUNDS.lr.min}
          max={FORM_BOUNDS.lr.max}
          step={0.0005}
          disabled={fixedByResume}
        />
        <NumberField
          label="LR schedule"
          hint="halve every"
          value={form.lr_halflife}
          onChange={(lr_halflife) => onForm({ lr_halflife })}
          min={FORM_BOUNDS.lr_halflife.min}
          max={FORM_BOUNDS.lr_halflife.max}
          step={100}
          suffix="steps"
          disabled={fixedByResume}
        />
        <NumberField
          label="Data batch"
          value={form.n_data}
          onChange={(n_data) => onForm({ n_data })}
          min={FORM_BOUNDS.n_data.min}
          max={FORM_BOUNDS.n_data.max}
          step={512}
          suffix="pts"
          disabled={fixedByResume}
        />
        <NumberField
          label="Collocation batch"
          value={form.n_coll}
          onChange={(n_coll) => onForm({ n_coll })}
          min={FORM_BOUNDS.n_coll.min}
          max={FORM_BOUNDS.n_coll.max}
          step={512}
          suffix="pts"
          disabled={fixedByResume}
        />
        <NumberField
          label="Boundary batch"
          value={form.n_bc}
          onChange={(n_bc) => onForm({ n_bc })}
          min={FORM_BOUNDS.n_bc.min}
          max={FORM_BOUNDS.n_bc.max}
          step={128}
          suffix="pts"
          disabled={fixedByResume}
        />
        <SelectField
          label="Holdout frame"
          hint="generalization metric"
          value={String(form.holdout_frame)}
          onChange={(value) => onForm({ holdout_frame: Number(value) })}
          options={HOLDOUT_OPTIONS}
          disabled={fixedByResume}
        />
        <NumberField
          label="Log every"
          value={form.log_every}
          onChange={(log_every) => onForm({ log_every })}
          min={FORM_BOUNDS.log_every.min}
          max={FORM_BOUNDS.log_every.max}
          step={10}
          suffix="steps"
          disabled={fixedByResume}
        />
      </div>
      <div className="switch-rows">
        <Switch
          label="Resume from checkpoint"
          hint={resume ? resumeHint : undefined}
          checked={resume}
          onChange={onResume}
          disabled={locked || resumableRuns.length === 0}
        />
        {resume && (
          <SelectField
            label="Run to resume"
            value={resumeRunId}
            onChange={onResumeRunId}
            options={resumableRuns.map((run) => ({
              value: run.id,
              label: run.steps != null ? `${run.id} · step ${run.steps}` : run.id,
            }))}
            disabled={locked}
          />
        )}
        <Switch
          label="Render deliverables"
          hint="figures + growth.mp4 after evaluation"
          checked={form.render}
          onChange={(render) => onForm({ render })}
          disabled={locked}
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

/** "Loss weights" — the initial per-term weights, rebalanced live by the trainer. */
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
