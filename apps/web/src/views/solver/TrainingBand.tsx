import { Band, SelectField } from "../../components";
import { TuningIcon } from "../../components/icons";
import {
  CAUSAL_MODE_OPTIONS,
  FORM_BOUNDS,
  WEIGHTING_OPTIONS,
  type SolverFormState,
} from "./form";
import { SubLabel, SwitchRow, ValueRow } from "./Rows";

interface TrainingBandProps {
  form: SolverFormState;
  onForm: (patch: Partial<SolverFormState>) => void;
  /** Steps still applies on resume; the rest is fixed by the original config. */
  locked: boolean;
  fixed: boolean;
}

/** The batch sizes, which are one idea in three fields. */
const BATCHES = [
  {
    key: "n_data",
    label: "Data points",
    hint: "sampled from the segmented frames",
    bounds: "n_data",
  },
  {
    key: "n_coll",
    label: "Collocation points",
    hint: "where the residuals are evaluated",
    bounds: "n_coll",
  },
  {
    key: "n_bc",
    label: "Boundary points",
    hint: "inlet and walls",
    bounds: "n_bc",
  },
] as const;

/** The valid range, in the same words the API's own bounds use. A config field
 * is an input to a real run, so it states what it will accept. */
const range = (key: keyof typeof FORM_BOUNDS) =>
  `${FORM_BOUNDS[key].min} – ${FORM_BOUNDS[key].max}`;

/**
 * How the objective is walked: for how long, at what rate, on how many points,
 * and which scheme decides what the optimizer attends to.
 *
 * Nothing here changes what the run means — only how well and how quickly it
 * gets there — which is why it sits below the objective rather than beside it.
 */
export function TrainingBand({
  form,
  onForm,
  locked,
  fixed,
}: TrainingBandProps) {
  const rba = form.weighting === "rba";

  return (
    <Band
      icon={TuningIcon}
      label="Training"
      hint={`${form.steps} steps · lr ${form.lr}`}
      collapsible
    >
      <ValueRow
        label="Steps"
        hint={range("steps")}
        value={form.steps}
        onChange={(steps) => onForm({ steps })}
        min={FORM_BOUNDS.steps.min}
        max={FORM_BOUNDS.steps.max}
        step={100}
        disabled={locked}
      />
      <ValueRow
        label="Learning rate"
        hint={`Adam, on every field network · ${range("lr")}`}
        value={form.lr}
        onChange={(lr) => onForm({ lr })}
        min={FORM_BOUNDS.lr.min}
        max={FORM_BOUNDS.lr.max}
        step={0.0005}
        disabled={fixed}
      />
      <ValueRow
        label="LR halved every"
        hint={`the decay schedule · ${range("lr_halflife")}`}
        value={form.lr_halflife}
        onChange={(lr_halflife) => onForm({ lr_halflife })}
        min={FORM_BOUNDS.lr_halflife.min}
        max={FORM_BOUNDS.lr_halflife.max}
        step={100}
        suffix="steps"
        disabled={fixed}
      />

      <SubLabel note="per step">Batch sizes</SubLabel>
      {BATCHES.map((batch) => (
        <ValueRow
          key={batch.key}
          label={batch.label}
          hint={`${batch.hint} · ${range(batch.bounds)}`}
          value={form[batch.key]}
          onChange={(value) => onForm({ [batch.key]: value })}
          min={FORM_BOUNDS[batch.bounds].min}
          max={FORM_BOUNDS[batch.bounds].max}
          step={batch.key === "n_bc" ? 128 : 512}
          suffix="pts"
          disabled={fixed}
        />
      ))}
      <ValueRow
        label="Log every"
        hint={`the live chart's resolution · ${range("log_every")}`}
        value={form.log_every}
        onChange={(log_every) => onForm({ log_every })}
        min={FORM_BOUNDS.log_every.min}
        max={FORM_BOUNDS.log_every.max}
        step={10}
        suffix="steps"
        disabled={fixed}
      />

      <SubLabel note="opt-in beyond the default">Loss balancing</SubLabel>
      <SelectField
        label="Weighting"
        hint="how the terms are kept comparable"
        value={form.weighting}
        onChange={(value) =>
          onForm(
            // Valid by construction, in both directions: RBA replaces the
            // gradient-norm rebalancer and the trainer rejects it with causal
            // weighting, while adaptive collocation refreshes the RBA pool and
            // cannot exist without it.
            value === "rba"
              ? { weighting: "rba", causal_weighting: false }
              : { weighting: "gradnorm", adaptive_collocation: false },
          )
        }
        options={WEIGHTING_OPTIONS}
        disabled={fixed}
      />
      <SwitchRow
        label="Causal weighting"
        hint="learn the early times first, and only then let the late ones count"
        checked={form.causal_weighting}
        onChange={(causal_weighting) => onForm({ causal_weighting })}
        disabled={fixed || rba}
        reason={
          rba
            ? "Not yet compatible with RBA; switch the weighting to gradient-norm."
            : undefined
        }
      >
        <SelectField
          label="Causal mode"
          value={form.causal_mode}
          onChange={(value) =>
            onForm({ causal_mode: value as SolverFormState["causal_mode"] })
          }
          options={CAUSAL_MODE_OPTIONS}
          disabled={fixed}
        />
      </SwitchRow>
      <SwitchRow
        label="Adaptive collocation"
        hint="refresh the point pool toward the highest residual"
        checked={form.adaptive_collocation}
        onChange={(adaptive_collocation) => onForm({ adaptive_collocation })}
        disabled={fixed || !rba}
        reason={
          rba
            ? undefined
            : "Refreshes the RBA pool, so it requires RBA weighting."
        }
      />
    </Band>
  );
}
