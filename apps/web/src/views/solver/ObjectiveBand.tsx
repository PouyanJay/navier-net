import type { ReactNode } from "react";

import { Band } from "../../components";
import { ObjectiveIcon } from "../../components/icons";
import type { LossWeightsInput } from "../../lib/api";
import { FORM_BOUNDS, type SolverFormState } from "./form";
import { treatmentOf } from "./formulation";
import { SubLabel, SwitchRow, ValueRow } from "./Rows";

interface ObjectiveBandProps {
  form: SolverFormState;
  onForm: (patch: Partial<SolverFormState>) => void;
  locked: boolean;
}

/** The five always-on loss terms, in the order the objective reads. */
const WEIGHT_TERMS: {
  term: keyof LossWeightsInput;
  label: ReactNode;
  hint: string;
}[] = [
  {
    term: "data",
    label: (
      <>
        w<sub>data</sub>
      </>
    ),
    hint: "the segmented frames",
  },
  {
    term: "vof",
    label: (
      <>
        w<sub>VOF</sub>
      </>
    ),
    hint: "interface advection",
  },
  {
    term: "div",
    label: (
      <>
        w<sub>div</sub>
      </>
    ),
    hint: "continuity with the phase-change source",
  },
  {
    term: "src",
    label: (
      <>
        w<sub>src</sub>
      </>
    ),
    hint: "the source held to the interface",
  },
  {
    term: "bc",
    label: (
      <>
        w<sub>BC</sub>
      </>
    ),
    hint: "inlet plug flow, no-slip walls",
  },
];

/**
 * What the run is scored on: the weight each term starts at, and the optional
 * terms that add to the sum.
 *
 * The distinction that earns this its own band: everything here changes the
 * LOSS. The interface treatment above changes the model; the optimizer below
 * changes how the loss is walked.
 */
export function ObjectiveBand({ form, onForm, locked }: ObjectiveBandProps) {
  const rba = form.weighting === "rba";
  const explicitFront = treatmentOf(form) !== "diffuse";
  const extras = (form.kinematics ? 1 : 0) + (form.front_velocity ? 1 : 0);

  return (
    <Band
      icon={ObjectiveIcon}
      label="Objective"
      hint={extras > 0 ? `5 terms + ${extras} extra` : "5 terms"}
      collapsible
    >
      <SubLabel note="initial · rebalanced live">Loss weights</SubLabel>
      {WEIGHT_TERMS.map(({ term, label, hint }) => (
        <ValueRow
          key={term}
          label={label}
          hint={hint}
          value={form.weights[term]}
          onChange={(value) =>
            onForm({ weights: { ...form.weights, [term]: value } })
          }
          min={FORM_BOUNDS.weight.min}
          max={FORM_BOUNDS.weight.max}
          step={0.1}
          disabled={locked}
        />
      ))}
      <ValueRow
        label="Rebalance"
        hint={
          rba
            ? "RBA weights every point every step"
            : "how often the gradient-norm balancer re-weights"
        }
        value={form.rebalance_every}
        onChange={(rebalance_every) => onForm({ rebalance_every })}
        min={FORM_BOUNDS.rebalance_every.min}
        max={FORM_BOUNDS.rebalance_every.max}
        step={100}
        suffix="steps"
        disabled={locked || rba}
      />

      <SubLabel note="opt-in · default off">Extra supervision</SubLabel>
      <SwitchRow
        label="Growth constraints"
        hint="hold the late-window volume budget to a monotone, source-balanced rate"
        checked={form.kinematics}
        onChange={(kinematics) => onForm({ kinematics })}
        disabled={locked}
      >
        <ValueRow
          label="Growth margin"
          hint="× the supervised-tail rate"
          value={form.kin_margin_frac}
          onChange={(kin_margin_frac) => onForm({ kin_margin_frac })}
          min={FORM_BOUNDS.kin_margin_frac.min}
          max={FORM_BOUNDS.kin_margin_frac.max}
          step={0.05}
          disabled={locked}
        />
        <ValueRow
          label="Monotone weight"
          value={form.kin_weight_mono}
          onChange={(kin_weight_mono) => onForm({ kin_weight_mono })}
          min={FORM_BOUNDS.kin_weight.min}
          max={FORM_BOUNDS.kin_weight.max}
          step={0.1}
          disabled={locked}
        />
        <ValueRow
          label="Balance weight"
          value={form.kin_weight_balance}
          onChange={(kin_weight_balance) => onForm({ kin_weight_balance })}
          min={FORM_BOUNDS.kin_weight.min}
          max={FORM_BOUNDS.kin_weight.max}
          step={0.1}
          disabled={locked}
        />
        <ValueRow
          label="Evap-floor weight"
          hint="0 = off · benched unstable when driven"
          value={form.kin_weight_evap}
          onChange={(kin_weight_evap) => onForm({ kin_weight_evap })}
          min={FORM_BOUNDS.kin_weight.min}
          max={FORM_BOUNDS.kin_weight.max}
          step={0.1}
          disabled={locked}
        />
      </SwitchRow>

      <SwitchRow
        label="Measured front velocity"
        hint="supervise the front's rate against the velocity measured from consecutive masks"
        checked={form.front_velocity}
        onChange={(front_velocity) => onForm({ front_velocity })}
        disabled={locked || !explicitFront}
        reason={
          explicitFront
            ? undefined
            : "There is no front to give a normal speed under the diffuse treatment. Choose an explicit front above."
        }
      >
        <ValueRow
          label="Normal-speed weight"
          hint="the binned profile along the front"
          value={form.fv_weight}
          onChange={(fv_weight) => onForm({ fv_weight })}
          min={FORM_BOUNDS.fv_weight.min}
          max={FORM_BOUNDS.fv_weight.max}
          step={0.1}
          disabled={locked}
        />
        <ValueRow
          label="Apex weight"
          hint="the nose's measured 2-D displacement"
          value={form.fv_apex_weight}
          onChange={(fv_apex_weight) => onForm({ fv_apex_weight })}
          min={FORM_BOUNDS.fv_weight.min}
          max={FORM_BOUNDS.fv_weight.max}
          step={0.1}
          disabled={locked}
        />
      </SwitchRow>
    </Band>
  );
}
