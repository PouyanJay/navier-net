/** The recipe layer: the benched configurations the platform itself would run.
 *
 * The rail below offers ~20 switches, and nearly every one has been benched --
 * but the verdicts live in journey notes, not the UI, so the first question a
 * user actually has ("what should I run?") had no control. A recipe is that
 * control: one card per benched configuration, stamping the whole form.
 *
 * A recipe is a BASELINE, not a mode: editing the form afterwards keeps the
 * recipe selected and the launch dock reports each departure as a chip. That
 * reading -- "recommended, plus nucleation root" -- is the reproducibility
 * statement a run's provenance needs, where a vague "custom" says nothing.
 */

import { fieldSpecs, optionLabel } from "../../lib/fields";
import trustedRecipe from "../../lib/trustedRecipe.json";
import type { SolverFormState } from "./form";
import { FORM_DEFAULTS, parseOverrides } from "./form";
import { treatmentMeta, treatmentOf, treatmentPatch } from "./formulation";

export type RecipeName = "full-physics" | "trusted" | "best-heldout" | "legacy";

export interface RecipeMeta {
  name: RecipeName;
  label: string;
  /** The measured evidence for the card, not an adjective. */
  evidence: string;
  /** What choosing it stamps, naming the ladder rung it sets. */
  detail: string;
}

export const RECIPES: RecipeMeta[] = [
  {
    name: "full-physics",
    label: "Full physics",
    evidence: "the platform default",
    detail:
      "Sets the sharp-front treatment: capsule front, Young-Laplace + kinematic condition, film pressure on.",
  },
  {
    name: "trusted",
    label: "Trusted fields",
    evidence: "audited chain",
    detail:
      "Sets the sharp treatment with physical walls, Brinkman no-slip, an outlet, the two-phase pressure, the sharp thermal model with film evaporation under the bubble, the mass balance, the measured front velocity and the outline guard: the chain the hidden fields are audited under (configs/model/trusted.yaml).",
  },
  {
    name: "best-heldout",
    label: "Best held-out IoU",
    evidence: "f11 0.768",
    detail:
      "Sets the explicit-front treatment + causal weighting, zero interface physics -- the measured generalisation winner.",
  },
  {
    name: "legacy",
    label: "Legacy baseline",
    evidence: "pre-R3",
    detail:
      "Sets the diffuse treatment: free level set, bulk residuals only -- for head-to-head runs against the old line.",
  },
];

export const recipeMeta = (name: RecipeName): RecipeMeta =>
  RECIPES.find((recipe) => recipe.name === name) ?? RECIPES[0];

/**
 * The form state a recipe stamps.
 *
 * "Full physics" IS `FORM_DEFAULTS`: the platform default and the recommended
 * recipe are the same configuration by design (asserted in tests), so the rail
 * opens with that card selected and zero departures.
 */
export function recipeState(name: RecipeName): SolverFormState {
  if (name === "trusted") return trustedState();
  if (name === "best-heldout") {
    return {
      ...FORM_DEFAULTS,
      ...treatmentPatch("front"),
      causal_weighting: true,
      causal_mode: "weight",
    };
  }
  if (name === "legacy") {
    return { ...FORM_DEFAULTS, ...treatmentPatch("diffuse") };
  }
  return { ...FORM_DEFAULTS };
}

/**
 * The trusted-fields recipe: `configs/model/trusted.yaml` and
 * `configs/training/trusted.yaml` exactly, as the launch form expresses them.
 * The stamp is a committed export of what those YAMLs compose to
 * (`lib/trustedRecipe.json`, written by `scripts/export_field_table.py` and
 * held current by the Python tests), so the card cannot drift from the CLI
 * recipe the benchmarks run. The physics choices switch on together because
 * each package of the chain was gated on the one before it.
 */
const TRUSTED_STAMP = trustedRecipe as Partial<SolverFormState>;

/** The gate the trusted recipe sets on the causal schedule, and the time bins
 * the thermal source needs: what the rail restores when either is switched
 * back on by hand. */
export const TRUSTED_CAUSAL_GATE = TRUSTED_STAMP.causal_gate_min_weight ?? 0.99;
export const TRUSTED_TIME_BINS = TRUSTED_STAMP.collocation_time_bins || 8;
/** The cadence the trusted recipe gives the outline guard: what the rail
 * restores when the guard is switched back on by hand. */
export const TRUSTED_GUARD_EVERY = TRUSTED_STAMP.outline_guard_every || 500;

const trustedState = (): SolverFormState => ({
  ...FORM_DEFAULTS,
  ...TRUSTED_STAMP,
});

/** A boolean whose departure reads as "+ label" / "− label". Exported so the
 * tests can hold it complete against `MODIFIERS`: a switch missing here would
 * silently drop out of the dock's reproducibility statement. */
export const SWITCH_LABELS: [keyof SolverFormState, string][] = [
  ["hard_pin", "hard root pin"],
  ["film_pressure", "film pressure"],
  ["liquid_film", "liquid film"],
  ["cap_freedom", "free end caps"],
  ["nucleation_root", "nucleation root"],
  ["nose_shape", "nose shape"],
  ["root_attachment", "attachment pressure"],
  ["receding_cap", "receding cap"],
  ["evolving_width", "evolving width"],
  ["allow_pinch", "pinch-off"],
  ["depletable_superheat", "depletable superheat"],
  ["evap_closure_two_way", "two-way evaporation"],
  ["causal_weighting", "causal weighting"],
  ["adaptive_collocation", "adaptive collocation"],
  ["kinematics", "growth constraints"],
  ["front_velocity", "front velocity"],
  ["render", "render"],
  ["mirror_symmetry", "mirror symmetry"],
  ["mass_balance", "mass balance"],
  ["log_chunk_residuals", "chunk residuals"],
  ["footprint_evaporation", "film evaporation"],
  ["freeze_geometry", "outline freeze"],
];

/** The physics choices that are a pick among options; a departure reads as
 * "label → option". Read off the field table; a new choice needs its form
 * field, which `fieldTable.test.ts` holds for every form-owned entry. */
const CHOICE_SPECS = fieldSpecs({ section: "physics", owner: "form" }).filter(
  (spec) => spec.kind === "enum",
);

/** A number whose departure reads as "label base → value". Sub-parameters are
 * gated on the switch that owns them: a changed kinematics weight is noise
 * while kinematics itself is off, because it rides the request inert. */
const VALUE_LABELS: {
  key: keyof SolverFormState;
  label: string;
  gate?: (form: SolverFormState) => boolean;
}[] = [
  { key: "steps", label: "steps" },
  { key: "lr", label: "lr" },
  { key: "lr_halflife", label: "lr halflife" },
  { key: "n_data", label: "n_data" },
  { key: "n_coll", label: "n_coll" },
  { key: "n_bc", label: "n_bc" },
  { key: "rebalance_every", label: "rebalance" },
  { key: "log_every", label: "log every" },
  { key: "val_fraction", label: "val split" },
  { key: "seed", label: "seed" },
  { key: "pin_d_ref", label: "pin scale", gate: (form) => form.hard_pin },
  { key: "cap_delta", label: "cap delta", gate: (form) => form.cap_freedom },
  {
    key: "alpha_eps_final",
    label: "final αε",
    gate: (form) => form.alpha_eps_anneal_steps > 0,
  },
  {
    key: "kin_margin_frac",
    label: "growth margin",
    gate: (form) => form.kinematics,
  },
  {
    key: "kin_weight_mono",
    label: "w_mono",
    gate: (form) => form.kinematics,
  },
  {
    key: "kin_weight_balance",
    label: "w_balance",
    gate: (form) => form.kinematics,
  },
  {
    key: "kin_weight_evap",
    label: "w_evap",
    gate: (form) => form.kinematics,
  },
  { key: "fv_weight", label: "w_fv", gate: (form) => form.front_velocity },
  {
    key: "fv_apex_weight",
    label: "w_fv_apex",
    gate: (form) => form.front_velocity,
  },
  {
    key: "mass_balance_weight",
    label: "w_mass",
    gate: (form) => form.mass_balance,
  },
  { key: "collocation_time_bins", label: "time bins" },
  {
    key: "causal_gate_min_weight",
    label: "causal gate",
    gate: (form) =>
      form.causal_weighting && form.causal_gate_min_weight != null,
  },
  { key: "rebalance_max_ratio", label: "weight cap" },
  { key: "grad_clip", label: "grad clip" },
  { key: "outline_guard_every", label: "outline guard" },
  {
    key: "outline_guard_drop",
    label: "guard drop",
    gate: (form) => form.outline_guard_every > 0,
  },
];

const WEIGHT_LABELS: {
  key: keyof SolverFormState["weights"];
  label: string;
}[] = [
  { key: "data", label: "w_data" },
  { key: "vof", label: "w_vof" },
  { key: "div", label: "w_div" },
  { key: "src", label: "w_src" },
  { key: "bc", label: "w_bc" },
];

/**
 * Every departure of `form` from `recipe`'s stamped state, as dock chips.
 *
 * A treatment change is ONE chip, not a fan of flag chips: the baseline is
 * re-normalised through `treatmentPatch` for the form's rung first, so the
 * flags a rung switches by construction (film pressure off under the explicit
 * front, say) are not re-reported as if the user chose them.
 */
export function recipeDiff(
  form: SolverFormState,
  recipe: RecipeName,
): string[] {
  const stamped = recipeState(recipe);
  const treatment = treatmentOf(form);
  const changed = treatment !== treatmentOf(stamped);
  const base = changed ? { ...stamped, ...treatmentPatch(treatment) } : stamped;
  return [
    ...(changed
      ? [`treatment → ${treatmentMeta(treatment).label.toLowerCase()}`]
      : []),
    ...switchChips(form, base),
    ...choiceChips(form, base),
    ...balancingChips(form, base),
    ...valueChips(form, base),
    ...overrideChips(form),
  ];
}

/** The physics choices, each as one "label → option" chip. */
function choiceChips(form: SolverFormState, base: SolverFormState): string[] {
  const chips: string[] = [];
  for (const spec of CHOICE_SPECS) {
    const key = spec.name as keyof SolverFormState;
    if (form[key] !== base[key]) {
      chips.push(
        `${spec.label.toLowerCase()} → ${optionLabel(String(form[key]))}`,
      );
    }
  }
  return chips;
}

/** Toggled booleans, each as one signed chip. */
function switchChips(form: SolverFormState, base: SolverFormState): string[] {
  const chips: string[] = [];
  for (const [key, label] of SWITCH_LABELS) {
    if (form[key] !== base[key]) {
      chips.push(`${form[key] ? "+" : "−"} ${label}`);
    }
  }
  const sharpening = form.alpha_eps_anneal_steps > 0;
  if (sharpening !== base.alpha_eps_anneal_steps > 0) {
    chips.push(`${sharpening ? "+" : "−"} sharpening`);
  }
  // The causal gate is a switch with a threshold: set or not set.
  const gated = form.causal_gate_min_weight != null;
  if (
    form.causal_weighting &&
    gated !== (base.causal_gate_min_weight != null)
  ) {
    chips.push(`${gated ? "+" : "−"} causal gate`);
  }
  return chips;
}

/** The loss-balancing enums: the scheme, and the causal variant while it is on. */
function balancingChips(
  form: SolverFormState,
  base: SolverFormState,
): string[] {
  const chips: string[] = [];
  if (form.weighting !== base.weighting) {
    chips.push(
      `weighting → ${form.weighting === "rba" ? "RBA" : "gradient-norm"}`,
    );
  }
  if (form.causal_weighting && form.causal_mode !== base.causal_mode) {
    chips.push(
      `causal → ${form.causal_mode === "march" ? "time-marching" : "soft weighting"}`,
    );
  }
  return chips;
}

/** Changed numbers (switch-gated where owned) and the loss-weight table. */
function valueChips(form: SolverFormState, base: SolverFormState): string[] {
  const chips: string[] = [];
  for (const { key, label, gate } of VALUE_LABELS) {
    if (gate && !gate(form)) continue;
    if (form[key] !== base[key]) {
      const shown =
        key === "val_fraction"
          ? `${Math.round((form[key] as number) * 100)}%`
          : String(form[key]);
      chips.push(`${label} ${shown}`);
    }
  }
  for (const { key, label } of WEIGHT_LABELS) {
    if (form.weights[key] !== base.weights[key]) {
      chips.push(`${label} ${form.weights[key]}`);
    }
  }
  return chips;
}

/** Advanced overrides are departures by definition: no recipe stamps any.
 * Unparseable text adds no chips; the dock is already blocked on it. */
function overrideChips(form: SolverFormState): string[] {
  return (parseOverrides(form.overridesText) ?? []).map(
    (token) => `+ ${token}`,
  );
}
