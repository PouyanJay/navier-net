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

import type { SolverFormState } from "./form";
import { FORM_DEFAULTS, parseOverrides } from "./form";
import { treatmentMeta, treatmentOf, treatmentPatch } from "./formulation";

export type RecipeName = "full-physics" | "best-heldout" | "legacy";

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

/** A boolean whose departure reads as "+ label" / "− label". */
const SWITCH_LABELS: [keyof SolverFormState, string][] = [
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
];

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
];

const WEIGHT_LABELS: { key: keyof SolverFormState["weights"]; label: string }[] =
  [
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
  const chips: string[] = [];

  let base = stamped;
  if (treatment !== treatmentOf(stamped)) {
    chips.push(`treatment → ${treatmentMeta(treatment).label.toLowerCase()}`);
    base = { ...stamped, ...treatmentPatch(treatment) };
  }

  for (const [key, label] of SWITCH_LABELS) {
    if (form[key] !== base[key]) {
      chips.push(`${form[key] ? "+" : "−"} ${label}`);
    }
  }
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
  const sharpening = form.alpha_eps_anneal_steps > 0;
  if (sharpening !== base.alpha_eps_anneal_steps > 0) {
    chips.push(`${sharpening ? "+" : "−"} sharpening`);
  }

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
  // Advanced overrides are departures by definition: no recipe stamps any.
  // Unparseable text adds no chips; the dock is already blocked on it.
  for (const token of parseOverrides(form.overridesText) ?? []) {
    chips.push(`+ ${token}`);
  }
  return chips;
}
