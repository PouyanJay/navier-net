/** The solver form's state and its mapping onto a launch request.
 *
 * Defaults mirror `configs/training/stage_a.yaml`; the form starts at the
 * exact configuration the CLI would run.
 */

import type { LossWeightsInput, RunLaunchRequest } from "../../lib/api";

export interface SolverFormState {
  steps: number;
  lr: number;
  lr_halflife: number;
  n_data: number;
  n_coll: number;
  n_bc: number;
  holdout_frame: number;
  val_fraction: number;
  val_strategy: "tail" | "scatter";
  rebalance_every: number;
  log_every: number;
  weights: LossWeightsInput;
  render: boolean;
  weighting: "gradnorm" | "rba";
  causal_weighting: boolean;
  causal_mode: "weight" | "march";
  adaptive_collocation: boolean;
  hard_pin: boolean;
  pin_d_ref: number;
  kinematics: boolean;
  kin_margin_frac: number;
  kin_weight_mono: number;
  kin_weight_balance: number;
  kin_weight_evap: number;
}

export const FORM_DEFAULTS: SolverFormState = {
  steps: 1500,
  lr: 0.002,
  lr_halflife: 800,
  n_data: 3072,
  n_coll: 3072,
  n_bc: 512,
  // The Solver no longer exposes a single-frame holdout; generalization is the
  // series hold-out and the validation split. -1 disables the legacy frame knob.
  holdout_frame: -1,
  // Off by default (train on every frame); the user opts into a split. Always
  // "tail": hold the last frames of each kept series -- the honest extrapolation
  // test, never a random/interior frame the neighbours already pin down.
  val_fraction: 0,
  val_strategy: "tail",
  rebalance_every: 500,
  log_every: 200,
  weights: { data: 10, vof: 1, div: 1, src: 0.1, bc: 5 },
  render: true,
  // Accuracy techniques: every default is a no-op, so the form starts at today's
  // recipe. See the README "Accuracy techniques" note.
  weighting: "gradnorm",
  causal_weighting: false,
  causal_mode: "weight",
  adaptive_collocation: false,
  // Hard root pin: anchor the interface at the measured nucleation site for
  // all t. The anchor is data-derived; only the gate scale is a knob.
  hard_pin: false,
  pin_d_ref: 0.1,
  // Kinematic growth constraints. The evap-floor weight defaults to 0 (the
  // bench showed it destabilizes the front) -- deliberate opt-in only.
  kinematics: false,
  kin_margin_frac: 0.3,
  kin_weight_mono: 1,
  kin_weight_balance: 1,
  kin_weight_evap: 0,
};

/** Loss-weighting schemes. "gradnorm" is the live rebalancer (needs a hand-picked
 * rebalance cadence); "rba" is bounded residual-based attention (stable, no tuning). */
export const WEIGHTING_OPTIONS = [
  { value: "gradnorm", label: "gradient-norm" },
  { value: "rba", label: "RBA (bounded)" },
];

/** Causal variant: soft per-time weighting, or a time-marching curriculum. */
export const CAUSAL_MODE_OPTIONS = [
  { value: "weight", label: "soft weighting" },
  { value: "march", label: "time-marching" },
];

/** Validation-split options: a fraction of each kept-in series' frames held from
 * training as an in-distribution validation set (deterministic, tail). Labels are
 * kept terse so the select doesn't force its grid column wider than the card. */
export const VAL_FRACTION_OPTIONS = [
  { value: "0", label: "none" },
  { value: "0.1", label: "10%" },
  { value: "0.2", label: "20%" },
  { value: "0.3", label: "30%" },
];

/** Bounds shown on the inputs; the API enforces the same ranges. */
export const FORM_BOUNDS = {
  steps: { min: 1, max: 20000 },
  lr: { min: 0.00001, max: 1 },
  lr_halflife: { min: 1, max: 100000 },
  n_data: { min: 16, max: 16384 },
  n_coll: { min: 16, max: 16384 },
  n_bc: { min: 8, max: 8192 },
  rebalance_every: { min: 10, max: 100000 },
  log_every: { min: 10, max: 5000 },
  weight: { min: 0, max: 10000 },
  pin_d_ref: { min: 0.01, max: 2 },
  kin_margin_frac: { min: 0, max: 2 },
  kin_weight: { min: 0, max: 100 },
} as const;

/** Holdout options in physical frame numbers; values are the 0-based index. */
export const HOLDOUT_OPTIONS = [
  { value: "5", label: "frame 6 · never supervised" },
  { value: "3", label: "frame 4" },
  { value: "7", label: "frame 8" },
  { value: "-1", label: "none (train on all frames)" },
];

export function toLaunchRequest(
  form: SolverFormState,
  target: { datasets: string[]; heldout?: string[] } | { resumeRunId: string },
): RunLaunchRequest {
  const base = { ...form };
  if ("resumeRunId" in target) {
    return { ...base, resume: true, run_id: target.resumeRunId };
  }
  // One dataset trains as today; several train one model jointly (the API reads
  // `datasets` either way). Held-out conditions (axis B) travel only when marked.
  const heldout = target.heldout ?? [];
  return {
    ...base,
    datasets: target.datasets,
    ...(heldout.length > 0 ? { heldout_datasets: heldout } : {}),
  };
}

/** Seeds a sweep may run: 1-6 unique non-negative integers. */
export const SWEEP_SEED_LIMIT = 6;

/** Parse a comma/space-separated seed list; null when invalid. */
export function parseSeeds(text: string): number[] | null {
  const parts = text.split(/[\s,]+/).filter((part) => part !== "");
  if (parts.length === 0 || parts.length > SWEEP_SEED_LIMIT) return null;
  const seeds = parts.map(Number);
  if (seeds.some((seed) => !Number.isInteger(seed) || seed < 0)) return null;
  if (new Set(seeds).size !== seeds.length) return null;
  return seeds;
}
