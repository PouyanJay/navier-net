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
}

export const FORM_DEFAULTS: SolverFormState = {
  steps: 1500,
  lr: 0.002,
  lr_halflife: 800,
  n_data: 3072,
  n_coll: 3072,
  n_bc: 512,
  holdout_frame: 5,
  // Off by default so a single-dataset run is unchanged; the joint pre-fill
  // (see JOINT_VAL_FRACTION) raises it to 0.2 when more than one dataset trains.
  val_fraction: 0,
  val_strategy: "tail",
  rebalance_every: 500,
  log_every: 200,
  weights: { data: 10, vof: 1, div: 1, src: 0.1, bc: 5 },
  render: true,
};

/** The validation split a joint run pre-fills: a standard 80/20, tail (the honest
 * extrapolation test for a growth series). A single-dataset run stays at 0. */
export const JOINT_VAL_FRACTION = 0.2;

/** Validation-split options: a per-dataset fraction of frames held from training. */
export const VAL_FRACTION_OPTIONS = [
  { value: "0", label: "none · train on every frame" },
  { value: "0.1", label: "10% · in-distribution validation" },
  { value: "0.2", label: "20% · in-distribution validation" },
  { value: "0.3", label: "30% · in-distribution validation" },
];

/** Which frames the split holds out. */
export const VAL_STRATEGY_OPTIONS = [
  { value: "tail", label: "tail · extrapolation" },
  { value: "scatter", label: "scatter · interpolation" },
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
