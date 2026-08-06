/** The Results canvas: what a run can still answer, which checks actually ran,
 * and what a tab holds before you open it. */

import { describe, expect, it } from "vitest";

import type { PhysicsValidation, RunDetail, RunSummary } from "../src/lib/api";
import {
  physicsVerdict,
  unsupervisedIou,
} from "../src/views/results/physicsChecks";
import { runCapability } from "../src/views/results/runCapability";
import { tabBadges } from "../src/views/results/tabBadges";

const RUN: RunSummary = {
  id: "fvb-w1-s0",
  dataset: "Series-1",
  status: "trained",
  steps: 3000,
  iou_holdout: null,
  datasets: ["Series-1"],
  heldout_datasets: [],
  iou_val: 0.8928,
  iou_mean: 0.9287,
  n_frames: 11,
  recipe: null,
  seed: null,
};

const DETAIL = (over: Partial<RunDetail> = {}): RunDetail => ({
  id: RUN.id,
  dataset: "Series-1",
  status: "trained",
  steps: 3000,
  metrics: null,
  config: {
    model: { fields: ["phi", "u", "v", "s", "p"], sharp_interface: true },
    training: { seed: 0, front_velocity: false },
  },
  artifacts: {
    checkpoint: true,
    metrics: true,
    groups: false,
    video: false,
    figures: [],
    front_velocity: false,
    config: true,
  },
  ...over,
});

const VALIDATION = (over: Partial<PhysicsValidation> = {}): PhysicsValidation =>
  ({
    nose_speed_inferred_mm_s: 171.96,
    nose_speed_measured_mm_s: null,
    nose_speed_error_pct: null,
    bretherton_film_um: null,
    hele_shaw: null,
    reynolds: null,
    weber: null,
    capillary: null,
    prandtl: null,
    iou_mean: 0.9287,
    iou_holdout: null,
    holdout_frame: null,
    val_iou_mean: null,
    iou_val: 0.8928,
    validation_frames: [10, 11],
    transfer_iou_mean: null,
    transfer_per_dataset: null,
    transfer_per_frame: null,
    per_dataset: null,
    training_datasets: null,
    heldout_datasets: null,
    physics: {
      laplace_error_nose: 0.1474,
      laplace_error_front: 0.6293,
      axial_capillary_gradient: 0.1129,
      neck_depth_model: 0.3322,
      neck_depth_measured: 0.4744,
      neck_location_model: 0.5,
      neck_location_measured: 0.5,
      profile_stations: [],
      per_frame: [],
      residual_convergence: {},
    },
    ...over,
  }) as PhysicsValidation;

describe("runCapability", () => {
  it("separates a run that lost its architecture from one that failed", () => {
    // The checkpoint is on disk either way; what a run without a config
    // snapshot lost is the recipe to rebuild the net around it.
    const withConfig = runCapability(RUN, DETAIL());
    expect(withConfig.replayable).toBe(true);

    const withoutConfig = runCapability(
      RUN,
      DETAIL({
        config: null,
        artifacts: { ...DETAIL().artifacts, config: false },
      }),
    );
    expect(withoutConfig.replayable).toBe(false);
    // And it claims nothing about the field set it cannot read.
    expect(withoutConfig.fields).toEqual([]);
  });

  it("reads the field set the run recorded, not the flags around it", () => {
    expect(runCapability(RUN, DETAIL()).fields).toEqual([
      "phi",
      "u",
      "v",
      "s",
      "p",
    ]);
  });

  it("tells a run that never measured the front velocity from one missing its report", () => {
    expect(runCapability(RUN, DETAIL()).frontVelocityRequested).toBe(false);
    expect(
      runCapability(
        RUN,
        DETAIL({
          config: { training: { front_velocity: true } },
        }),
      ).frontVelocityRequested,
    ).toBe(true);
  });
});

describe("physicsVerdict", () => {
  it("scores the unsupervised agreement a modern run actually recorded", () => {
    // The old check read `iou_holdout ?? val_iou_mean` — null on every run
    // since the single-frame holdout was retired — and so reported the
    // generalization evidence as absent while it sat at 0.893.
    expect(unsupervisedIou(VALIDATION())).toBeCloseTo(0.8928, 4);
    const verdict = physicsVerdict(RUN, VALIDATION(), "Series-1");
    const check = verdict.measured.find((c) => c.id === "unsupervised")!;
    expect(check.value).toBe("0.893");
    expect(check.ok).toBe(true);
  });

  it("states a tolerance on every measured check, and flags what exceeds it", () => {
    const verdict = physicsVerdict(RUN, VALIDATION(), "Series-1");
    expect(verdict.measured.every((check) => check.against.length > 0)).toBe(
      true,
    );
    // 0.629 against a 0.25 band, and a neck 30 % shallower than measured.
    expect(verdict.flags.map((check) => check.id).sort()).toEqual([
      "laplace-front",
      "neck",
    ]);
  });

  it("puts a missing input in 'not run' rather than failing it", () => {
    // Series-1 records no measured nose speed. Nothing failed; a number was
    // never entered.
    const verdict = physicsVerdict(RUN, VALIDATION(), "Series-1");
    expect(verdict.measured.some((check) => check.id === "nose")).toBe(false);
    expect(verdict.notRun.map((check) => check.id)).toContain("nose");
    expect(verdict.notRun[0].reason).toMatch(/Series-1/);
  });

  it("says a run with no held-out frame has no unsupervised evidence", () => {
    const verdict = physicsVerdict(
      RUN,
      VALIDATION({ iou_val: null, validation_frames: [] }),
      "Series-1",
    );
    expect(verdict.notRun.map((check) => check.id)).toContain("unsupervised");
  });
});

describe("tabBadges", () => {
  it("marks the tabs a run without a snapshot cannot fill", () => {
    const capability = runCapability(
      RUN,
      DETAIL({
        config: null,
        artifacts: { ...DETAIL().artifacts, config: false },
      }),
    );
    const badges = tabBadges(RUN, capability, VALIDATION(), 3, "Series-1");

    expect(badges.recon).toMatchObject({ text: "gated", warn: true });
    expect(badges.fields).toMatchObject({ text: "gated", warn: true });
    // And every badge that warns explains itself on hover.
    expect(badges.recon?.title).toMatch(/config snapshot/);
  });

  it("counts what a tab holds, and warns when it holds nothing", () => {
    const capability = runCapability(RUN, DETAIL());
    const badges = tabBadges(RUN, capability, VALIDATION(), 1, "Series-1");

    expect(badges.agreement).toEqual({ text: "11" });
    expect(badges.fields).toEqual({ text: "5" });
    expect(badges.physics).toMatchObject({ text: "2 flags", warn: true });
    expect(badges.velocity).toMatchObject({ text: "none", warn: true });
    // One trained run in the project is not a comparison.
    expect(badges.compare).toMatchObject({ text: "1 run", warn: true });
  });
});
