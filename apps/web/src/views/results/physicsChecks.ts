/** The physics checks a run's own measurements support.
 *
 * A check is a measurement with a verdict. The tab used to mix three other
 * things in with them under the same failure mark: a constant that said the
 * same thing about every run ever trained, a check reading a retired metric so
 * it reported the generalization evidence as absent while it sat at 0.89, and a
 * missing input rendered as a failure. Those are a note, a bug, and a
 * configuration — none of them a failed check.
 *
 * So this returns two lists. `measured` holds checks that ran, each with the
 * tolerance it is read against. `notRun` holds the ones whose input the series
 * never recorded, with what would supply it.
 */

import type { PhysicsValidation, RunSummary } from "../../lib/api";
import { NOSE_SPEED_TOLERANCE_PCT } from "./format";

/** Above this the pressure field is not tracking the interface's own curvature
 * — the threshold the R4 physics gate is stated against. */
export const JUMP_TOLERANCE = 0.25;

/** A neck this much shallower than the measured one is the R3 failure mode,
 * and reads as a flag however good the IoU is. */
const NECK_SHORTFALL = 0.2;

export interface PhysicsCheck {
  id: string;
  name: string;
  /** What was compared, in the interface's own voice. */
  detail: string;
  value: string;
  /** The tolerance or reference the value is read against. */
  against: string;
  ok: boolean;
}

export interface UnrunCheck {
  id: string;
  name: string;
  /** Why it could not run, and what would let it. */
  reason: string;
}

export interface PhysicsVerdict {
  measured: PhysicsCheck[];
  notRun: UnrunCheck[];
  /** The measured checks outside their tolerance, worst first. */
  flags: PhysicsCheck[];
}

/** The in-distribution score, walking the same ladder the run list leads with:
 * `iou_holdout` belongs to the retired single-frame holdout and `val_iou_mean`
 * only to joint runs, so the old pair reported "no unsupervised agreement" for
 * every ordinary evaluated run. */
export function unsupervisedIou(
  validation: PhysicsValidation | null,
): number | null {
  return (
    validation?.val_iou_mean ??
    validation?.iou_val ??
    validation?.iou_holdout ??
    null
  );
}

export function physicsVerdict(
  run: RunSummary,
  validation: PhysicsValidation | null,
  seriesName: string,
): PhysicsVerdict {
  const physics = validation?.physics ?? null;
  const measured: PhysicsCheck[] = [];
  const notRun: UnrunCheck[] = [];

  if (physics?.laplace_error_front != null) {
    measured.push({
      id: "laplace-front",
      name: "Young–Laplace · front",
      detail: "pressure jump across the whole interface",
      value: physics.laplace_error_front.toFixed(3),
      against: `tol ${JUMP_TOLERANCE}`,
      ok: physics.laplace_error_front <= JUMP_TOLERANCE,
    });
  }
  if (physics?.laplace_error_nose != null) {
    measured.push({
      id: "laplace-nose",
      name: "Young–Laplace · nose",
      detail: "the cap's own curvature",
      value: physics.laplace_error_nose.toFixed(3),
      against: `tol ${JUMP_TOLERANCE}`,
      ok: physics.laplace_error_nose <= JUMP_TOLERANCE,
    });
  }
  if (
    physics?.neck_depth_model != null &&
    physics.neck_depth_measured != null
  ) {
    const short =
      physics.neck_depth_measured > 0
        ? 1 - physics.neck_depth_model / physics.neck_depth_measured
        : null;
    measured.push({
      id: "neck",
      name: "Neck depth",
      detail:
        physics.neck_depth_model === 0
          ? "the model forms no neck at all"
          : "model against the measured mask",
      value: physics.neck_depth_model.toFixed(3),
      against:
        short == null
          ? `vs ${physics.neck_depth_measured.toFixed(3)}`
          : `vs ${physics.neck_depth_measured.toFixed(3)} · ${(short * 100).toFixed(0)} % short`,
      ok: short == null || short <= NECK_SHORTFALL,
    });
  }
  if (physics?.axial_capillary_gradient != null) {
    measured.push({
      id: "axial",
      name: "Axial capillary gradient",
      detail: "along the bubble's body",
      value: physics.axial_capillary_gradient.toFixed(3),
      against: `tol ${JUMP_TOLERANCE}`,
      ok: physics.axial_capillary_gradient <= JUMP_TOLERANCE,
    });
  }

  const noseError = validation?.nose_speed_error_pct ?? null;
  if (noseError != null) {
    measured.push({
      id: "nose",
      name: "Nose-speed agreement",
      detail: "inferred against measured; neither was given to the model",
      value: `${noseError.toFixed(1)} %`,
      against: `tol ${NOSE_SPEED_TOLERANCE_PCT} %`,
      ok: noseError < NOSE_SPEED_TOLERANCE_PCT,
    });
  } else {
    notRun.push({
      id: "nose",
      name: "Nose-speed agreement",
      reason: `${seriesName} records no measured nose speed to compare against.`,
    });
  }

  const unsupervised = unsupervisedIou(validation);
  if (unsupervised != null) {
    const frames = validation?.validation_frames ?? [];
    measured.push({
      id: "unsupervised",
      name: "Unsupervised agreement",
      detail: frames.length
        ? `frames ${frames.join(", ")}, never supervised`
        : "frames the model was never shown",
      value: unsupervised.toFixed(3),
      against: "the generalization evidence",
      ok: true,
    });
  } else {
    notRun.push({
      id: "unsupervised",
      name: "Unsupervised agreement",
      reason:
        "This run held no frames out of training, so there is no unseen frame to score. Set a validation split in the Solver.",
    });
  }
  void run;

  return {
    measured,
    notRun,
    flags: measured.filter((check) => !check.ok),
  };
}
