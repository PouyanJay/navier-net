import type { RunSummary } from "./api";

/** The single definition of "this run trained" / "this run was evaluated",
 * shared by the shell's stage flags and the project cards so they can't
 * drift on what those words mean. */
export const isTrainedRun = (run: RunSummary) => run.status === "trained";

/**
 * Whether the evaluate stage has produced a score for this run.
 *
 * All four metrics, not the two the summary used to carry: `iou_holdout`
 * belongs to the retired single-frame holdout and `val_iou_mean` only to joint
 * runs, so an ordinary evaluated run answered "no" here — and the Results dot
 * on its project card stayed unlit however many times it had been evaluated.
 */
export const hasEvaluation = (run: RunSummary) =>
  run.iou_holdout != null ||
  run.val_iou_mean != null ||
  run.iou_val != null ||
  run.iou_mean != null;

/** Comparing more runs than the chart has series colours would un-name the
 * lines, so both the rail that assembles a comparison and the tab that draws
 * one stop at four. */
export const MAX_COMPARED = 4;
