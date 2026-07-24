import type { RunSummary } from "./api";

/** The single definition of "this run trained" / "this run was evaluated",
 * shared by the shell's stage flags and the project cards so they can't
 * drift on what those words mean. */
export const isTrainedRun = (run: RunSummary) => run.status === "trained";
export const hasEvaluation = (run: RunSummary) => run.iou_holdout != null;
