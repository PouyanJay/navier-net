/** What a run can still answer, and what it cannot.
 *
 * The canvas asks a run for two different kinds of thing. Most tabs read what
 * the evaluate stage *wrote* — metrics, per-frame IoU, the trajectory, the
 * physics diagnostics — and those survive anything. The reconstruction and the
 * field maps instead ask the checkpoint for values at arbitrary points, which
 * means rebuilding the network, which needs the architecture from the run's
 * config snapshot.
 *
 * A run without that snapshot therefore keeps every measurement it ever made
 * and can never be replayed. That is a knowable state, not a failure — and
 * saying so once, here, is what stops two tabs showing a red error box that
 * blames a missing model while `checkpoints/ckpt.pt` sits on disk.
 */

import type { RunDetail, RunSummary } from "../../lib/api";

/** Every field the platform can train, in the order the Physics stage lists
 * them. A run's own set is a subset, recorded in its config as `model.fields`. */
export const ALL_FIELDS = ["phi", "u", "v", "s", "p", "T"] as const;

export interface RunCapability {
  /** The network can be rebuilt, so the player and the field maps can run. */
  replayable: boolean;
  /** The model fields this run trains; empty when it recorded no config. */
  fields: string[];
  /** The evaluate stage wrote a front-velocity report. */
  hasFrontVelocity: boolean;
  /** Whether the run asked for the measured front velocity at all — the
   * difference between "never measured" and "measured, report missing". */
  frontVelocityRequested: boolean | null;
  figures: number;
  hasVideo: boolean;
  /** Frames the evaluation covered. */
  nFrames: number | null;
}

type Section = Record<string, unknown> | undefined;

const section = (config: RunDetail["config"], name: string): Section =>
  (config as Record<string, Section> | null)?.[name];

function flag(config: RunDetail["config"], group: string, key: string) {
  const value = section(config, group)?.[key];
  return typeof value === "boolean" ? value : null;
}

export function runCapability(
  run: RunSummary,
  detail: RunDetail | null,
): RunCapability {
  const config = detail?.config ?? null;
  const artifacts = detail?.artifacts;
  // The composed field set is recorded verbatim; never inferred from the
  // equation flags, which would drift from what the run actually built.
  const recorded = section(config, "model")?.["fields"];
  const fields = Array.isArray(recorded)
    ? ALL_FIELDS.filter((name) => recorded.includes(name))
    : [];

  return {
    // `artifacts.config` is the API's own answer; the presence of the snapshot
    // in `detail.config` is the same fact, and covers an older API.
    replayable: artifacts?.config ?? config != null,
    fields,
    hasFrontVelocity: artifacts?.front_velocity ?? false,
    frontVelocityRequested: flag(config, "training", "front_velocity"),
    figures: artifacts?.figures.length ?? 0,
    hasVideo: artifacts?.video ?? false,
    nFrames: run.n_frames ?? null,
  };
}
