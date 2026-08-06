/** What each output tab holds, before you click it.
 *
 * The strip looked identical whether a tab held six panels or an empty state,
 * so finding out cost a click every time. These badges are read from what the
 * page has already loaded — the run summary, its detail, its validation — and
 * never from a fetch of their own.
 */

import type { PhysicsValidation, RunSummary } from "../../lib/api";
import { physicsVerdict } from "./physicsChecks";
import type { RunCapability } from "./runCapability";
import type { ResultTabId } from "./ResultsPage";

export interface TabBadge {
  /** The short text on the badge. */
  text: string;
  /** Worth attention: nothing there, or something outside tolerance. */
  warn?: boolean;
  /** Why, for the tab's title attribute. */
  title?: string;
}

export function tabBadges(
  run: RunSummary,
  capability: RunCapability,
  validation: PhysicsValidation | null,
  runCount: number,
  seriesName: string,
): Partial<Record<ResultTabId, TabBadge>> {
  const verdict = physicsVerdict(run, validation, seriesName);
  const frames = run.n_frames ?? capability.nFrames;
  const badges: Partial<Record<ResultTabId, TabBadge>> = {};

  if (!capability.replayable) {
    badges.recon = {
      text: "gated",
      warn: true,
      title: "This run recorded no config snapshot, so it cannot be replayed",
    };
    badges.fields = {
      text: "gated",
      warn: true,
      title: "Field maps need the network rebuilt from a config snapshot",
    };
  } else if (capability.fields.length > 0) {
    badges.fields = { text: String(capability.fields.length) };
  }

  if (frames) badges.agreement = { text: String(frames) };

  if (verdict.measured.length > 0) {
    badges.physics = verdict.flags.length
      ? {
          text: `${verdict.flags.length} flag${verdict.flags.length === 1 ? "" : "s"}`,
          warn: true,
          title: "Measured checks outside their tolerance",
        }
      : { text: "ok" };
  }

  badges.velocity = capability.hasFrontVelocity
    ? { text: "ok" }
    : {
        text: "none",
        warn: true,
        title:
          capability.frontVelocityRequested === false
            ? "This run did not measure the front's velocity"
            : "No front-velocity report was written",
      };

  if (runCount < 2) {
    badges.compare = {
      text: "1 run",
      warn: true,
      title: "Comparison needs a second trained run in this project",
    };
  }

  const downloads = 2 + (capability.hasFrontVelocity ? 1 : 0);
  badges.export = capability.figures
    ? { text: `${capability.figures} figures` }
    : { text: String(downloads) };

  return badges;
}
