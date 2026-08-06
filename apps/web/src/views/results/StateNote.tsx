import type { ReactNode } from "react";

import { HugeiconsIcon, InfoIcon, WarningIcon } from "../../components/icons";

interface StateNoteProps {
  /** `info` states a configuration; `caution` states something worth acting on. */
  tone?: "info" | "caution";
  title: string;
  children: ReactNode;
}

/**
 * Why a panel is showing what it is showing.
 *
 * A panel with nothing in it is either broken or configured that way, and the
 * difference matters enormously: the run this stage cannot replay is not a
 * failed run, it is a run that kept its measurements and lost its architecture.
 * A red error Callout said the first thing about the second, so this states the
 * situation and what would change it instead.
 */
export function StateNote({ tone = "info", title, children }: StateNoteProps) {
  return (
    <p className={tone === "caution" ? "statenote caution" : "statenote"}>
      <HugeiconsIcon
        icon={tone === "caution" ? WarningIcon : InfoIcon}
        size={14}
        aria-hidden="true"
      />
      <span>
        <b>{title}</b> {children}
      </span>
    </p>
  );
}

/** The run cannot be replayed, said the same way wherever that bites. */
export function NotReplayable({ runId }: { runId: string }) {
  return (
    <StateNote tone="caution" title="This run cannot be replayed.">
      <span className="mono">{runId}</span> recorded no{" "}
      <span className="mono">.hydra/config.yaml</span>, so the network cannot be
      rebuilt around its checkpoint. Its measurements are unaffected: agreement,
      kinematics, physics and every CSV read from what the evaluate stage wrote.
      Re-running the evaluate stage on a run launched from the Solver restores
      the player.
    </StateNote>
  );
}
