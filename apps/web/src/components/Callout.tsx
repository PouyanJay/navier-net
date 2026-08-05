import type { ReactNode } from "react";

import { ErrorIcon, HugeiconsIcon, InfoIcon, WarningIcon } from "./icons";

export type CalloutTone = "error" | "caution" | "info";

interface CalloutProps {
  tone: CalloutTone;
  /** Short lead-in, rendered emphasised before the body (e.g. "Preprocessing failed"). */
  title?: string;
  children: ReactNode;
  className?: string;
}

/**
 * A message that needs to be read, not skimmed past: a failure, a caution, or a
 * state the user has to act on. The tone drives the colour, the icon, and the
 * live-region role, so no caller has to remember to pair them.
 *
 * Neutral status text ("Loading…", "No runs yet") stays a plain `.state-note`;
 * boxing everything would flatten the distinction this component exists to make.
 */
export function Callout({ tone, title, children, className }: CalloutProps) {
  return (
    <div
      className={["callout", tone, className].filter(Boolean).join(" ")}
      role={tone === "error" ? "alert" : "status"}
    >
      <CalloutIcon tone={tone} />
      <p>
        {title && <b>{title}</b>}
        {children}
      </p>
    </div>
  );
}

/**
 * State is carried by the icon's shape as well as its colour (WCAG 1.4.1), so
 * the three tones take structurally different glyphs — a circle, a triangle and
 * a crossed circle — rather than one shape in three colours.
 */
const TONE_ICON: Record<CalloutTone, typeof InfoIcon> = {
  info: InfoIcon,
  caution: WarningIcon,
  error: ErrorIcon,
};

function CalloutIcon({ tone }: { tone: CalloutTone }) {
  return <HugeiconsIcon icon={TONE_ICON[tone]} size={16} />;
}
