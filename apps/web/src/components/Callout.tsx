import type { ReactNode } from "react";

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

/** State is carried by the icon shape as well as the colour (WCAG 1.4.1). */
function CalloutIcon({ tone }: { tone: CalloutTone }) {
  if (tone === "info") {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <circle
          cx="8"
          cy="8"
          r="6.4"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.4"
        />
        <circle cx="8" cy="5" r="0.95" fill="currentColor" />
        <path
          d="M8 7.4v4.2"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinecap="round"
        />
      </svg>
    );
  }
  if (tone === "caution") {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <path
          d="M8 1.9 15 14H1Z"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinejoin="round"
        />
        <path
          d="M8 6.2v3.4"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinecap="round"
        />
        <circle cx="8" cy="11.7" r="0.95" fill="currentColor" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <circle
        cx="8"
        cy="8"
        r="6.4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
      />
      <path
        d="M5.6 5.6 10.4 10.4M10.4 5.6 5.6 10.4"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}
