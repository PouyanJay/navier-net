import type { ReactNode } from "react";

type ChipTone = "default" | "accent" | "green" | "amber" | "red";

interface ChipProps {
  children: ReactNode;
  tone?: ChipTone;
}

/** A small pill for labels, tags, and short statuses. */
export function Chip({ children, tone = "default" }: ChipProps) {
  return (
    <span className="chip" data-tone={tone}>
      {children}
    </span>
  );
}
