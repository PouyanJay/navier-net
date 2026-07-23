import { useEffect, useRef } from "react";

import type { ConsoleLine } from "../lib/api";

interface ConsoleProps {
  lines: ConsoleLine[];
  label: string;
}

/**
 * The solver console: streamed monospace log lines on the dark view surface.
 * `role="log"` makes it a polite live region so screen readers hear progress
 * (enterprise-ui §3); it pins itself to the newest line as output arrives.
 */
export function Console({ lines, label }: ConsoleProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines.length]);

  return (
    <div ref={ref} className="console" role="log" aria-label={label}>
      {lines.map((entry, i) => (
        <div key={i} className={entry.tone ?? undefined}>
          {entry.line}
        </div>
      ))}
    </div>
  );
}
