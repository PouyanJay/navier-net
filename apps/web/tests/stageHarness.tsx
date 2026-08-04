import { render } from "@testing-library/react";
import { useMemo, useState, type ReactNode } from "react";

import { AsideSlotProvider } from "../src/app/StageAside";

/**
 * Stands in for AppShell's secondary aside.
 *
 * A stage renders its inputs into that slot, so a test without one would mount
 * none of them. This provides a real node exactly as the shell does, leaving
 * the shell's own chrome — heading, collapse, persistence — to
 * `stageAside.test.tsx`, which exercises the real `AppShell` directly.
 */
export function StageHarness({ children }: { children: ReactNode }) {
  const [node, setNode] = useState<HTMLElement | null>(null);
  const slot = useMemo(() => ({ node, claim: () => {} }), [node]);
  return (
    <AsideSlotProvider slot={slot}>
      <div ref={setNode} />
      {children}
    </AsideSlotProvider>
  );
}

export function renderStage(ui: ReactNode) {
  return render(<StageHarness>{ui}</StageHarness>);
}
