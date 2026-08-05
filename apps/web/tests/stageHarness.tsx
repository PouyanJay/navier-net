import { render } from "@testing-library/react";
import { useMemo, useState, type ReactNode } from "react";

import { AsideSlotProvider } from "../src/app/StageAside";
import { StageHeaderProvider } from "../src/app/StageHeader";

/**
 * Stands in for the two slots AppShell gives a stage: the secondary aside and
 * the header row.
 *
 * A stage portals its inputs into the first and the identity of what it is
 * showing into the second, so a test without them would mount neither. This
 * provides real nodes exactly as the shell does, leaving the shell's own chrome
 * — heading, collapse, persistence — to `stageAside.test.tsx`, which exercises
 * the real `AppShell` directly.
 */
export function StageHarness({ children }: { children: ReactNode }) {
  const [node, setNode] = useState<HTMLElement | null>(null);
  const [header, setHeader] = useState<HTMLElement | null>(null);
  const slot = useMemo(() => ({ node, claim: () => {} }), [node]);
  return (
    <AsideSlotProvider slot={slot}>
      <StageHeaderProvider node={header}>
        <div ref={setHeader} />
        <div ref={setNode} />
        {children}
      </StageHeaderProvider>
    </AsideSlotProvider>
  );
}

export function renderStage(ui: ReactNode) {
  return render(<StageHarness>{ui}</StageHarness>);
}
