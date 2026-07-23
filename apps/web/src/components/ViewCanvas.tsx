import type { ReactNode } from "react";

/** The dark visualization surface D3 charts render onto (enterprise-ui §2). */
export function ViewCanvas({ children }: { children: ReactNode }) {
  return <div className="view-canvas">{children}</div>;
}
