/** Per-run data several views share: the pixel transform, the audit, and the
 * interface contour of the current frame. */

import { api } from "./api/client";
import type { Audit, InterfaceContour, Transform } from "./api/types";
import { studio } from "../stores/studio.svelte";

export class RunData {
  transform = $state<Transform | null>(null);
  audit = $state<Audit | null>(null);
  contour = $state<InterfaceContour | null>(null);
  measured = $state<number[][][]>([]);

  constructor() {
    $effect(() => {
      const run = studio.runId;
      if (!run) return;
      let cancelled = false;
      api.getTransform(run).then((t) => {
        if (!cancelled) this.transform = t;
      });
      api
        .getAudit(run)
        .then((a) => {
          if (!cancelled) this.audit = a;
        })
        .catch(() => {
          if (!cancelled) this.audit = null;
        });
      return () => {
        cancelled = true;
      };
    });
    $effect(() => {
      const run = studio.runId;
      const frame = studio.frame;
      if (!run) return;
      let cancelled = false;
      api.getInterface(run, frame).then((c) => {
        if (!cancelled) this.contour = c;
      });
      api
        .getAgreement(run, frame)
        .then((a) => {
          if (!cancelled) this.measured = a.measured_rings_um;
        })
        .catch(() => {
          if (!cancelled) this.measured = [];
        });
      return () => {
        cancelled = true;
      };
    });
  }

  get extent() {
    const d = studio.run?.domain_um;
    if (!d) return { x0: 0, x1: 1, y0: 0, y1: 1 };
    const pad = 12;
    return { x0: d.x[0] - pad, x1: d.x[1] + pad, y0: d.y_walls[0] - pad, y1: d.y_walls[1] + pad };
  }
}

/** Series colours cycle through the palette's six tokens. */
export function seriesColor(index: number): string {
  return `var(--series-${(index % 6) + 1})`;
}
