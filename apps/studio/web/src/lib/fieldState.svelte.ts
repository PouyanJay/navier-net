/** What the Fields and Residuals views share: the grid of the current
 * quantity at the current frame, prefetch of the other frames, the colormap
 * and the range rule. */

import { api } from "./api/client";
import type { Grid, Quantity } from "./api/types";
import { defaultColormap, type ColormapName } from "./colormaps";
import { GRID_NX, GRID_NY, studio } from "../stores/studio.svelte";
import { niceRange, percentileRange } from "./units";

export class FieldState {
  grid = $state<Grid | null>(null);
  error = $state<string | null>(null);
  busy = $state(false);
  /** Ranges of every event frame of the current quantity, for the "all" rule. */
  allRange = $state<[number, number] | null>(null);

  readonly quantity = $derived<Quantity | null>(studio.current);
  readonly colormap = $derived<ColormapName>(
    studio.colormap === "auto"
      ? defaultColormap(studio.current?.cmap ?? "sequential", studio.theme)
      : studio.colormap,
  );
  readonly range = $derived.by((): [number, number] => {
    const q = studio.current;
    if (studio.rangeMode === "custom") return studio.customRange;
    if (studio.rangeMode === "all" && this.allRange)
      return niceRange(this.allRange[0], this.allRange[1], !!q?.signed);
    if (!this.grid) return [0, 1];
    const [lo, hi] = percentileRange(this.grid.planes[0]);
    return niceRange(lo, hi, !!q?.signed);
  });

  constructor() {
    $effect(() => {
      const run = studio.runId;
      const quantity = studio.quantity;
      const frame = studio.frame;
      if (!run || !studio.current?.available) {
        this.grid = null;
        return;
      }
      let cancelled = false;
      this.busy = true;
      api
        .getGrid(run, [quantity], frame, GRID_NX, GRID_NY)
        .then((grid) => {
          if (cancelled) return;
          this.grid = grid;
          this.error = null;
        })
        .catch((error: Error) => {
          if (!cancelled) this.error = error.message;
        })
        .finally(() => {
          if (!cancelled) this.busy = false;
        });
      return () => {
        cancelled = true;
      };
    });

    // Prefetch the other event frames so scrubbing and play never wait.
    $effect(() => {
      const run = studio.runId;
      const quantity = studio.quantity;
      const frames = studio.eventFrames.map((f) => f.frame);
      if (!run || !studio.current?.available || frames.length === 0) return;
      let cancelled = false;
      this.allRange = null;
      Promise.all(frames.map((f) => api.getGrid(run, [quantity], f, GRID_NX, GRID_NY))).then(
        (grids) => {
          if (cancelled) return;
          let lo = Infinity;
          let hi = -Infinity;
          for (const g of grids) {
            const [a, b] = percentileRange(g.planes[0]);
            lo = Math.min(lo, a);
            hi = Math.max(hi, b);
          }
          this.allRange = [lo, hi];
        },
      );
      return () => {
        cancelled = true;
      };
    });
  }
}
