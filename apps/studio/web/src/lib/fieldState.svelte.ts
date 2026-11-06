/** What the Fields and Residuals views share: the grid of the current
 * quantity at the current frame, prefetch of the other frames, the colormap
 * and the range rule. */

import { api } from "./api/client";
import type { Grid, Quantity } from "./api/types";
import { defaultColormap, type ColormapName } from "./colormaps";
import { GRID_NX, GRID_NY, studio } from "../stores/studio.svelte";
import { niceRange, percentileRange } from "./units";

const SCRUB_DEBOUNCE_MS = 150;

export class FieldState {
  /** The quantity of the grid on screen; plain, so reading it does not re-run effects. */
  private shownQuantity: string | null = null;
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
    // The grid for the frame on screen. Scrubbing waits a moment before asking,
    // so the server is not queued with a request for every frame passed.
    $effect(() => {
      const run = studio.runId;
      const quantity = studio.quantity;
      const frame = studio.frame;
      // A picture of another quantity must not be drawn with this quantity's
      // colour scale, so the old grid goes as soon as the quantity changes.
      if (this.shownQuantity !== quantity) this.grid = null;
      if (!run || !studio.current?.available) {
        this.grid = null;
        return;
      }
      let cancelled = false;
      this.busy = true;
      const wait = setTimeout(() => {
        api
          .getGrid(run, [quantity], frame, GRID_NX, GRID_NY)
          .then((grid) => {
            if (cancelled) return;
            this.shownQuantity = quantity;
            this.grid = grid;
            this.error = null;
          })
          .catch((error: Error) => {
            if (!cancelled) this.error = error.message;
          })
          .finally(() => {
            if (!cancelled) this.busy = false;
          });
      }, SCRUB_DEBOUNCE_MS);
      return () => {
        cancelled = true;
        clearTimeout(wait);
      };
    });

    // The colour range over all event frames is only needed for the "all frames"
    // rule. Those grids are requested one at a time, after the frame on screen.
    $effect(() => {
      const run = studio.runId;
      const quantity = studio.quantity;
      const mode = studio.rangeMode;
      const frames = studio.eventFrames.map((f) => f.frame);
      this.allRange = null;
      if (!run || mode !== "all" || !studio.current?.available || frames.length === 0) return;
      let cancelled = false;
      (async () => {
        let lo = Infinity;
        let hi = -Infinity;
        for (const f of frames) {
          if (cancelled) return;
          const grid = await api.getGrid(run, [quantity], f, GRID_NX, GRID_NY);
          const [a, b] = percentileRange(grid.planes[0]);
          lo = Math.min(lo, a);
          hi = Math.max(hi, b);
        }
        if (!cancelled) this.allRange = [lo, hi];
      })().catch(() => {});
      return () => {
        cancelled = true;
      };
    });
  }
}
