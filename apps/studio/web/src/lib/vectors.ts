/** Arrows and streamlines of a vector field held on the studio grid. Both are
 * built in micrometres on a lattice whose spacing the scene chooses from its
 * zoom, so the picture keeps one density on screen however far in or out the
 * view is. Cells inside the vapour hold NaN and are left empty. */

import type { Extent } from "./transform";

export interface VectorField {
  u: Float32Array;
  v: Float32Array;
  nx: number;
  ny: number;
  /** x0, x1, y0, y1 in µm, as the grid header gives it. */
  extent: [number, number, number, number];
}

export interface Arrow {
  /** Where the arrow is centred (µm). */
  x: number;
  y: number;
  /** Unit direction, y toward the upper wall. */
  dx: number;
  dy: number;
  magnitude: number;
}

export interface Streamline {
  /** Along the flow, from the upstream end. */
  points: [number, number][];
  /** Where the arrowhead sits: the midpoint and the flow direction there. */
  head: { x: number; y: number; dx: number; dy: number } | null;
}

// A line is traced half a grid cell at a time and no further than this many
// steps each way from its seed, so a closed orbit cannot run forever.
const STEP_CELLS = 0.5;
const MAX_STEPS = 600;
// Lines are kept apart by an occupancy grid of half the lattice spacing: a
// line stops on entering a cell another line has claimed (Jobard and Lefer's
// idea in its simplest form).
const OCCUPANCY_CELLS_PER_LATTICE = 2;
// The occupancy grid covers the visible window grown by this fraction each
// way, so lines run a little past the edge of the screen but the grid stays
// small however far the view is zoomed in.
const REGION_MARGIN = 0.5;
// Shorter lines than this (in lattice spacings) are clutter, not flow.
const MIN_LENGTH_LATTICE = 0.75;
const TINY = 1e-12;

/** Bilinear sample of both components. NaN whenever a corner is missing, so
 * nothing is drawn across the bubble edge. */
export function sampleVector(field: VectorField, x: number, y: number): [number, number] {
  const { nx, ny, extent } = field;
  const [x0, x1, y0, y1] = extent;
  const fx = ((x - x0) / (x1 - x0)) * (nx - 1);
  const fy = ((y - y0) / (y1 - y0)) * (ny - 1);
  if (!(fx >= 0 && fy >= 0 && fx <= nx - 1 && fy <= ny - 1)) return [NaN, NaN];
  const i = Math.min(nx - 2, Math.floor(fx));
  const j = Math.min(ny - 2, Math.floor(fy));
  const a = fx - i;
  const b = fy - j;
  const k = j * nx + i;
  const w00 = (1 - a) * (1 - b);
  const w10 = a * (1 - b);
  const w01 = (1 - a) * b;
  const w11 = a * b;
  const u = field.u;
  const v = field.v;
  return [
    w00 * u[k] + w10 * u[k + 1] + w01 * u[k + nx] + w11 * u[k + nx + 1],
    w00 * v[k] + w10 * v[k + 1] + w01 * v[k + nx] + w11 * v[k + nx + 1],
  ];
}

/** The lattice points `spacing` apart inside the visible window, anchored to
 * the field's origin so a pan never shifts them. */
export function lattice(
  extent: [number, number, number, number],
  visible: Extent,
  spacing: number,
): [number, number][] {
  if (!(spacing > 0)) return [];
  const [x0, , y0] = extent;
  const i0 = Math.floor((visible.x0 - x0) / spacing);
  const i1 = Math.ceil((visible.x1 - x0) / spacing);
  const j0 = Math.floor((visible.y0 - y0) / spacing);
  const j1 = Math.ceil((visible.y1 - y0) / spacing);
  const out: [number, number][] = [];
  for (let j = j0; j < j1; j++) {
    const y = y0 + (j + 0.5) * spacing;
    if (y < visible.y0 || y > visible.y1) continue;
    for (let i = i0; i < i1; i++) {
      const x = x0 + (i + 0.5) * spacing;
      if (x < visible.x0 || x > visible.x1) continue;
      out.push([x, y]);
    }
  }
  return out;
}

/** One arrow per lattice point that has a finite, non-zero vector. */
export function arrows(field: VectorField, visible: Extent, spacing: number): Arrow[] {
  const out: Arrow[] = [];
  for (const [x, y] of lattice(field.extent, visible, spacing)) {
    const [u, v] = sampleVector(field, x, y);
    const magnitude = Math.hypot(u, v);
    if (!(magnitude > TINY)) continue;
    out.push({ x, y, dx: u / magnitude, dy: v / magnitude, magnitude });
  }
  return out;
}

function direction(field: VectorField, x: number, y: number): [number, number] | null {
  const [u, v] = sampleVector(field, x, y);
  const magnitude = Math.hypot(u, v);
  if (!(magnitude > TINY)) return null;
  return [u / magnitude, v / magnitude];
}

/** One Runge-Kutta step of length h along the unit direction field. */
function rk4(field: VectorField, x: number, y: number, h: number): [number, number] | null {
  const k1 = direction(field, x, y);
  if (!k1) return null;
  const k2 = direction(field, x + 0.5 * h * k1[0], y + 0.5 * h * k1[1]);
  if (!k2) return null;
  const k3 = direction(field, x + 0.5 * h * k2[0], y + 0.5 * h * k2[1]);
  if (!k3) return null;
  const k4 = direction(field, x + h * k3[0], y + h * k3[1]);
  if (!k4) return null;
  return [
    x + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
    y + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
  ];
}

class Occupancy {
  readonly owner: Int32Array;
  private readonly cols: number;
  private readonly rows: number;

  constructor(
    private readonly region: Extent,
    private readonly cell: number,
  ) {
    this.cols = Math.max(1, Math.ceil((region.x1 - region.x0) / cell));
    this.rows = Math.max(1, Math.ceil((region.y1 - region.y0) / cell));
    this.owner = new Int32Array(this.cols * this.rows);
  }

  /** The cell holding the point, or -1 outside the region. */
  cellOf(x: number, y: number): number {
    const c = Math.floor((x - this.region.x0) / this.cell);
    const r = Math.floor((y - this.region.y0) / this.cell);
    if (c < 0 || r < 0 || c >= this.cols || r >= this.rows) return -1;
    return r * this.cols + c;
  }
}

/** Trace from (x, y) one way; the claimed cells are recorded so a line that
 * turns out too short can give them back. */
function trace(
  field: VectorField,
  x: number,
  y: number,
  step: number,
  id: number,
  occupancy: Occupancy,
  claimed: number[],
): [number, number][] {
  const points: [number, number][] = [];
  for (let n = 0; n < MAX_STEPS; n++) {
    const next = rk4(field, x, y, step);
    if (!next) break;
    [x, y] = next;
    const k = occupancy.cellOf(x, y);
    if (k < 0) break;
    const owner = occupancy.owner[k];
    if (owner !== 0 && owner !== id) break;
    if (owner === 0) {
      occupancy.owner[k] = id;
      claimed.push(k);
    }
    points.push([x, y]);
  }
  return points;
}

function pathLength(points: [number, number][]): number {
  let total = 0;
  for (let i = 1; i < points.length; i++)
    total += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
  return total;
}

/** The midpoint by arc length and the flow direction there. */
function headOf(points: [number, number][]): Streamline["head"] {
  if (points.length < 3) return null;
  const half = pathLength(points) / 2;
  let run = 0;
  for (let i = 1; i < points.length; i++) {
    const [ax, ay] = points[i - 1];
    const [bx, by] = points[i];
    const seg = Math.hypot(bx - ax, by - ay);
    if (run + seg >= half && seg > 0) {
      const f = (half - run) / seg;
      return {
        x: ax + f * (bx - ax),
        y: ay + f * (by - ay),
        dx: (bx - ax) / seg,
        dy: (by - ay) / seg,
      };
    }
    run += seg;
  }
  return null;
}

/** Evenly spaced streamlines seeded on the lattice inside the visible window. */
export function streamlines(field: VectorField, visible: Extent, spacing: number): Streamline[] {
  if (!(spacing > 0)) return [];
  const [x0, x1, y0, y1] = field.extent;
  const step = STEP_CELLS * Math.min((x1 - x0) / (field.nx - 1), (y1 - y0) / (field.ny - 1));
  const marginX = REGION_MARGIN * (visible.x1 - visible.x0);
  const marginY = REGION_MARGIN * (visible.y1 - visible.y0);
  const region: Extent = {
    x0: Math.max(x0, visible.x0 - marginX),
    x1: Math.min(x1, visible.x1 + marginX),
    y0: Math.max(y0, visible.y0 - marginY),
    y1: Math.min(y1, visible.y1 + marginY),
  };
  const occupancy = new Occupancy(region, spacing / OCCUPANCY_CELLS_PER_LATTICE);
  const out: Streamline[] = [];
  let id = 0;
  for (const [sx, sy] of lattice(field.extent, visible, spacing)) {
    const seedCell = occupancy.cellOf(sx, sy);
    if (seedCell < 0 || occupancy.owner[seedCell] !== 0) continue;
    if (!direction(field, sx, sy)) continue;
    id += 1;
    const claimed = [seedCell];
    occupancy.owner[seedCell] = id;
    const forward = trace(field, sx, sy, step, id, occupancy, claimed);
    const backward = trace(field, sx, sy, -step, id, occupancy, claimed);
    const points: [number, number][] = [...backward.reverse(), [sx, sy], ...forward];
    if (pathLength(points) < MIN_LENGTH_LATTICE * spacing) {
      for (const k of claimed) occupancy.owner[k] = 0;
      continue;
    }
    out.push({ points, head: headOf(points) });
  }
  return out;
}
