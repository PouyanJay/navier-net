/** Coordinate transforms. The scene works in micrometres with y increasing
 * toward the upper wall; the screen has y increasing downward; the raw camera
 * frame is flipped in x relative to the flow and offset in rows. */

export interface Extent {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

export interface View {
  /** Screen pixels per micrometre. */
  scale: number;
  /** Screen position of the µm origin. */
  tx: number;
  ty: number;
}

export interface Size {
  width: number;
  height: number;
}

/** The view that fits the extent in the box with a margin, preserving aspect. */
export function fitView(extent: Extent, box: Size, margin = 16): View {
  const w = Math.max(1e-9, extent.x1 - extent.x0);
  const h = Math.max(1e-9, extent.y1 - extent.y0);
  const scale = Math.min((box.width - 2 * margin) / w, (box.height - 2 * margin) / h);
  const tx = (box.width - w * scale) / 2 - extent.x0 * scale;
  const ty = (box.height - h * scale) / 2 + extent.y1 * scale;
  return { scale, tx, ty };
}

export function toScreen(view: View, xUm: number, yUm: number): [number, number] {
  return [view.tx + xUm * view.scale, view.ty - yUm * view.scale];
}

export function toUm(view: View, sx: number, sy: number): [number, number] {
  return [(sx - view.tx) / view.scale, (view.ty - sy) / view.scale];
}

/** Zoom about a screen point by a factor. */
export function zoomAt(
  view: View,
  sx: number,
  sy: number,
  factor: number,
  limits = [0.2, 60],
): View {
  const scale = Math.min(limits[1], Math.max(limits[0], view.scale * factor));
  const k = scale / view.scale;
  return { scale, tx: sx - (sx - view.tx) * k, ty: sy - (sy - view.ty) * k };
}

export function pan(view: View, dx: number, dy: number): View {
  return { ...view, tx: view.tx + dx, ty: view.ty + dy };
}

/** The extent (µm) a raw-frame crop covers, from the service's transform. */
export interface PixelMap {
  width: number;
  umPerPx: number;
  roiTop: number;
  cropTop: number;
}

/** µm -> raw pixel (column, row), as preprocessing defines it. */
export function umToPixel(map: PixelMap, xUm: number, yUm: number): [number, number] {
  const col = map.width - 0.5 - xUm / map.umPerPx;
  const row = yUm / map.umPerPx - 0.5 + map.roiTop;
  return [col, row];
}

export function pixelToUm(map: PixelMap, col: number, row: number): [number, number] {
  return [(map.width - 0.5 - col) * map.umPerPx, (row + 0.5 - map.roiTop) * map.umPerPx];
}

/** The µm extent of a cropped frame (rows cropTop..cropBottom), x flipped. */
export function cropExtent(map: PixelMap, cropBottom: number): Extent {
  const [xLeft] = pixelToUm(map, map.width, 0);
  const [xRight] = pixelToUm(map, 0, 0);
  const [, yTop] = pixelToUm(map, 0, map.cropTop - 0.5);
  const [, yBottom] = pixelToUm(map, 0, cropBottom - 0.5);
  return { x0: xLeft, x1: xRight, y0: yTop, y1: yBottom };
}

export function svgPath(ring: number[][], view: View): string {
  if (ring.length === 0) return "";
  return (
    ring
      .map(([x, y], i) => {
        const [sx, sy] = toScreen(view, x, y);
        return `${i === 0 ? "M" : "L"}${sx.toFixed(1)} ${sy.toFixed(1)}`;
      })
      .join(" ") + " Z"
  );
}

export function polylinePath(points: [number, number][], view: View): string {
  return points
    .map(([x, y], i) => {
      const [sx, sy] = toScreen(view, x, y);
      return `${i === 0 ? "M" : "L"}${sx.toFixed(1)} ${sy.toFixed(1)}`;
    })
    .join(" ");
}

/** Nearest vertex of any ring to a point, with its distance. */
export function nearestOnRings(
  rings: number[][][],
  xUm: number,
  yUm: number,
): { x: number; y: number; distance: number } | null {
  let best: { x: number; y: number; distance: number } | null = null;
  for (const ring of rings) {
    for (const [x, y] of ring) {
      const d = Math.hypot(x - xUm, y - yUm);
      if (!best || d < best.distance) best = { x, y, distance: d };
    }
  }
  return best;
}

/** Bilinear sample of a row-major plane over an extent; NaN outside or on gaps. */
export function sampleGrid(
  plane: Float32Array,
  nx: number,
  ny: number,
  extent: [number, number, number, number],
  xUm: number,
  yUm: number,
): number {
  const [x0, x1, y0, y1] = extent;
  const fx = ((xUm - x0) / (x1 - x0)) * (nx - 1);
  const fy = ((yUm - y0) / (y1 - y0)) * (ny - 1);
  if (fx < 0 || fy < 0 || fx > nx - 1 || fy > ny - 1) return NaN;
  const i = Math.min(nx - 2, Math.floor(fx));
  const j = Math.min(ny - 2, Math.floor(fy));
  const a = fx - i;
  const b = fy - j;
  const v00 = plane[j * nx + i];
  const v10 = plane[j * nx + i + 1];
  const v01 = plane[(j + 1) * nx + i];
  const v11 = plane[(j + 1) * nx + i + 1];
  if ([v00, v10, v01, v11].some((v) => !Number.isFinite(v))) {
    const nearest = plane[Math.round(fy) * nx + Math.round(fx)];
    return Number.isFinite(nearest) ? nearest : NaN;
  }
  return (1 - b) * ((1 - a) * v00 + a * v10) + b * ((1 - a) * v01 + a * v11);
}
