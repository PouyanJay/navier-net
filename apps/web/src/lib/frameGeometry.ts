/** Mapping detected/modeled contours onto raw camera frames.
 *
 * The pipeline flips x (raw camera flow runs right to left) and crops the
 * imaged band off the top of the frame; every overlay undoes both. One module
 * so the datasets lightbox and the results frame-matching panel cannot drift
 * on the transform.
 */

export interface FrameGeometry {
  /** Raw frame width in pixels. */
  width: number;
  /** Raw frame height in pixels. */
  height: number;
  /** x*,y* are in L_ref units; multiply by this to get pixels. */
  pxPerStar: number;
  /** Top row of the imaged band the contours were cut to. */
  yRoiTop: number;
}

/** A ring of [x*, y*] points as an SVG path in raw-frame pixel space. */
export function ringPath(ring: number[][], geo: FrameGeometry): string {
  const points = ring.map(([xStar, yStar]) => {
    const col = geo.width - 0.5 - xStar * geo.pxPerStar;
    const row = yStar * geo.pxPerStar - 0.5 + geo.yRoiTop;
    return `${col.toFixed(2)} ${row.toFixed(2)}`;
  });
  return points.length ? `M${points.join("L")}Z` : "";
}

/** A polyline of [x, y] points in µm (the /interface payload's units) as an
 * SVG path in raw-frame pixel space. µm / µm-per-px = px directly. */
export function umPath(
  contour: number[][],
  geo: FrameGeometry,
  umPerPx: number,
): string {
  const points = contour.map(([xUm, yUm]) => {
    const col = geo.width - 0.5 - xUm / umPerPx;
    const row = yUm / umPerPx - 0.5 + geo.yRoiTop;
    return `${col.toFixed(2)} ${row.toFixed(2)}`;
  });
  return points.length ? `M${points.join("L")}` : "";
}
