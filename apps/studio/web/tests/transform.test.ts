import { describe, expect, it } from "vitest";

import {
  fitView,
  nearestOnRings,
  pixelToUm,
  sampleGrid,
  toScreen,
  toUm,
  umToPixel,
  zoomAt,
} from "../src/lib/transform";

describe("transform", () => {
  const extent = { x0: 0, x1: 300, y0: 0, y1: 150 };

  it("maps µm to screen and back", () => {
    const view = fitView(extent, { width: 800, height: 400 });
    const [sx, sy] = toScreen(view, 120, 30);
    const [x, y] = toUm(view, sx, sy);
    expect(x).toBeCloseTo(120, 6);
    expect(y).toBeCloseTo(30, 6);
  });

  it("keeps the point under the cursor fixed when zooming", () => {
    const view = fitView(extent, { width: 800, height: 400 });
    const before = toUm(view, 300, 200);
    const zoomed = zoomAt(view, 300, 200, 3);
    const after = toUm(zoomed, 300, 200);
    expect(after[0]).toBeCloseTo(before[0], 6);
    expect(after[1]).toBeCloseTo(before[1], 6);
  });

  it("inverts the raw-pixel mapping (x flipped, y offset by the ROI)", () => {
    const map = { width: 1024, umPerPx: 1.6854, roiTop: 472, cropTop: 445 };
    const [col, row] = umToPixel(map, 500, 150);
    const [x, y] = pixelToUm(map, col, row);
    expect(x).toBeCloseTo(500, 3);
    expect(y).toBeCloseTo(150, 3);
  });

  it("returns NaN outside the field and falls back to a defined neighbour beside a gap", () => {
    const plane = new Float32Array([1, 2, Number.NaN, 4]);
    expect(sampleGrid(plane, 2, 2, [0, 1, 0, 1], 2, 0.5)).toBeNaN();
    expect(sampleGrid(plane, 2, 2, [0, 1, 0, 1], 0.5, 0.5)).toBe(4);
  });

  it("finds the nearest boundary vertex", () => {
    const rings = [
      [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
    ];
    const near = nearestOnRings(rings, 9, 1);
    expect(near).toEqual({ x: 10, y: 0, distance: Math.hypot(1, 1) });
  });
});
