import { describe, expect, it } from "vitest";
import { arrows, sampleVector, streamlines, type VectorField } from "../src/lib/vectors";

/** A 41 by 21 field over 0..400 by 0..200 µm: uniform flow in +x on the left
 * half, NaN (vapour) on the right half. */
function halfField(): VectorField {
  const nx = 41;
  const ny = 21;
  const u = new Float32Array(nx * ny);
  const v = new Float32Array(nx * ny);
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const vapour = i > 20;
      u[j * nx + i] = vapour ? NaN : 2;
      v[j * nx + i] = vapour ? NaN : 0;
    }
  return { u, v, nx, ny, extent: [0, 400, 0, 200] };
}

const whole = { x0: 0, x1: 400, y0: 0, y1: 200 };

describe("sampleVector", () => {
  it("interpolates inside the liquid and is NaN near a missing corner or outside", () => {
    const field = halfField();
    expect(sampleVector(field, 50, 50)).toEqual([2, 0]);
    expect(sampleVector(field, 205, 50)[0]).toBeNaN();
    expect(sampleVector(field, -1, 50)[0]).toBeNaN();
  });
});

describe("arrows", () => {
  it("fills the liquid lattice points only, pointing along the flow", () => {
    const list = arrows(halfField(), whole, 40);
    // 5 by 5 lattice across 0..400 by 0..200 at 40 µm; the right half is empty.
    expect(list.length).toBe(5 * 5);
    for (const a of list) {
      expect(a.x).toBeLessThan(200);
      expect(a.dx).toBeCloseTo(1);
      expect(a.dy).toBeCloseTo(0);
      expect(a.magnitude).toBeCloseTo(2);
    }
  });

  it("is empty when the spacing is not positive", () => {
    expect(arrows(halfField(), whole, 0)).toEqual([]);
  });
});

describe("streamlines", () => {
  it("runs straight along a uniform flow, stops at the vapour, and keeps lines apart", () => {
    const lines = streamlines(halfField(), whole, 40);
    expect(lines.length).toBeGreaterThan(0);
    const rows = new Set<number>();
    for (const line of lines) {
      const ys = line.points.map((p) => p[1]);
      expect(Math.max(...ys) - Math.min(...ys)).toBeLessThan(1e-6);
      expect(Math.max(...line.points.map((p) => p[0]))).toBeLessThanOrEqual(200);
      expect(line.head).not.toBeNull();
      expect(line.head!.dx).toBeCloseTo(1);
      rows.add(Math.round(ys[0]));
    }
    // One line per lattice row: the others are stopped by the occupancy grid.
    expect(lines.length).toBe(rows.size);
    expect(rows.size).toBe(5);
  });
});
