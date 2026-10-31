import { describe, expect, it } from "vitest";

import { boundaryTone, buildLut, colormap, LUT_SIZE } from "../src/lib/colormaps";

describe("colormaps", () => {
  it("returns in-range channels across the whole ramp", () => {
    for (const name of [
      "viridis",
      "inferno",
      "magma",
      "plasma",
      "cividis",
      "grey",
      "diverging",
    ] as const) {
      for (let i = 0; i <= 10; i++) {
        const rgb = colormap(name, i / 10);
        for (const c of rgb) expect(c).toBeGreaterThanOrEqual(0);
        for (const c of rgb) expect(c).toBeLessThanOrEqual(1);
      }
    }
  });

  it("builds an opaque lookup table of the expected size", () => {
    const lut = buildLut("viridis");
    expect(lut.length).toBe(LUT_SIZE * 4);
    expect(lut[3]).toBe(255);
    expect(lut[LUT_SIZE * 4 - 1]).toBe(255);
  });

  it("gives a boundary tone that is visible on every ramp", () => {
    for (const name of ["viridis", "inferno", "magma", "plasma", "cividis", "grey"] as const) {
      expect(["light", "dark"]).toContain(boundaryTone(name));
    }
  });
});
