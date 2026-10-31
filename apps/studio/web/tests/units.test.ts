import { describe, expect, it } from "vitest";

import { fmt, niceRange } from "../src/lib/units";

describe("units", () => {
  it("leaves nulls blank rather than zero", () => {
    expect(fmt(null)).toBe("");
    expect(fmt(Number.NaN)).toBe("");
  });

  it("uses stated precision and exponents for extremes", () => {
    expect(fmt(0)).toBe("0");
    expect(fmt(1.23456, 3)).toBe("1.23");
    expect(fmt(2e6, 3)).toMatch(/e/);
  });

  it("centres signed ranges on zero", () => {
    expect(niceRange(-3, 1, true)).toEqual([-3, 3]);
    expect(niceRange(0, 2, false)).toEqual([0, 2]);
  });
});
