import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { MARK_VIEWBOX, vorticityPoints } from "../src/components/BrandMark";
import { buildSvg, points } from "../scripts/build-brand.mjs";

/**
 * The mark is drawn twice: once by the React component for the app, once by
 * scripts/build-brand.mjs for the favicons, which cannot import TSX. These
 * tests are what stops the two drifting into different logos.
 */

describe("the mark's geometry", () => {
  it("is generated identically by the component and the asset script", () => {
    const fromComponent = vorticityPoints().map((p) => [
      Number(p.cx.toFixed(6)),
      Number(p.cy.toFixed(6)),
      Number(p.r.toFixed(6)),
      Number(p.opacity.toFixed(6)),
    ]);
    const fromScript = points().map((p) => [
      Number(p.cx.toFixed(6)),
      Number(p.cy.toFixed(6)),
      Number(p.r.toFixed(6)),
      Number(p.opacity.toFixed(6)),
    ]);
    expect(fromScript).toEqual(fromComponent);
  });

  it("stays inside the viewBox at every point", () => {
    const escaping = vorticityPoints().filter(
      (p) =>
        p.cx - p.r < 0 ||
        p.cy - p.r < 0 ||
        p.cx + p.r > MARK_VIEWBOX ||
        p.cy + p.r > MARK_VIEWBOX,
    );
    expect(escaping).toEqual([]);
  });

  it("decays outward, which is what reads as rotation standing still", () => {
    const spiral = vorticityPoints();
    const radii = spiral.map((p) => p.r);
    const opacities = spiral.map((p) => p.opacity);
    // Strictly decreasing: a dot further out is never larger or brighter.
    expect(radii.every((r, i) => i === 0 || r < radii[i - 1])).toBe(true);
    expect(opacities.every((o, i) => i === 0 || o < opacities[i - 1])).toBe(
      true,
    );
  });
});

describe("the committed brand assets", () => {
  it("match what the script generates today", () => {
    // A stale favicon is invisible in every test that only renders the app, so
    // the committed file is compared against a fresh build instead.
    const committed = readFileSync(
      resolve("public/brand/navnet-mark.svg"),
      "utf8",
    );
    expect(committed).toBe(buildSvg());
  });

  it("carry both themes' ink, since a favicon cannot inherit one", () => {
    const svg = buildSvg();
    expect(svg).toContain("prefers-color-scheme: dark");
    // The light and dark --primary from tokens.css.
    expect(svg).toContain("#2e6cca");
    expect(svg).toContain("#7fb2f0");
  });

  it("is small enough to inline, unlike the traced mark it replaces", () => {
    // The previous asset was a 6,143-byte outline trace of the same idea.
    expect(buildSvg().length).toBeLessThan(1500);
  });
});
