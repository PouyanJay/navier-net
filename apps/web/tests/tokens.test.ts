import { readFileSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  CANVAS_TOKENS,
  THEME_INDEPENDENT,
  THEME_VARYING,
  VOCABULARY,
} from "./tokenContract";

/**
 * The token layer's contract with DESIGN_SYSTEM.md §2.2.
 *
 * tokens.css is the artifact this suite verifies, so it reads the real file
 * rather than a fixture: jsdom does not resolve var(), so a computed-style test
 * here would assert nothing. Resolved values are covered in a real browser by
 * tests/e2e/tokens.spec.ts.
 */

// The jsdom environment gives import.meta.url an http: origin, so resolve from
// the Vitest root instead.
const TOKENS_CSS = readFileSync(resolve("src/tokens.css"), "utf8");

/** Every file that can carry a token reference, for whole-app assertions. */
const SOURCE_FILES = readdirSync(resolve("src"), {
  recursive: true,
  encoding: "utf8",
})
  .filter((name) => /\.(css|tsx?)$/.test(name))
  .map((name) => ({
    path: join("src", name),
    text: readFileSync(resolve("src", name), "utf8"),
  }));

/** Every pre-migration name, which the shim keeps resolving until Phase 1. */
const LEGACY_TOKENS = [
  "--shell",
  "--panel",
  "--panel2",
  "--ink",
  "--ink2",
  "--ink3",
  "--side",
  "--side2",
  "--sideline",
  "--sidetxt",
  "--sidehl",
  "--view",
  "--view2",
  "--viewline",
  "--viewtxt",
  "--acc",
  "--acc2",
  "--accsoft",
  "--green",
  "--greensoft",
  "--amber",
  "--ambersoft",
  "--red",
  "--redsoft",
  "--redtext",
  "--purple",
  "--teal",
  "--sh",
  "--r",
  "--rs",
  "--mono",
  "--sans",
  "--serif",
  "--f-phi",
  "--f-u",
  "--f-v",
  "--f-s",
  "--f-p",
  "--f-t",
];

/** Returns the body of the first rule whose selector list matches exactly. */
function ruleBody(selector: string): string {
  const start = TOKENS_CSS.indexOf(`${selector} {`);
  expect(start, `tokens.css has no "${selector}" rule`).toBeGreaterThanOrEqual(
    0,
  );
  const open = TOKENS_CSS.indexOf("{", start);
  const close = TOKENS_CSS.indexOf("}", open);
  return TOKENS_CSS.slice(open + 1, close);
}

function declaredIn(body: string): Set<string> {
  return new Set(
    [...body.matchAll(/^\s*(--[a-z0-9-]+)\s*:/gm)].map((m) => m[1]),
  );
}

const rootTokens = declaredIn(ruleBody(":root"));
const lightTokens = declaredIn(ruleBody(':root[data-theme="light"]'));

describe("token vocabulary (DESIGN_SYSTEM §2.2)", () => {
  for (const [group, tokens] of Object.entries(VOCABULARY)) {
    it(`declares every ${group} token on :root`, () => {
      const missing = tokens.filter((token) => !rootTokens.has(token));
      expect(missing).toEqual([]);
    });
  }
});

describe("themes (DESIGN_SYSTEM §2.1, §3)", () => {
  it("makes dark the default by putting it on :root itself", () => {
    expect(ruleBody(":root")).toMatch(/color-scheme:\s*dark/);
    expect(ruleBody(':root[data-theme="light"]')).toMatch(
      /color-scheme:\s*light/,
    );
  });

  it("redefines every theme-varying token in light", () => {
    // Presence only. That a token's light *value* actually differs from its
    // dark one is asserted in tests/e2e/tokens.spec.ts, where a browser has
    // resolved both — a copy-pasted dark value would satisfy this check.
    const missing = THEME_VARYING.filter((token) => !lightTokens.has(token));
    expect(missing).toEqual([]);
  });

  it("keeps the canvas and its series identical across themes", () => {
    const themed = CANVAS_TOKENS.filter((token) => lightTokens.has(token));
    expect(themed).toEqual([]);
  });

  it("does not re-declare dimensionless tokens per theme", () => {
    const themed = THEME_INDEPENDENT.filter((token) => lightTokens.has(token));
    expect(themed).toEqual([]);
  });
});

describe("the Phase 1 compatibility shim", () => {
  it("keeps every pre-migration name resolving", () => {
    const missing = LEGACY_TOKENS.filter((token) => !rootTokens.has(token));
    expect(missing).toEqual([]);
  });

  it("aliases onto the new vocabulary rather than restating values", () => {
    const body = ruleBody(":root");
    const literal = LEGACY_TOKENS.filter((token) => {
      const declaration = new RegExp(`^\\s*${token}\\s*:([^;]+);`, "m").exec(
        body,
      );
      return declaration !== null && !declaration[1].includes("var(--");
    });
    // --r/--rs are the exception: radius becomes contextual in Phase 2 (§5.1),
    // so there is no single new token to alias them onto.
    expect(literal).toEqual(["--r", "--rs"]);
  });

  it("is confined to :root, so no theme can fork the aliases", () => {
    const leaked = LEGACY_TOKENS.filter((token) => lightTokens.has(token));
    expect(leaked).toEqual([]);
  });

  it("cannot alias the line pair, so that pair is already migrated", () => {
    // The old --line2 was the hairline, so the old --line and the new --line
    // are one name with inverted meanings; an alias would silently downgrade
    // every structural border to a hairline. Both were renamed in Phase 0.
    expect(rootTokens.has("--line2")).toBe(false);
    const stragglers = SOURCE_FILES.filter(({ text }) =>
      text.includes("var(--line2)"),
    ).map(({ path }) => path);
    expect(stragglers).toEqual([]);
  });
});

describe("contrast (WCAG 2.2 AA, in both themes)", () => {
  /** Resolves a token to sRGB, following var() aliases within its own theme. */
  function srgb(
    token: string,
    theme: Map<string, string>,
  ): [number, number, number] {
    let value = theme.get(token);
    expect(value, `${token} is not declared`).toBeDefined();
    while (value!.startsWith("var(")) {
      const alias = value!.slice(4, value!.indexOf(")"));
      value = theme.get(alias);
      expect(value, `${token} aliases undeclared ${alias}`).toBeDefined();
    }
    const hex = /^#([0-9a-f]{6})$/.exec(value!.trim());
    if (hex) {
      const n = parseInt(hex[1], 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    const rgb = /^rgb\(\s*(\d+)\s+(\d+)\s+(\d+)/.exec(value!.trim());
    expect(rgb, `${token} is not an opaque colour: ${value}`).not.toBeNull();
    return [Number(rgb![1]), Number(rgb![2]), Number(rgb![3])];
  }

  function ratio(a: [number, number, number], b: [number, number, number]) {
    const luminance = (c: [number, number, number]) => {
      const [r, g, bl] = c.map((v) => {
        const x = v / 255;
        return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * r + 0.7152 * g + 0.0722 * bl;
    };
    const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
  }

  function declarations(body: string): Map<string, string> {
    return new Map(
      [...body.matchAll(/^\s*(--[a-z0-9-]+)\s*:\s*([^;]+);/gm)].map((m) => [
        m[1],
        m[2].trim(),
      ]),
    );
  }

  const dark = declarations(ruleBody(":root"));
  const light = new Map([
    ...dark,
    ...declarations(ruleBody(':root[data-theme="light"]')),
  ]);

  /** Text that can appear at body size, so it owes the full 4.5:1. */
  const TEXT_PAIRS: ReadonlyArray<readonly [string, string]> = [
    ["--text", "--bg"],
    ["--text-soft", "--bg"],
    ["--muted", "--bg"],
    ["--faint", "--bg"],
    ["--text", "--surface"],
    ["--text-soft", "--surface"],
    ["--muted", "--surface"],
    ["--faint", "--surface"],
    ["--primary", "--bg"],
    ["--primary", "--surface"],
    ["--success", "--surface"],
    ["--holdout", "--surface"],
    ["--danger", "--surface"],
    ["--danger-text", "--danger-surface"],
    ["--primary-ink", "--primary"],
    // The canvas ramp is theme-independent, so one check covers both themes.
    ["--canvas-ink", "--canvas"],
    ["--canvas-ink-strong", "--canvas"],
    ["--canvas-quiet", "--canvas"],
    ["--console-text", "--canvas"],
  ];

  /** Series and focus rings are graphics, held to the 3:1 non-text threshold. */
  const GRAPHIC_PAIRS: ReadonlyArray<readonly [string, string]> = [
    ["--series-1", "--canvas"],
    ["--series-2", "--canvas"],
    ["--series-3", "--canvas"],
    ["--series-4", "--canvas"],
    ["--series-5", "--canvas"],
    ["--focus", "--bg"],
    ["--focus", "--surface"],
  ];

  for (const [theme, tokens] of [
    ["dark", dark],
    ["light", light],
  ] as const) {
    it(`meets 4.5:1 for every text pair in ${theme}`, () => {
      const failures = TEXT_PAIRS.map(([fg, bg]) => {
        const r = ratio(srgb(fg, tokens), srgb(bg, tokens));
        return { pair: `${fg} on ${bg}`, ratio: Number(r.toFixed(2)) };
      }).filter(({ ratio: r }) => r < 4.5);
      expect(failures).toEqual([]);
    });

    it(`meets 3:1 for every graphic pair in ${theme}`, () => {
      const failures = GRAPHIC_PAIRS.map(([fg, bg]) => {
        const r = ratio(srgb(fg, tokens), srgb(bg, tokens));
        return { pair: `${fg} on ${bg}`, ratio: Number(r.toFixed(2)) };
      }).filter(({ ratio: r }) => r < 3);
      expect(failures).toEqual([]);
    });
  }
});

describe("no reference dangles", () => {
  it("declares every custom property the app references", () => {
    // The failure this guards is silent: an unresolved var() paints nothing at
    // all rather than erroring, so a typo during Phase 1's ~1,300-reference
    // rename would reach the browser looking like a styling bug. Declarations
    // are collected from every source file, not just tokens.css, because the
    // shell declares its own layout properties locally.
    const declared = new Set<string>();
    const referenced = new Map<string, string>();

    for (const { path, text } of SOURCE_FILES) {
      for (const [, name] of text.matchAll(/^\s*(--[a-z0-9-]+)\s*:/gm)) {
        declared.add(name);
      }
      // Custom properties set from TSX, e.g. { "--stage-aside-w": "320px" }.
      for (const [, name] of text.matchAll(/["'](--[a-z0-9-]+)["']\s*:/g)) {
        declared.add(name);
      }
      for (const [, name] of text.matchAll(/var\((--[a-z0-9-]+)/g)) {
        if (!referenced.has(name)) referenced.set(name, path);
      }
    }

    const dangling = [...referenced]
      .filter(([name]) => !declared.has(name))
      .map(([name, path]) => `${name} (first seen in ${path})`);
    expect(dangling).toEqual([]);
  });
});

describe("the chart readout is drawn in canvas ink", () => {
  /** One rule's declarations, by exact selector. */
  function ruleBody(css: string, selector: string): string {
    const at = css.indexOf(`\n${selector} {`);
    expect(at, `${selector} not found`).toBeGreaterThan(-1);
    return css.slice(at, css.indexOf("}", at));
  }

  /* Chrome ink and its shim aliases. Any of these on the canvas inverts
     underneath itself, because the canvas does not follow the theme. */
  const CHROME_INK = [
    "--text",
    "--text-soft",
    "--muted",
    "--faint",
    "--ink",
    "--ink2",
    "--ink3",
    "--sidetxt",
    "--sidehl",
  ];

  it("keeps the crosshair and its tooltip off chrome ink", () => {
    // The regression this guards was live: .chart-tip resolved --sidetxt and
    // .tip-x resolved --sidehl, which are --muted and --text. On the dark view
    // canvas in the light theme that is near-black on near-black — the tooltip
    // measured 1.06:1 and was, in practice, invisible.
    const css = readFileSync(resolve("src/components/components.css"), "utf8");
    const offenders = [".chart-cursor", ".chart-tip", ".chart-tip .tip-x"]
      .flatMap((selector) => {
        const body = ruleBody(css, selector);
        return CHROME_INK.filter((token) =>
          new RegExp(`var\\(${token}[),]`).test(body),
        ).map((token) => `${selector} uses ${token}`);
      })
      .sort();
    expect(offenders).toEqual([]);
  });
});

describe("icons come from the library, not by hand", () => {
  /**
   * Files allowed to draw SVG directly: charts, the reconstruction viewport,
   * the ensemble, and the brand mark. Everything in that list renders DATA or
   * the logo. A UI glyph drawn by hand drifts from the set around it -- wrong
   * stroke, wrong grid, wrong optical size -- and cannot be swapped in one file
   * the way DESIGN_SYSTEM §7 requires.
   */
  const DRAWS_ITS_OWN = [
    "src/components/BrandMark.tsx",
    "src/components/ReconstructionViewport.tsx",
    "src/components/charts/",
    "src/views/datasets/FrameLightbox.tsx",
    "src/views/datasets/QcPanel.tsx",
    "src/views/physics/EnsembleCanvas.tsx",
    "src/views/results/FrameMatchPanel.tsx",
  ];

  it("has no hand-drawn glyph outside the visualisation files", () => {
    const offenders = SOURCE_FILES.filter(
      ({ path, text }) =>
        /\.tsx$/.test(path) &&
        text.includes("<svg") &&
        !DRAWS_ITS_OWN.some((allowed) => path.startsWith(allowed)),
    ).map(({ path }) => path);
    expect(offenders).toEqual([]);
  });

  it("routes every icon through the one barrel", () => {
    // Importing straight from @hugeicons in a view defeats the point: the
    // barrel is what keeps the bundle to referenced glyphs and makes swapping
    // libraries a one-file edit.
    const direct = SOURCE_FILES.filter(
      ({ path, text }) =>
        path !== "src/components/icons.ts" && text.includes("@hugeicons/"),
    ).map(({ path }) => path);
    expect(direct).toEqual([]);
  });
});
