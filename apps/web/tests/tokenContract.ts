/**
 * The token vocabulary of DESIGN_SYSTEM.md §2.2, shared by the two suites that
 * check it — tests/tokens.test.ts (what tokens.css declares) and
 * tests/e2e/tokens.spec.ts (what a browser resolves). They must agree on the
 * list, so it lives here rather than in either one.
 */

/** Grouped as the specification groups them, so a group maps to a spec table. */
export const VOCABULARY = {
  ground: [
    "--bg",
    "--surface",
    "--surface-overlay",
    "--surface-hover",
    "--chrome-surface",
    "--chrome-backdrop",
    "--modal-surface",
    "--modal-scrim",
    "--notice",
  ],
  controls: [
    "--control",
    "--control-hover",
    "--control-strong",
    "--control-border",
    "--track",
  ],
  ink: ["--text", "--text-soft", "--muted", "--faint"],
  lines: ["--line", "--line-strong"],
  accent: [
    "--primary",
    "--primary-hover",
    "--primary-ink",
    "--primary-soft",
    "--holdout",
    "--holdout-soft",
    "--success",
    "--success-soft",
    "--danger",
    "--danger-soft",
    "--danger-surface",
    "--danger-text",
    "--focus",
    "--hover",
    "--hover-strong",
    "--selection",
    "--selection-ink",
  ],
  canvas: [
    "--canvas",
    "--canvas-stage",
    "--canvas-raised",
    "--canvas-ink",
    "--canvas-ink-strong",
    "--canvas-quiet",
    "--canvas-rule",
    "--canvas-scroll-thumb",
    "--focus-item-bg",
    "--focus-item-ink",
  ],
  series: [
    "--series-1",
    "--series-2",
    "--series-3",
    "--series-4",
    "--series-5",
    "--series-measured",
    "--series-model",
  ],
  /** Canvas residents: the solver console's tones. */
  console: [
    "--console-text",
    "--console-ok",
    "--console-em",
    "--console-dim",
    "--console-err",
  ],
  /** Canvas residents: the network ensemble's per-field hues. */
  field: [
    "--field-phi",
    "--field-u",
    "--field-v",
    "--field-s",
    "--field-p",
    "--field-t",
  ],
  structure: [
    "--chrome-size",
    "--topbar-height",
    "--rail-width",
    "--rail-collapsed-width",
    "--aside-width",
    "--aside-collapsed-width",
    "--brand-slot-width",
    "--radius",
    "--ease",
  ],
  type: ["--font-ui", "--font-display", "--font-mono"],
  spacing: ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7"],
  elevation: [
    "--shadow-chrome",
    "--shadow-menu",
    "--shadow-flyout",
    "--shadow-tooltip",
    "--shadow-dialog",
  ],
} as const;

/**
 * Tokens the light theme must give a *different* value, not merely redeclare.
 * `--chrome-backdrop` is excluded because the frosted recipe is one blur value
 * that both themes share.
 */
export const THEME_VARYING: readonly string[] = [
  ...VOCABULARY.ground,
  ...VOCABULARY.controls,
  ...VOCABULARY.ink,
  ...VOCABULARY.lines,
  ...VOCABULARY.accent,
  ...VOCABULARY.elevation,
].filter((token) => token !== "--chrome-backdrop");

/**
 * Everything that lives on the canvas. The canvas is dark in both themes, so
 * every one of these must resolve to the same value regardless of the theme —
 * and none may be defined in terms of a theme-varying token.
 */
export const CANVAS_TOKENS: readonly string[] = [
  ...VOCABULARY.canvas,
  ...VOCABULARY.series,
  ...VOCABULARY.console,
  ...VOCABULARY.field,
];

/** Dimensionless: neither theme may restate these. */
export const THEME_INDEPENDENT: readonly string[] = [
  ...VOCABULARY.structure,
  ...VOCABULARY.type,
  ...VOCABULARY.spacing,
];
