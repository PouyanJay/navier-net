import { expect, test, type Page } from "@playwright/test";

import { CANVAS_TOKENS, THEME_VARYING } from "../tokenContract";

/**
 * The token layer, resolved by a real browser.
 *
 * tests/tokens.test.ts asserts what tokens.css declares; only a real engine
 * resolves var(), color-mix() and the cascade, so the assertions that matter —
 * "every alias points somewhere", "the canvas ignores the theme" — live here.
 */

/** Reads the given custom properties off <html> as the browser resolved them. */
async function resolved(
  page: Page,
  tokens: readonly string[],
): Promise<Record<string, string>> {
  return page.evaluate((names) => {
    const style = getComputedStyle(document.documentElement);
    return Object.fromEntries(
      names.map((name) => [name, style.getPropertyValue(name).trim()]),
    );
  }, tokens);
}

async function setTheme(page: Page, theme: "light" | "dark"): Promise<void> {
  await page.evaluate((next) => {
    document.documentElement.dataset.theme = next;
  }, theme);
}

/** Every alias the shim declares, paired with the role it must resolve to. */
const SHIM: ReadonlyArray<readonly [string, string]> = [
  ["--shell", "--bg"],
  ["--panel", "--surface"],
  ["--panel2", "--control"],
  ["--ink", "--text"],
  ["--ink2", "--text-soft"],
  ["--ink3", "--muted"],
  ["--side", "--chrome-surface"],
  ["--side2", "--surface-hover"],
  ["--sideline", "--line"],
  ["--sidetxt", "--muted"],
  ["--sidehl", "--text"],
  ["--view", "--canvas"],
  ["--view2", "--canvas-raised"],
  ["--viewline", "--canvas-rule"],
  ["--viewtxt", "--canvas-quiet"],
  ["--acc", "--primary"],
  ["--acc2", "--primary-hover"],
  ["--accsoft", "--primary-soft"],
  ["--green", "--success"],
  ["--greensoft", "--success-soft"],
  ["--amber", "--holdout"],
  ["--ambersoft", "--holdout-soft"],
  ["--red", "--danger"],
  ["--redsoft", "--danger-soft"],
  ["--redtext", "--danger-text"],
  ["--purple", "--series-3"],
  ["--teal", "--series-5"],
  ["--sh", "--shadow-chrome"],
  ["--mono", "--font-mono"],
  ["--sans", "--font-ui"],
  ["--serif", "--font-display"],
  ["--f-phi", "--field-phi"],
  ["--f-u", "--field-u"],
  ["--f-v", "--field-v"],
  ["--f-s", "--field-s"],
  ["--f-p", "--field-p"],
  ["--f-t", "--field-t"],
];

for (const theme of ["dark", "light"] as const) {
  test(`every shim alias resolves to its role in ${theme}`, async ({
    page,
  }) => {
    await page.goto("/");
    await setTheme(page, theme);

    const values = await resolved(page, SHIM.flat());

    const unresolved = SHIM.map(([alias]) => alias).filter(
      (alias) => values[alias] === "",
    );
    expect(unresolved, "aliases that resolve to nothing").toEqual([]);

    const mismatched = SHIM.filter(
      ([alias, role]) => values[alias] !== values[role],
    ).map(([alias, role]) => `${alias} → ${role}`);
    expect(mismatched, "aliases pointing at the wrong role").toEqual([]);
  });
}

test("the canvas ramp is identical in both themes", async ({ page }) => {
  await page.goto("/");

  await setTheme(page, "dark");
  const dark = await resolved(page, CANVAS_TOKENS);
  await setTheme(page, "light");
  const light = await resolved(page, CANVAS_TOKENS);

  const drifted = CANVAS_TOKENS.filter((token) => dark[token] !== light[token]);
  expect(drifted, "canvas tokens that changed with the theme").toEqual([]);

  // Two empty strings are "identical", so a deleted token would pass the check
  // above on its own.
  const unset = CANVAS_TOKENS.filter((token) => dark[token] === "");
  expect(unset, "canvas tokens that resolve to nothing").toEqual([]);
});

test("every theme-varying token actually changes value", async ({ page }) => {
  // Redeclaring a token in the light block is not the same as giving it a
  // different value. A copy-pasted dark value would satisfy the static contract
  // in tests/tokens.test.ts and ship a light theme that is wrong in exactly the
  // places nobody screenshots.
  await page.goto("/");

  await setTheme(page, "dark");
  const dark = await resolved(page, THEME_VARYING);
  await setTheme(page, "light");
  const light = await resolved(page, THEME_VARYING);

  const unset = THEME_VARYING.filter((token) => dark[token] === "");
  expect(unset, "theme-varying tokens that resolve to nothing").toEqual([]);

  const unchanged = THEME_VARYING.filter(
    (token) => dark[token] === light[token],
  );
  expect(unchanged, "tokens that did not change with the theme").toEqual([]);
});

test("light and dark both paint the page from the token layer", async ({
  page,
}) => {
  await page.goto("/");

  for (const theme of ["dark", "light"] as const) {
    await setTheme(page, theme);
    const painted = await page.evaluate(() => {
      const style = getComputedStyle(document.body);
      const root = getComputedStyle(document.documentElement);
      return {
        background: style.backgroundColor,
        color: style.color,
        bg: root.getPropertyValue("--bg").trim(),
        colorScheme: root.colorScheme,
      };
    });
    // The body paints --bg, so a missing token layer would leave it transparent.
    expect(painted.background).not.toBe("rgba(0, 0, 0, 0)");
    expect(painted.color).not.toBe("");
    expect(painted.colorScheme).toBe(theme);
  }
});
