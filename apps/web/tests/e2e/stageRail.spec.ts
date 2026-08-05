import { expect, test, type Page } from "@playwright/test";

/**
 * The platform's organising rule, checked in a real browser: a stage's
 * configuration lives in its rail, and the canvas carries only what that
 * configuration derives.
 *
 * These assertions need real layout — which element contains which, how wide
 * the rail actually is, whether a popover is clipped by its scroll container —
 * so none of them can be made in jsdom.
 */

async function openStage(page: Page, stage: string | null) {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await page.getByRole("button", { name: /^Open/ }).first().click();
  if (stage) {
    await page.getByRole("button", { name: stage, exact: true }).click();
    // Wait for THIS stage's rail, not merely for a rail: the previous stage's
    // is still mounted for a beat, so a bare `.stage-aside` wait returns while
    // the incoming stage's content is still empty.
    await page
      .getByRole("complementary", { name: stage, exact: true })
      .waitFor();
  } else {
    await page.locator(".stage-aside").waitFor();
  }
}

/** Every control a stage keeps in its rail, by the stage that owns it. */
const IN_THE_RAIL: ReadonlyArray<{
  stage: string | null;
  name: string;
  controls: readonly string[];
}> = [
  {
    stage: null,
    name: "Datasets & conditions",
    controls: ["Edit conditions"],
  },
  {
    stage: "Physics & model",
    name: "Physics & model",
    controls: ["Save"],
  },
  {
    stage: "Solver",
    name: "Solver",
    controls: ["Reset", "Run"],
  },
];

for (const { stage, name, controls } of IN_THE_RAIL) {
  test(`${name} keeps its configuration in the rail`, async ({ page }) => {
    await openStage(page, stage);

    for (const control of controls) {
      const button = page
        .getByRole("button", { name: control, exact: true })
        .first();
      await expect(button, `${control} is missing`).toBeVisible();
      // Inside the rail, not left behind on the canvas.
      const inRail = await button.evaluate(
        (el) => el.closest(".stage-aside") !== null,
      );
      expect(inRail, `${control} is not in the rail`).toBe(true);
    }
  });
}

test("Results keeps its run list in the rail", async ({ page }) => {
  await openStage(page, "Results & validation");

  const list = page.getByRole("listbox", { name: "Runs of this project" });
  await expect(list).toBeVisible();
  expect(await list.evaluate((el) => el.closest(".stage-aside") !== null)).toBe(
    true,
  );
});

test("the Solver's launch controls stay reachable without scrolling", async ({
  page,
}) => {
  await openStage(page, "Solver");

  const body = await page.locator(".stage-aside-body").boundingBox();
  const run = await page
    .getByRole("button", { name: "Run", exact: true })
    .boundingBox();
  expect(body).not.toBeNull();
  expect(run).not.toBeNull();
  // Pinned to the foot of the rail, so a long form never scrolls it away.
  expect(run!.y + run!.height).toBeLessThanOrEqual(body!.y + body!.height + 1);
});

test("surfaces inside the rail are opaque", async ({ page }) => {
  // The rail's own fill is translucent, so anything that borrows it as a card
  // or popover background lets the rows underneath read straight through —
  // which is exactly what a reader sees as "the text is hiding behind things".
  await openStage(page, "Physics & model");

  const resolved = await page.evaluate(() => {
    const rail = document.querySelector(".stage-aside")!;
    const panel = getComputedStyle(rail).getPropertyValue("--panel").trim();
    // Round-trip through a real element so var() and color-mix() are resolved
    // to a concrete colour the way the painted card sees them.
    const probe = document.createElement("div");
    probe.style.backgroundColor = panel;
    document.body.append(probe);
    const painted = getComputedStyle(probe).backgroundColor;
    probe.remove();
    return { declared: panel, painted };
  });

  // Assert the alpha, rather than excluding one spelling of transparency: a
  // token that resolves to nothing paints rgba(0, 0, 0, 0), which is the worst
  // case of this bug and reads as opaque to any "is it not rgba(…, 0.84)" test.
  // Counted, not pattern-matched: "rgb(20, 23, 30)" has three components and an
  // implicit alpha of 1, and a regex looking for a trailing number in it reads
  // the blue channel as the alpha.
  const components = resolved.painted.match(/[\d.]+/g) ?? [];
  const alpha = components.length >= 4 ? Number(components[3]) : 1;
  expect(
    alpha,
    `--panel resolves to ${resolved.painted} (declared ${resolved.declared})`,
  ).toBe(1);
});

test("an equation popover stays inside the rail that scrolls it", async ({
  page,
}) => {
  await openStage(page, "Physics & model");

  await page.locator(".infob").first().hover();
  const pop = page.locator(".infopop").first();
  await expect(pop).toBeVisible();

  const box = (await pop.boundingBox())!;
  const body = (await page.locator(".stage-aside-body").boundingBox())!;
  // A popover wider than its scroll container is clipped, not overhung — the
  // detail simply disappears under the pipeline rail.
  expect(box.x).toBeGreaterThanOrEqual(body.x - 1);
  expect(box.x + box.width).toBeLessThanOrEqual(body.x + body.width + 1);
});

test("the pipeline rail marks its active stage without a border", async ({
  page,
}) => {
  // The active stage used to carry a 2px accent bar down its left edge. State
  // is a tinted plate and accent ink now, so this asserts on the real computed
  // border, which only a layout engine resolves.
  await openStage(page, null);

  const active = page.locator('.nav button[aria-current="page"]');
  const style = await active.evaluate((el) => {
    const c = getComputedStyle(el);
    return {
      widths: [
        c.borderTopWidth,
        c.borderRightWidth,
        c.borderBottomWidth,
        c.borderLeftWidth,
      ],
      background: c.backgroundColor,
      color: c.color,
    };
  });

  expect(style.widths, "the active stage draws a border").toEqual([
    "0px",
    "0px",
    "0px",
    "0px",
  ]);
  // Tinted, not transparent, and its ink differs from a resting sibling's.
  expect(style.background).not.toBe("rgba(0, 0, 0, 0)");
  const resting = await page
    .locator('.nav button:not([aria-current="page"])')
    .first()
    .evaluate((el) => getComputedStyle(el).color);
  expect(style.color).not.toBe(resting);
});

test("a stage names its object, not itself", async ({ page }) => {
  // The rail said "Datasets & conditions" to get you here; an <h1> repeating it
  // plus a paragraph explaining the stage cost about 140px above the fold and
  // said nothing about the series in view.
  await openStage(page, null);

  const ident = page.locator(".ident");
  await expect(ident).toBeVisible();
  const text = await ident.innerText();
  expect(text).toContain("Series-1");
  expect(text).toMatch(/\d+ frames/);
  // There is still exactly one h1, but it names the SERIES rather than the
  // stage: the page's heading should be its object.
  await expect(page.locator("h1")).toHaveCount(1);
  await expect(page.locator("h1")).toHaveText("Series-1");
  expect(text).not.toContain("Datasets & conditions");
});

test("a stage with nothing to say draws no header", async ({ page }) => {
  // Results has no forward action and does not claim the slot, so the header
  // would otherwise be a rule across an empty row.
  await openStage(page, "Results & validation");
  await expect(page.locator(".stagehead")).toBeHidden();
});

test("the forward action stays at the far end of the header", async ({
  page,
}) => {
  await openStage(page, null);

  const head = (await page.locator(".stagehead").boundingBox())!;
  const button = (await page.locator(".stagehead > .btn").boundingBox())!;
  // Right-aligned whether or not the stage filled the slot beside it.
  expect(button.x + button.width).toBeGreaterThan(head.x + head.width * 0.75);
});

test("the frame ribbon holds every frame and never scrolls", async ({
  page,
}) => {
  // n_frames is data: 12 here, hundreds for a long acquisition. The strip can
  // only ever show a window of that, and what scrolling costs is the shape of
  // the sequence — so the overview stays put while the detail moves.
  await openStage(page, null);

  // The frames arrive from the API, so wait for the ribbon itself rather than
  // for the rail that was already mounted by the previous stage.
  await page.locator(".ribbon-tick").first().waitFor();
  const ticks = page.locator(".ribbon-tick");
  const strip = page.locator(".strip");
  const shown = await strip.locator(".fr").count();
  expect(await ticks.count()).toBeGreaterThan(0);

  const ribbonBox = (await page.locator(".ribbon").boundingBox())!;
  const stripBox = (await strip.boundingBox())!;
  // Never wider than its container, at any frame count.
  expect(ribbonBox.width).toBeLessThanOrEqual(stripBox.width + 1);
  // And it carries frames the strip is not currently showing.
  expect(await ticks.count()).toBeGreaterThanOrEqual(shown);

  // The roles are structural, not colour-only: the summary names them.
  const label = await page.locator(".ribbon").getAttribute("aria-label");
  expect(label).toMatch(/\d+ frames/);
});

test("the QC picker swaps the chart at full width", async ({ page }) => {
  // Two-up was tried and abandoned: the checks have very different natural
  // heights, so one panel became a tall column beside a short box with dead
  // space under it. One chart at a time, full width, chosen by a select.
  await openStage(page, null);
  const picker = page.getByLabel("Preprocessing check");
  await picker.waitFor();

  const svg = page.locator(".qc-sub svg[role='img']");
  await expect(svg).toHaveCount(1);
  const wide = (await svg.boundingBox())!;
  const card = (await page.locator(".qc-sub").boundingBox())!;
  // Full width, not half of it.
  expect(wide.width).toBeGreaterThan(card.width * 0.85);

  await picker.click();
  await page.getByRole("option", { name: /Interface evolution/ }).click();
  await expect(
    page.getByRole("img", { name: /Bubble outline for \d+ frames/ }),
  ).toBeVisible();
  await expect(svg).toHaveCount(1);
});

test("the check chooser is the app's own menu, not the platform's", async ({
  page,
}) => {
  // A native <select> paints the operating system's menu: system font, system
  // metrics, system colours, on neither of our surfaces and in neither theme.
  await openStage(page, null);
  const picker = page.getByLabel("Preprocessing check");
  await picker.waitFor();
  expect(await picker.evaluate((el) => el.tagName)).toBe("BUTTON");
  await expect(page.locator(".qc-sub select")).toHaveCount(0);

  await picker.click();
  const menu = page.locator(".qc-sub .menu");
  await expect(menu).toBeVisible();
  // Drawn on the app's overlay surface, opaque, above the chart.
  const surface = await menu.evaluate((el) =>
    getComputedStyle(el).backgroundColor.replace(/\s/g, ""),
  );
  expect(surface).not.toContain("rgba");
  await expect(menu.getByRole("option")).toHaveCount(2);

  // Openable and takeable without a pointer.
  await page.keyboard.press("Escape");
  await expect(menu).toBeHidden();
  await picker.press("ArrowDown");
  await expect(menu).toBeVisible();
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Enter");
  await expect(
    page.getByRole("img", { name: /Bubble outline for \d+ frames/ }),
  ).toBeVisible();
  await expect(picker).toBeFocused();
});

test("the QC chart is drawn at the width it is rendered at", async ({
  page,
}) => {
  // A viewBox scales its type with everything else, so a coordinate system
  // sized for the wrong width sets its labels at the wrong size.
  await openStage(page, null);
  const svg = page.locator(".qc-sub svg[role='img']").first();
  await svg.waitFor();

  const box = (await svg.boundingBox())!;
  const viewBox = (await svg.getAttribute("viewBox"))!.split(" ").map(Number);
  expect(box.width / viewBox[2]).toBeGreaterThan(0.6);
});

test("the QC card carries one strip of controls, not two", async ({ page }) => {
  // The picker used to sit in a header row while four export buttons sat on
  // their own row underneath, leaving a ragged two-row header with a gap in it.
  await openStage(page, null);
  const picker = page.getByLabel("Preprocessing check");
  await picker.waitFor();

  const pick = (await picker.boundingBox())!;
  const download = (await page.getByLabel(/^Download /).boundingBox())!;
  const finding = (await page.locator(".qc-finding").boundingBox())!;
  const centre = (b: { y: number; height: number }) => b.y + b.height / 2;

  // One row: both menus and the reading share it.
  expect(Math.abs(centre(pick) - centre(download))).toBeLessThan(4);
  expect(Math.abs(centre(pick) - centre(finding))).toBeLessThan(8);
  // The menus cluster at the left, the reading closes the row at the right.
  expect(pick.x).toBeLessThan(download.x);
  expect(download.x + download.width).toBeLessThan(finding.x);

  // No heading above it: the chooser names the chart, the axes state the rest.
  await expect(page.locator(".qc-sub h3")).toHaveCount(0);
});

test("expand rides the plot and opens it larger", async ({ page }) => {
  await openStage(page, null);
  await page.getByLabel("Preprocessing check").waitFor();

  const expand = page.getByLabel(/^Expand /);
  // On the picture, not in the toolbar row above it.
  const plot = (await page.locator(".qc-sub .cf-plot").boundingBox())!;
  const box = (await expand.boundingBox())!;
  expect(box.y).toBeGreaterThan(plot.y);

  await expand.click();
  await expect(page.getByRole("dialog", { name: /kinematics/i })).toBeVisible();
});

test("the QC chart reads out the frame under the pointer", async ({ page }) => {
  await openStage(page, null);
  const svg = page.locator(".qc-sub svg[role='img']").first();
  await svg.waitFor();
  const box = (await svg.boundingBox())!;

  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.35);

  const tip = page.locator(".qc-sub .chart-tip");
  await expect(tip).toBeVisible();
  // Measured, fit and the gap between them: the check's whole question.
  await expect(tip).toContainText("measured");
  await expect(tip).toContainText("fit");
  await expect(tip).toContainText("residual");
  // The same frame is marked in the fit and in the residual strip below it.
  const dot = page.locator(".qc-sub .qc-dot.hot");
  await expect(dot).toHaveCount(1);
  await expect(page.locator(".qc-sub .qc-resid-dot.hot")).toHaveCount(1);

  // The cursor snaps to that frame rather than tracking the raw pointer, so
  // the number in the readout is the one the line is standing on. (A zero-area
  // <line> never satisfies toBeVisible, hence the geometric assertion.)
  const cursor = page.locator(".qc-sub .chart-cursor");
  await expect(cursor).not.toHaveCSS("display", "none");
  expect(await cursor.getAttribute("x1")).toBe(await dot.getAttribute("cx"));
});

test("the QC chart can be read without a pointer", async ({ page }) => {
  // A crosshair is a mouse affordance and says nothing to anyone without one.
  await openStage(page, null);
  const svg = page.locator(".qc-sub svg[role='img']").first();
  await svg.waitFor();

  await svg.focus();
  await expect(page.locator(".qc-sub .chart-tip")).toBeVisible();
  const live = page.locator(".qc-sub [role='status']");
  await expect(live).toContainText("frame 1");

  await page.keyboard.press("ArrowRight");
  await expect(live).toContainText("frame 2");

  await page.keyboard.press("Escape");
  await expect(page.locator(".qc-sub .chart-tip")).toBeHidden();
});
