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
