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
  }
  await page.locator(".stage-aside").waitFor();
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

  const opacity = await page.evaluate(() => {
    const rail = document.querySelector(".stage-aside")!;
    const panel = getComputedStyle(rail).getPropertyValue("--panel").trim();
    const probe = document.createElement("div");
    probe.style.backgroundColor = panel;
    document.body.append(probe);
    const resolved = getComputedStyle(probe).backgroundColor;
    probe.remove();
    return resolved;
  });
  // rgb(...) is opaque; rgba(..., a<1) is not.
  expect(opacity).not.toMatch(/rgba\([^)]*,\s*0?\.\d+\s*\)/);
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
