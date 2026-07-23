import { existsSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test } from "@playwright/test";

/**
 * The Phase 4 journey, for real: configure a short run in the Solver, launch
 * it, watch the console and loss chart update live over SSE, and find the
 * finished run in Results — every layer (browser → API → run_manager → the
 * actual PINN trainer → outputs/) exercised with no mocks.
 *
 * The run is tiny (25 steps, small batches, no rendering) but real, so it
 * lands in the repo's outputs/; the spec removes exactly that run afterwards.
 */

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
const RUN_ID_SHAPE = /^run-\d{8}-\d{6}(-\d+)?$/;

let launchedRunId: string | null = null;

test.afterAll(() => {
  // Delete only the run this spec created, and only if it looks like one.
  if (launchedRunId && RUN_ID_SHAPE.test(launchedRunId)) {
    const runDir = resolve(REPO_ROOT, "outputs", launchedRunId);
    if (existsSync(runDir)) rmSync(runDir, { recursive: true });
  }
});

test("configure, launch, watch live, and find the run in Results", async ({ page }) => {
  test.setTimeout(180_000); // a real (if tiny) training + evaluation run

  await page.goto("/");
  await page.getByRole("button", { name: "Solver", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Solver", exact: true })).toBeVisible();

  // A tiny but real configuration; every field maps onto cfg.training.
  await page.getByLabel("Steps").fill("25");
  await page.getByLabel("Data batch").fill("64");
  await page.getByLabel("Collocation batch").fill("64");
  await page.getByLabel("Boundary batch").fill("16");
  await page.getByLabel(/Log every/).fill("10");
  await page.getByRole("switch", { name: "Render deliverables" }).click();

  await page.getByRole("button", { name: "Run", exact: true }).click();

  // The run is live: id assigned, pill on, console streaming, chart drawing.
  const runId = (await page.locator(".solver-head .id").textContent({ timeout: 15_000 }))!;
  expect(runId).toMatch(RUN_ID_SHAPE);
  launchedRunId = runId;
  await expect(page.getByText(`[naviernet] starting run ${runId}`, { exact: false })).toBeVisible();
  await expect(page.locator(".runpill")).toBeVisible();
  await expect(page.getByText(/training steps 1-25/)).toBeVisible({ timeout: 30_000 });
  await expect(page.locator("path.chart-line")).toHaveCount(3, { timeout: 60_000 });

  // It finishes: done state, full progress, transcript milestone, pill off.
  await expect(page.locator(".solver-head .dot")).toHaveText("done", { timeout: 120_000 });
  await expect(page.getByText("[naviernet] run complete", { exact: false })).toBeVisible();
  await expect(page.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "25");
  await expect(page.locator(".runpill")).toHaveCount(0);

  // The finished run is a first-class run in Results.
  await page.getByRole("button", { name: "Results & validation" }).click();
  await page.locator(".results-head select").selectOption(runId);
  await expect(page.locator(".results-head .id")).toHaveText(runId);
  await expect(page.getByText("Agreement per frame")).toBeVisible();
});
