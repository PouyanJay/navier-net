import { expect, test } from "@playwright/test";

/**
 * Walking-skeleton round trip: the browser loads the app, the app calls the real
 * API, and the API's real run (from outputs/) appears on the page. This proves
 * every layer is wired together.
 */
test("renders the shell and a real run from the API", async ({ page }) => {
  await page.goto("/");

  // Shell
  await expect(page.getByText("NavierNet")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Results & validation" })).toBeVisible();

  // A real trained run, read end-to-end from outputs/ via the API.
  await expect(page.getByText("highest_t").first()).toBeVisible();
  await expect(page.getByText("trained").first()).toBeVisible();

  // Phase 1 Results content, all read live from the pipeline's artifacts.
  await expect(page.getByText("Agreement per frame")).toBeVisible();
  await expect(page.getByText(/holdout — never supervised/)).toBeVisible();
  await expect(page.getByText("Physics validation")).toBeVisible();
  await expect(page.getByText("177").first()).toBeVisible(); // inferred nose speed
  await expect(page.getByText("215.5")).toBeVisible(); // Reynolds
  await expect(page.getByText("training_data.npz")).toBeVisible();
});

test("datasets view shows operating conditions and live dimensionless groups", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Datasets & conditions" }).click();

  // Target panel headings (the page intro paragraph also names these sections).
  await expect(page.getByRole("heading", { name: "Operating conditions" })).toBeVisible();
  await expect(page.getByText("FC-72")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Derived dimensionless groups" }),
  ).toBeVisible();
  await expect(page.getByText("215.5")).toBeVisible(); // live Reynolds
  await expect(
    page.getByRole("heading", { name: "Calibration & segmentation" }),
  ).toBeVisible();
});

test("physics & model view shows equations and the live topology", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Physics & model" }).click();

  await expect(page.getByRole("heading", { name: "Governing equations" })).toBeVisible();
  await expect(page.locator(".katex").first()).toBeVisible(); // KaTeX rendered
  await expect(page.getByRole("heading", { name: "Model topology — live" })).toBeVisible();
  await expect(page.getByText("phi, u, v, s", { exact: true })).toBeVisible(); // architecture fields
});

test("theme toggle flips the document theme", async ({ page }) => {
  await page.goto("/");
  const html = page.locator("html");
  const before = await html.getAttribute("data-theme");

  await page.getByRole("button", { name: /Switch to (light|dark) theme/ }).click();

  const after = await html.getAttribute("data-theme");
  expect(after).not.toBe(before);
});
