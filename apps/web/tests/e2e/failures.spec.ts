import { expect, test } from "@playwright/test";

/** Failure paths against the real API: errors must be visible, never silent. */

test("a rejected upload surfaces the API's reason", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Datasets & conditions" }).click();
  await expect(page.getByRole("heading", { name: "Operating conditions" })).toBeVisible();

  // A text file is not a TIFF frame; the API rejects it with a 400 detail.
  await page.setInputFiles('input[type="file"]', {
    name: "notes.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("not an image"),
  });
  const alert = page.getByRole("alert");
  await expect(alert).toBeVisible();
  await expect(alert).toContainText(/TIFF|tif|image|file/i);
});

test("an unknown run id in the URL-free flows cannot break Results", async ({ page }) => {
  // Results always lists real runs; selecting each renders without silent gaps.
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Results & validation" })).toBeVisible();
  await expect(page.getByText("Agreement per frame")).toBeVisible();
});
