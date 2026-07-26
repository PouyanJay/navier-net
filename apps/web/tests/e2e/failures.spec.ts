import { expect, test } from "@playwright/test";

/** Failure paths against the real API: errors must be visible, never silent. */

test("a rejected upload surfaces the API's reason", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Datasets & conditions" }).click();
  await expect(
    page.getByRole("heading", { name: "Operating conditions" }),
  ).toBeVisible();

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

test("a failed reconstruction fetch surfaces an error, not a blank panel", async ({
  page,
}) => {
  await page.route("**/api/runs/*/interface*", (route) =>
    route.fulfill({
      status: 500,
      contentType: "application/json",
      body: JSON.stringify({ detail: "reconstruction backend unavailable" }),
    }),
  );
  await page.goto("/");
  const alert = page
    .getByRole("alert")
    .filter({ hasText: "Could not load the reconstruction" });
  await expect(alert).toBeVisible();
  await expect(alert).toContainText("reconstruction backend unavailable");
  // The rest of Results still renders; one failed panel never blanks the view.
  await expect(page.getByText("Agreement per frame")).toBeVisible();
});
