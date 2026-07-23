import { defineConfig, devices } from "@playwright/test";

/**
 * E2E config. Starts the real API (reading the real outputs/) and the Vite dev
 * server, so the round-trip test exercises every layer for real.
 *
 * Requires a one-time `npx playwright install chromium`.
 */
export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 30_000,
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      // API — run from the repo root so it finds outputs/ and the venv package.
      command: "../../.venv/bin/python -m naviernet_api",
      cwd: "../..",
      env: { NAVIERNET_API_PORT: "8000" },
      url: "http://127.0.0.1:8000/healthz",
      reuseExistingServer: true,
      timeout: 30_000,
    },
    {
      command: "npm run dev",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: true,
      timeout: 30_000,
    },
  ],
});
