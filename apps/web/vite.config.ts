/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The API defaults to :8000; proxy /api so the browser talks to one origin in
// dev. scripts/run.sh injects NAVIERNET_API_PORT when it reallocates ports.
const apiPort = process.env.NAVIERNET_API_PORT ?? "8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": `http://127.0.0.1:${apiPort}`,
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    css: true,
    include: ["tests/**/*.test.{ts,tsx}"],
  },
});
