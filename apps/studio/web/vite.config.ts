/// <reference types="vitest/config" />
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

const STUDIO_API = process.env.NAVIERNET_STUDIO_API ?? "http://127.0.0.1:8010";

export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      "/api": { target: STUDIO_API, changeOrigin: true },
      "/healthz": { target: STUDIO_API, changeOrigin: true },
    },
  },
  build: { target: "es2022", sourcemap: true },
  test: {
    environment: "jsdom",
    include: ["tests/**/*.test.ts"],
    setupFiles: ["./tests/setup.ts"],
  },
});
