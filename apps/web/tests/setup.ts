import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

// jsdom has no canvas; a silent stub keeps chart components renderable in
// tests without "Not implemented" noise hiding real regressions.
HTMLCanvasElement.prototype.getContext = vi.fn(() => null) as never;

// jsdom has no ResizeObserver either (every target browser does). Components
// that measure themselves, such as the frame strip's scrollbar, need it to
// exist; jsdom reports zero sizes anyway, so a no-op is the honest stub.
globalThis.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as never;
