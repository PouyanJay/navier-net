import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

// jsdom has no canvas; a silent stub keeps chart components renderable in
// tests without "Not implemented" noise hiding real regressions.
HTMLCanvasElement.prototype.getContext = vi.fn(() => null) as never;
