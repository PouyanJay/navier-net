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

// Node >= 22 ships an experimental `localStorage` global that shadows jsdom's
// (its methods throw off the main thread); tests that mount the AppShell read
// the persisted theme, so give them a real in-memory Storage.
const themeStore = new Map<string, string>();
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: {
    getItem: (key: string) => themeStore.get(key) ?? null,
    setItem: (key: string, value: string) => {
      themeStore.set(key, String(value));
    },
    removeItem: (key: string) => {
      themeStore.delete(key);
    },
    clear: () => themeStore.clear(),
    key: (index: number) => [...themeStore.keys()][index] ?? null,
    get length() {
      return themeStore.size;
    },
  },
});

// jsdom has no matchMedia; the theme bootstrap asks for the OS color scheme.
globalThis.matchMedia = ((query: string) => ({
  matches: false,
  media: query,
  onchange: null,
  addEventListener() {},
  removeEventListener() {},
  addListener() {},
  removeListener() {},
  dispatchEvent: () => false,
})) as never;
