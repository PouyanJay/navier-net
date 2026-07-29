import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ChartFrame } from "../src/components/ChartFrame";
import { ToastProvider } from "../src/components/Toast";
import { rowsToCsv } from "../src/lib/chartExport";

afterEach(() => vi.restoreAllMocks());

const ROWS = [
  { series: "pinn", t_ms: 0, nose_um: 12.5 },
  { series: "measured", t_ms: 0.5, nose_um: 13.1 },
];

function renderFrame() {
  return render(
    <ToastProvider>
      <ChartFrame
        name="test-chart"
        title="Test chart"
        rows={ROWS}
        render={(expanded) => (
          <svg viewBox="0 0 10 10" role="img" aria-label="tiny">
            <text>{expanded ? "big" : "small"}</text>
          </svg>
        )}
      />
    </ToastProvider>,
  );
}

describe("rowsToCsv", () => {
  it("writes a header from the union of keys and quotes as needed", () => {
    const csv = rowsToCsv([
      { a: 1, b: 'say "hi"' },
      { a: 2, c: "x,y" },
    ]);
    expect(csv.split("\n")[0]).toBe("a,b,c");
    expect(csv).toContain('"say ""hi"""');
    expect(csv).toContain('"x,y"');
  });
});

describe("ChartFrame", () => {
  it("offers the standard chart actions", () => {
    renderFrame();
    for (const action of ["Expand", "PNG", "SVG", "CSV", "JSON"]) {
      expect(screen.getByRole("button", { name: action })).toBeInTheDocument();
    }
  });

  it("expands the chart into a modal with a live re-render", async () => {
    renderFrame();
    fireEvent.click(screen.getByRole("button", { name: "Expand" }));
    const dialog = await screen.findByRole("dialog", { name: "Test chart" });
    expect(dialog).toHaveTextContent("big");
    fireEvent.keyDown(dialog, { key: "Escape" });
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
  });

  it("downloads the charted data as CSV with the real values", async () => {
    const captured: Blob[] = [];
    vi.spyOn(URL, "createObjectURL").mockImplementation((blob) => {
      captured.push(blob as Blob);
      return "blob:test";
    });
    renderFrame();
    fireEvent.click(screen.getByRole("button", { name: "CSV" }));
    await waitFor(() => expect(captured.length).toBe(1));
    // jsdom blobs predate Blob#text; FileReader is the portable read.
    const text = await new Promise<string>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.readAsText(captured[0]);
    });
    expect(text.split("\n")[0]).toBe("series,t_ms,nose_um");
    expect(text).toContain("measured,0.5,13.1");
  });
});
