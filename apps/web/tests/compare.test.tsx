/** Run comparison: selection, overlaid interactive charts, metrics table. */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ComparePanel } from "../src/views/results/ComparePanel";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

const CANDIDATES = [
  {
    id: "sweep-1-s0",
    dataset: "highest_t",
    status: "trained",
    steps: 40,
    iou_holdout: 0.62,
  },
  {
    id: "sweep-1-s1",
    dataset: "highest_t",
    status: "trained",
    steps: 40,
    iou_holdout: 0.64,
  },
  {
    id: "highest_t",
    dataset: "highest_t",
    status: "trained",
    steps: 1500,
    iou_holdout: 0.968,
  },
] as const;

const HISTORY = [
  { step: 10, lr: 2e-3, data: 0.5, vof: 0.04, div: 0.01, src: 1e-3, bc: 0.02 },
  {
    step: 20,
    lr: 2e-3,
    data: 0.3,
    vof: 0.03,
    div: 0.008,
    src: 9e-4,
    bc: 0.015,
  },
];

function detailFor(id: string) {
  return {
    id,
    dataset: "highest_t",
    status: "trained",
    steps: 40,
    metrics: {
      iou_per_frame: { "1": 0.9, "2": 0.91 },
      iou_mean: 0.905,
      iou_holdout: 0.89,
      nose_speed_mm_s: 170,
    },
    config: null,
    artifacts: {
      checkpoint: true,
      metrics: true,
      groups: false,
      video: false,
      figures: [],
    },
  };
}

function stubApi() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: RequestInfo | URL) => {
      const path = String(url);
      const lossMatch = path.match(/\/api\/runs\/([^/]+)\/loss-history$/);
      if (lossMatch) return json(HISTORY);
      const detailMatch = path.match(/\/api\/runs\/([^/]+)$/);
      if (detailMatch)
        return json(detailFor(decodeURIComponent(detailMatch[1])));
      return json({ detail: "not found" }, 404);
    }),
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("ComparePanel", () => {
  it("asks for a second run before comparing", () => {
    stubApi();
    render(<ComparePanel candidates={[...CANDIDATES]} />);
    fireEvent.click(screen.getByRole("button", { name: "sweep-1-s0" }));
    expect(
      screen.getByText("Select at least two runs to compare them."),
    ).toBeInTheDocument();
  });

  it("renders overlaid loss + IoU charts and the metrics table for a selection", async () => {
    stubApi();
    const { container } = render(<ComparePanel candidates={[...CANDIDATES]} />);
    fireEvent.click(screen.getByRole("button", { name: "sweep-1-s0" }));
    fireEvent.click(screen.getByRole("button", { name: "sweep-1-s1" }));

    await waitFor(() =>
      expect(
        container.querySelectorAll("path.chart-line.series-0"),
      ).toHaveLength(2),
    );
    // Two charts (loss + IoU), each with one line per selected run.
    expect(container.querySelectorAll("path.chart-line")).toHaveLength(4);
    // Metrics table: one row per run with the headline numbers.
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getAllByText("0.905")).toHaveLength(2);
    // Selected chips are marked pressed with their series index.
    const pressed = container.querySelectorAll(
      '.compare-choice[aria-pressed="true"]',
    );
    expect(pressed).toHaveLength(2);
  });

  it("caps the selection at four runs", () => {
    stubApi();
    render(
      <ComparePanel
        candidates={[
          ...CANDIDATES,
          {
            id: "r4",
            dataset: null,
            status: "trained",
            steps: 1,
            iou_holdout: null,
          },
          {
            id: "r5",
            dataset: null,
            status: "trained",
            steps: 1,
            iou_holdout: null,
          },
        ]}
      />,
    );
    for (const id of ["sweep-1-s0", "sweep-1-s1", "highest_t", "r4", "r5"]) {
      fireEvent.click(screen.getByRole("button", { name: id }));
    }
    const pressed = document.querySelectorAll(
      '.compare-choice[aria-pressed="true"]',
    );
    expect(pressed).toHaveLength(4);
  });
});
