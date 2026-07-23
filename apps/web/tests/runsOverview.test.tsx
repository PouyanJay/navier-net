import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RunSummary } from "../src/lib/api";
import { RunsOverview } from "../src/views/RunsOverview";

function mockRuns(runs: RunSummary[]) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(runs), { status: 200 })),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("RunsOverview (walking skeleton)", () => {
  it("renders a run returned by the API with its holdout IoU", async () => {
    mockRuns([
      { id: "highest_t", dataset: "highest_t", status: "trained", steps: 1530, iou_holdout: 0.968 },
    ]);

    render(<RunsOverview />);

    // "highest_t" is both the run id and the dataset chip, so scope to the id.
    expect(await screen.findByText("highest_t", { selector: ".id" })).toBeInTheDocument();
    expect(screen.getByText("0.968")).toBeInTheDocument();
    expect(screen.getByText("trained")).toBeInTheDocument();
  });

  it("shows an empty state when there are no runs", async () => {
    mockRuns([]);
    render(<RunsOverview />);
    expect(await screen.findByText(/No runs yet/)).toBeInTheDocument();
  });

  it("shows an error state when the API is unreachable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("boom", { status: 500, statusText: "Server Error" })),
    );
    render(<RunsOverview />);
    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
  });
});
