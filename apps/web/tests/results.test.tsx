import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ResultsView } from "../src/views/ResultsView";

const RUN = {
  id: "highest_t",
  dataset: "highest_t",
  status: "trained",
  steps: 1530,
  iou_holdout: 0.968,
};

const DETAIL = {
  id: "highest_t",
  dataset: "highest_t",
  status: "trained",
  steps: 1530,
  metrics: {
    iou_per_frame: { "1": 0.973, "6": 0.968, "10": 0.921 },
    iou_mean: 0.962,
    iou_holdout: 0.968,
    holdout_frame: 6,
    nose_speed_mm_s: 177.3,
  },
  config: { experiment: { dt_frame_ms: 0.5 } },
  artifacts: { checkpoint: true, metrics: true, groups: true, video: false, figures: [] },
};

const VALIDATION = {
  nose_speed_inferred_mm_s: 177.3,
  nose_speed_measured_mm_s: 180.0,
  nose_speed_error_pct: 1.5,
  bretherton_film_um: 4.875,
  hele_shaw: 0.2228,
  reynolds: 215.5,
  weber: 2.302,
  capillary: 0.01068,
  prandtl: 9.411,
  iou_mean: 0.962,
  iou_holdout: 0.968,
  holdout_frame: 6,
};

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200 });
}

function mockApi() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL) => {
      const u = String(url);
      if (u.endsWith("/api/runs")) return json([RUN]);
      if (u.endsWith("/validation")) return json(VALIDATION);
      if (/\/api\/runs\/[^/]+$/.test(u)) return json(DETAIL);
      return new Response("not found", { status: 404 });
    }),
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("ResultsView", () => {
  it("shows agreement per frame with the holdout called out", async () => {
    mockApi();
    render(<ResultsView />);

    expect(await screen.findByText("highest_t", { selector: ".id" })).toBeInTheDocument();
    // Per-frame IoU values render in the table.
    expect(await screen.findByText("0.968")).toBeInTheDocument();
    expect(screen.getByText("0.973")).toBeInTheDocument();
    expect(screen.getByText(/holdout — never supervised/)).toBeInTheDocument();
  });

  it("shows physics validation: inferred vs measured nose speed and groups", async () => {
    mockApi();
    render(<ResultsView />);

    expect(await screen.findByText("177")).toBeInTheDocument(); // inferred, mm/s
    expect(screen.getByText("180")).toBeInTheDocument(); // measured
    expect(screen.getByText(/1\.5% error/)).toBeInTheDocument();
    expect(screen.getByText("215.5")).toBeInTheDocument(); // Reynolds
  });

  it("shows deliverables whose download links point at the run's artifacts", async () => {
    mockApi();
    render(<ResultsView />);

    expect(await screen.findByText("training_data.npz")).toBeInTheDocument();
    expect(screen.getByText("ckpt.pt")).toBeInTheDocument();

    const links = screen.getAllByRole("link", { name: "Download" });
    const hrefs = links.map((a) => a.getAttribute("href"));
    expect(hrefs).toContain("/api/runs/highest_t/tensors");
    expect(hrefs).toContain("/api/runs/highest_t/checkpoint");
  });

  it("shows an empty state when there are no runs", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json([])));
    render(<ResultsView />);
    expect(await screen.findByText(/No runs yet/)).toBeInTheDocument();
  });

  it("shows an error state when the run list fails to load", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("boom", { status: 500, statusText: "Server Error" })),
    );
    render(<ResultsView />);
    expect(await screen.findByText(/Could not load runs/)).toBeInTheDocument();
  });

  it("shows an error state when the selected run's detail fails to load", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/api/runs")) return json([RUN]);
        return new Response("boom", { status: 500 }); // detail + validation fail
      }),
    );
    render(<ResultsView />);
    expect(await screen.findByText(/Could not load results/)).toBeInTheDocument();
  });
});
