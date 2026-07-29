import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "../src/App";
import { ToastProvider } from "../src/components/Toast";

const PID = "a".repeat(32);

const PROJECT = {
  id: PID,
  name: "Microchannel FC-72",
  description: "",
  datasets: ["highest_t"],
  created_at: "2026-07-24T00:00:00+00:00",
};

const RUNS = [
  {
    id: "demo_run",
    dataset: "highest_t",
    status: "trained",
    steps: 1500,
    iou_holdout: 0.968,
    datasets: ["highest_t"],
    heldout_datasets: [],
    val_iou_mean: null,
    date: "2026-07-26T09:14:02+00:00",
  },
  {
    id: "second_run",
    dataset: "highest_t",
    status: "trained",
    steps: 900,
    iou_holdout: null,
    datasets: ["highest_t", "low_t"],
    heldout_datasets: ["low_t"],
    val_iou_mean: 0.941,
    date: "2026-07-27T10:00:00+00:00",
  },
  {
    id: "live_run",
    dataset: "highest_t",
    status: "running",
    steps: null,
    iou_holdout: null,
    datasets: ["highest_t"],
    heldout_datasets: [],
    val_iou_mean: null,
    date: "2026-07-28T08:11:00+00:00",
    steps_done: 40,
    steps_total: 200,
  },
  {
    id: "dead_run",
    dataset: "highest_t",
    status: "failed",
    steps: null,
    iou_holdout: null,
    datasets: ["highest_t"],
    heldout_datasets: [],
    val_iou_mean: null,
    date: "2026-07-24T23:31:00+00:00",
  },
];

const DATASETS = [
  { id: "highest_t", label: "highest_t", frames: 11, processed: true },
];

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200 });
}

/** The endpoints the shell + results skeleton touch; everything else 404s. */
function mockApi(): { runListCalls: string[] } {
  const calls = { runListCalls: [] as string[] };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL) => {
      const u = String(url);
      if (u.includes("/api/projects/" + PID)) return json(PROJECT);
      if (u.includes("/api/projects")) return json([PROJECT]);
      if (u.includes("/api/datasets")) return json(DATASETS);
      if (u.includes("/api/runs/active")) return json(null);
      if (u.includes("/api/runs")) {
        calls.runListCalls.push(u);
        return json(RUNS);
      }
      return new Response("not found", { status: 404 });
    }),
  );
  return calls;
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <ToastProvider>
        <App />
      </ToastProvider>
    </MemoryRouter>,
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("results routing", () => {
  it("renders the project-scoped results page with run browser and tabs", async () => {
    const calls = mockApi();
    renderAt(`/projects/${PID}/results`);

    // Page identity + run browser rows from the project-scoped listing.
    expect(
      await screen.findByRole("heading", { name: /results & validation/i }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("option", { name: /demo_run/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: /second_run/i }),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(calls.runListCalls.some((u) => u.includes(`project=${PID}`))).toBe(
        true,
      ),
    );

    // Every run state renders its own row shape: a joint run leads with its
    // axis-A validation IoU, a live run with progress, a failed run says so.
    const jointRow = screen.getByRole("option", { name: /second_run/i });
    expect(jointRow).toHaveTextContent("0.941");
    expect(jointRow).toHaveTextContent(/val iou/i);
    expect(jointRow).toHaveTextContent(/1 held out/i);
    expect(screen.getByRole("option", { name: /live_run/i })).toHaveTextContent(
      "20%",
    );
    expect(screen.getByRole("option", { name: /dead_run/i })).toHaveTextContent(
      /failed/i,
    );

    // The eight output tabs exist.
    const tablist = screen.getByRole("tablist", { name: /run outputs/i });
    for (const tab of ["Overview", "Reconstruction", "Fields", "Compare"]) {
      expect(
        screen.getAllByRole("tab", { name: new RegExp(tab, "i") })[0],
      ).toBeInTheDocument();
    }
    expect(tablist).toBeInTheDocument();
  });

  it("deep-links a run and tab from the URL", async () => {
    mockApi();
    renderAt(`/projects/${PID}/results/second_run/fields`);

    const row = await screen.findByRole("option", { name: /second_run/i });
    expect(row).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: /fields/i })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("navigates run selection and tab clicks into the URL", async () => {
    mockApi();
    renderAt(`/projects/${PID}/results`);

    // Default: first trained run selected, Overview tab active.
    const first = await screen.findByRole("option", { name: /demo_run/i });
    expect(first).toHaveAttribute("aria-selected", "true");

    fireEvent.click(screen.getByRole("option", { name: /second_run/i }));
    await waitFor(() =>
      expect(
        screen.getByRole("option", { name: /second_run/i }),
      ).toHaveAttribute("aria-selected", "true"),
    );

    fireEvent.click(screen.getByRole("tab", { name: /reconstruction/i }));
    await waitFor(() =>
      expect(
        screen.getByRole("tab", { name: /reconstruction/i }),
      ).toHaveAttribute("aria-selected", "true"),
    );
  });
});
