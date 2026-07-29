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

function detailOf(run: (typeof RUNS)[number]) {
  return {
    id: run.id,
    dataset: run.dataset,
    status: run.status === "trained" ? "trained" : "empty",
    steps: run.steps,
    metrics: {
      iou_holdout: run.iou_holdout,
      val_iou_mean: run.val_iou_mean,
      heldout_datasets: run.heldout_datasets,
      datasets: run.datasets,
    },
    config: {
      dataset: run.dataset,
      datasets: run.datasets,
      training: {
        steps: run.steps ?? 1500,
        seed: 1234,
        val_fraction: 0.2,
        val_strategy: "tail",
        device: "cpu",
      },
    },
    artifacts: {
      checkpoint: run.status === "trained",
      metrics: true,
      groups: true,
      video: false,
      figures: [],
    },
  };
}

/** The endpoints the shell + results skeleton touch; everything else 404s. */
function mockApi(): { runListCalls: string[]; launches: unknown[] } {
  const calls = { runListCalls: [] as string[], launches: [] as unknown[] };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL, opts?: RequestInit) => {
      const u = String(url);
      if (u.includes("/api/projects/" + PID)) return json(PROJECT);
      if (u.includes("/api/projects")) return json([PROJECT]);
      if (u.includes("/api/datasets")) return json(DATASETS);
      if (u.includes("/api/runs/active")) return json(null);
      if (u.endsWith("/api/runs") && opts?.method === "POST") {
        calls.launches.push(JSON.parse(String(opts.body)));
        return new Response(
          JSON.stringify({
            run_id: "demo_run",
            dataset: "highest_t",
            state: "queued",
          }),
          { status: 202 },
        );
      }
      const validation = u.match(/\/api\/runs\/([^/?]+)\/validation$/);
      if (validation) {
        const run = RUNS.find((r) => r.id === validation[1]);
        if (!run) return new Response("not found", { status: 404 });
        const joint = run.id === "second_run";
        return json({
          nose_speed_inferred_mm_s: 177.3,
          nose_speed_measured_mm_s: joint ? null : 180.0,
          nose_speed_error_pct: joint ? null : 1.5,
          bretherton_film_um: 4.9,
          hele_shaw: 1.9,
          reynolds: 320,
          weber: 1.8,
          capillary: 0.0056,
          prandtl: 9.4,
          iou_mean: joint ? 0.953 : 0.962,
          iou_holdout: run.iou_holdout,
          holdout_frame: run.iou_holdout != null ? 6 : null,
          val_iou_mean: run.val_iou_mean,
          iou_val: null,
          validation_frames: [],
          transfer_iou_mean: joint ? 0.903 : null,
          transfer_per_dataset: joint ? { low_t: 0.903 } : null,
          per_dataset: joint
            ? {
                highest_t: {
                  iou_mean: 0.958,
                  iou_val: 0.941,
                  validation_frames: [8, 9, 10],
                  iou_per_frame: { "0": 0.96, "8": 0.94 },
                },
              }
            : null,
          training_datasets: joint ? ["highest_t"] : null,
          heldout_datasets: joint ? ["low_t"] : null,
        });
      }
      const detail = u.match(/\/api\/runs\/([^/?]+)$/);
      if (detail) {
        const run = RUNS.find((r) => r.id === detail[1]);
        return run
          ? json(detailOf(run))
          : new Response("not found", { status: 404 });
      }
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

  it("overview leads with the two-axis generalization scorecard", async () => {
    mockApi();
    renderAt(`/projects/${PID}/results/second_run/overview`);

    const panel = await screen.findByTestId("overview-scorecard");
    // Axis A — in-distribution validation IoU of the trained conditions
    // (waits: the validation fetch resolves after first paint).
    await waitFor(() => expect(panel).toHaveTextContent("0.941"));
    expect(panel).toHaveTextContent(/in-distribution/i);
    // Axis B — transfer to the held-out condition.
    expect(panel).toHaveTextContent("0.903");
    expect(panel).toHaveTextContent(/transfer/i);

    // The verdict narrative states what the numbers argue.
    expect(
      screen.getByText(/learned physics, not footage/i),
    ).toBeInTheDocument();

    // At-a-glance chips jump to their tab.
    fireEvent.click(screen.getByRole("button", { name: /field maps/i }));
    await waitFor(() =>
      expect(screen.getByRole("tab", { name: /fields/i })).toHaveAttribute(
        "aria-selected",
        "true",
      ),
    );
  });

  it("shows the selected run's header: status, pedigree, config, resume", async () => {
    const calls = mockApi();
    renderAt(`/projects/${PID}/results/second_run`);

    // Identity + condition chips: training conditions plus the held-out one.
    const header = await screen.findByTestId("run-header");
    expect(header).toHaveTextContent("second_run");
    expect(header).toHaveTextContent(/trained/i);
    expect(header).toHaveTextContent(/held out/i);

    // Pedigree comes from the config snapshot the run actually recorded
    // (waits: the detail fetch resolves after first paint).
    await waitFor(() => expect(header).toHaveTextContent("1234")); // seed
    expect(header).toHaveTextContent(/20\s?% · tail/); // val split

    // The reproducibility affordance: the resolved config, openable in place.
    fireEvent.click(screen.getByText(/config snapshot/i));
    expect(await screen.findByText(/val_fraction/)).toBeInTheDocument();

    // A joint run gets a viewing-condition selector for per-condition panels.
    const selector = screen.getByLabelText(/viewing condition/i);
    expect(selector).toBeInTheDocument();

    // Resume posts a real resume request for this run.
    fireEvent.click(screen.getByRole("button", { name: /resume training/i }));
    await waitFor(() => expect(calls.launches.length).toBe(1));
    expect(calls.launches[0]).toMatchObject({
      resume: true,
      run_id: "second_run",
    });
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
