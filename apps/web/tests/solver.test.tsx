/** Solver view: config form → launch → live SSE console/chart → terminal states. */

import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Console } from "../src/components/Console";
import { LossChart } from "../src/components/charts/LossChart";
import { Meter } from "../src/components/Meter";
import { Switch } from "../src/components/Switch";
import { SolverView } from "../src/views/SolverView";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const DATASETS = [{ id: "highest_t", n_frames: 11, processed: true }];
const TRAINED_RUNS = [
  { id: "highest_t", dataset: "highest_t", status: "trained", steps: 1500, iou_holdout: 0.968 },
];
const LAUNCHED = {
  run_id: "run-test",
  dataset: "highest_t",
  state: "running",
  stage: "train",
  message: null,
  steps_done: 0,
  steps_total: 40,
};

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  url: string;
  listeners: Record<string, ((event: MessageEvent) => void)[]> = {};
  onerror: (() => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(name: string, callback: (event: MessageEvent) => void) {
    (this.listeners[name] ??= []).push(callback);
  }

  close() {
    this.closed = true;
  }

  emit(name: string, data: unknown) {
    for (const callback of this.listeners[name] ?? []) {
      callback({ data: JSON.stringify(data) } as MessageEvent);
    }
  }
}

function stubApi({ launchStatus = 202 }: { launchStatus?: number } = {}) {
  const posts: unknown[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: RequestInfo | URL, options?: RequestInit) => {
      const path = String(url);
      if (options?.method === "POST" && path.endsWith("/api/runs")) {
        posts.push(JSON.parse(String(options.body)));
        return launchStatus === 202
          ? json(LAUNCHED)
          : json({ detail: "a training run is already in progress" }, launchStatus);
      }
      if (path.endsWith("/api/datasets")) return json(DATASETS);
      if (path.endsWith("/api/runs/active")) return json(null);
      if (path.endsWith("/api/runs/run-test")) {
        return json({
          id: "run-test",
          dataset: "highest_t",
          status: "trained",
          steps: 40,
          metrics: { iou_holdout: 0.901 },
          config: null,
          artifacts: { checkpoint: true, metrics: true, groups: false, video: false, figures: [] },
        });
      }
      if (path.endsWith("/api/runs")) return json(TRAINED_RUNS);
      return json({ detail: "not found" }, 404);
    }),
  );
  return posts;
}

beforeEach(() => {
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource as unknown as typeof EventSource);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("SolverView", () => {
  it("renders the run configuration at the pipeline's defaults", async () => {
    stubApi();
    render(<SolverView />);
    expect(await screen.findByLabelText("Dataset")).toHaveValue("highest_t");
    expect(screen.getByLabelText("Steps")).toHaveValue(1500);
    expect(screen.getByLabelText(/Learning rate/)).toHaveValue(0.002);
    expect(screen.getByLabelText(/w\s*data/)).toHaveValue(10);
    const idleLine = "[naviernet] solver idle — configure a run and press Run";
    expect(screen.getByText(idleLine)).toBeInTheDocument();
  });

  it("launches a run from the form and follows it live over SSE", async () => {
    const posts = stubApi();
    render(<SolverView />);
    await screen.findByLabelText("Dataset");

    fireEvent.change(screen.getByLabelText("Steps"), { target: { value: "40" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.dataset).toBe("highest_t");
    expect(body.steps).toBe(40);
    expect(body.weights).toEqual({ data: 10, vof: 1, div: 1, src: 0.1, bc: 5 });
    expect(body.render).toBe(true);

    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    const stream = FakeEventSource.instances[0];
    expect(stream.url).toContain("/api/runs/run-test/stream");

    const record = { step: 10, lr: 0.002, data: 0.5, vof: 0.04, div: 0.01, src: 0.001, bc: 0.02 };
    act(() => {
      stream.emit("log", { line: "[naviernet] starting run run-test", tone: "dim" });
      stream.emit("hist", record);
    });
    expect(screen.getByText("[naviernet] starting run run-test")).toBeInTheDocument();
    expect(screen.getByText("5.00e-1")).toBeInTheDocument(); // latest data loss
    // Step progress advances from hist records alone (status only changes per stage).
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "10");
    expect(screen.getByText("running · train")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();

    act(() => {
      stream.emit("status", { ...LAUNCHED, state: "done", stage: null, steps_done: 40 });
    });
    expect(await screen.findByText("done")).toBeInTheDocument();
    expect(stream.closed).toBe(true);
    // The holdout IoU arrives from the finished run's metrics.
    expect(await screen.findByText("0.901")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();
  });

  it("surfaces a rejected launch as an alert", async () => {
    stubApi({ launchStatus: 409 });
    render(<SolverView />);
    await screen.findByLabelText("Dataset");
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("a training run is already in progress");
  });

  it("resume locks the config to the original run and posts run_id", async () => {
    const posts = stubApi();
    render(<SolverView />);
    await screen.findByLabelText("Dataset");

    fireEvent.click(screen.getByRole("switch", { name: "Resume from checkpoint" }));
    expect(screen.getByLabelText("Dataset")).toBeDisabled();
    expect(screen.getByLabelText(/Learning rate/)).toBeDisabled();
    const resumeSelect = await screen.findByLabelText("Run to resume");
    expect(resumeSelect).toHaveValue("highest_t");

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.resume).toBe(true);
    expect(body.run_id).toBe("highest_t");
    expect(body.dataset).toBeUndefined();
  });
});

describe("SolverView sweep mode", () => {
  it("launches a sweep with parsed seeds and shows the child chips", async () => {
    const posts: unknown[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: RequestInfo | URL, options?: RequestInit) => {
        const path = String(url);
        if (options?.method === "POST" && path.endsWith("/api/sweeps")) {
          posts.push(JSON.parse(String(options.body)));
          return json({
            sweep_id: "sweep-test",
            dataset: "highest_t",
            state: "running",
            message: null,
            seeds: [3, 4],
            children: [
              { run_id: "sweep-test-s3", dataset: "highest_t", state: "queued", stage: null, message: null, steps_done: 0, steps_total: 0 },
              { run_id: "sweep-test-s4", dataset: "highest_t", state: "queued", stage: null, message: null, steps_done: 0, steps_total: 0 },
            ],
          });
        }
        if (path.endsWith("/api/datasets")) return json(DATASETS);
        if (path.endsWith("/api/sweeps/active")) return json(null);
        if (path.endsWith("/api/runs/active")) return json(null);
        if (path.endsWith("/api/sweeps/sweep-test")) {
          return json({
            sweep_id: "sweep-test",
            dataset: "highest_t",
            state: "running",
            message: null,
            seeds: [3, 4],
            children: [
              { run_id: "sweep-test-s3", dataset: "highest_t", state: "running", stage: "train", message: null, steps_done: 0, steps_total: 40 },
              { run_id: "sweep-test-s4", dataset: "highest_t", state: "queued", stage: null, message: null, steps_done: 0, steps_total: 0 },
            ],
          });
        }
        if (path.endsWith("/api/runs")) return json(TRAINED_RUNS);
        return json({ detail: "not found" }, 404);
      }),
    );

    render(<SolverView />);
    await screen.findByLabelText("Dataset");
    fireEvent.click(screen.getByRole("switch", { name: "Seed sweep" }));
    fireEvent.change(screen.getByLabelText(/Seeds/), { target: { value: "3, 4" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.seeds).toEqual([3, 4]);
    expect(body.dataset).toBe("highest_t");
    expect(body.resume).toBeUndefined();

    // The sweep panel lists both children (queued until each starts).
    expect(await screen.findByText("Seed sweep", { selector: "h2" })).toBeInTheDocument();
    expect(screen.getByText(/seed 3/)).toBeInTheDocument();
    expect(screen.getByText(/seed 4/)).toBeInTheDocument();
    // The running child gets streamed once polling notices it.
    await waitFor(() => expect(FakeEventSource.instances.length).toBeGreaterThan(0), {
      timeout: 3000,
    });
    expect(FakeEventSource.instances[0].url).toContain("/api/runs/sweep-test-s3/stream");
  });

  it("rejects an unparseable seed list by disabling Run", async () => {
    stubApi();
    render(<SolverView />);
    await screen.findByLabelText("Dataset");
    fireEvent.click(screen.getByRole("switch", { name: "Seed sweep" }));
    fireEvent.change(screen.getByLabelText(/Seeds/), { target: { value: "1, 1, x" } });
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
    expect(screen.getByLabelText(/Seeds/)).toHaveAttribute("aria-invalid", "true");
  });
});

describe("solver components", () => {
  it("Console renders tones and stays a live region", () => {
    const { container } = render(
      <Console
        lines={[
          { line: "plain", tone: null },
          { line: "good", tone: "ok" },
          { line: "note", tone: "dim" },
        ]}
        label="Solver console"
      />,
    );
    expect(screen.getByRole("log", { name: "Solver console" })).toBeInTheDocument();
    expect(container.querySelector(".ok")).toHaveTextContent("good");
    expect(container.querySelector(".dim")).toHaveTextContent("note");
  });

  it("Switch toggles aria-checked", () => {
    const onChange = vi.fn();
    render(<Switch label="Render deliverables" checked={false} onChange={onChange} />);
    const sw = screen.getByRole("switch", { name: "Render deliverables" });
    expect(sw).toHaveAttribute("aria-checked", "false");
    fireEvent.click(sw);
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("Meter exposes progress semantics", () => {
    render(<Meter value={10} max={40} label="Training progress" />);
    const bar = screen.getByRole("progressbar", { name: "Training progress" });
    expect(bar).toHaveAttribute("aria-valuenow", "10");
    expect(bar).toHaveAttribute("aria-valuemax", "40");
  });

  it("LossChart draws one line per loss term once two records exist", () => {
    const records = [
      { step: 10, lr: 2e-3, data: 0.5, vof: 0.04, div: 0.01, src: 1e-3, bc: 0.02 },
      { step: 20, lr: 2e-3, data: 0.3, vof: 0.03, div: 0.008, src: 9e-4, bc: 0.015 },
    ];
    const { container } = render(<LossChart records={records} rebalanceSteps={[10]} />);
    expect(container.querySelectorAll("path.chart-line")).toHaveLength(3);
    expect(container.querySelectorAll("line.chart-marker")).toHaveLength(1);
    const empty = render(<LossChart records={[]} />);
    expect(empty.container.querySelectorAll("path.chart-line")).toHaveLength(0);
  });
});
