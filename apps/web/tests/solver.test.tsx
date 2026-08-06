/** Solver view: config form → launch → live SSE console/chart → terminal states. */

import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Console } from "../src/components/Console";
import { LossChart } from "../src/components/charts/LossChart";
import { Meter } from "../src/components/Meter";
import { Switch } from "../src/components/Switch";
import { SolverView } from "../src/views/SolverView";
import { MonitorPanel } from "../src/views/solver/MonitorPanel";
import { renderStage } from "./stageHarness";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

const DATASETS = [{ id: "highest_t", n_frames: 11, processed: true }];
const TRAINED_RUNS = [
  {
    id: "highest_t",
    dataset: "highest_t",
    status: "trained",
    steps: 1500,
    iou_holdout: 0.968,
  },
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

/** What a series trains. The Solver reads this to know whether the sharp
 * treatment (pressure) and the depletable superheat (temperature) are even
 * reachable, so every stub has to answer it. */
const STAGE_B_FIELDS = ["phi", "u", "v", "s", "p", "T"];
const STAGE_A_FIELDS = ["phi", "u", "v", "s"];

function stubApi({
  launchStatus = 202,
  launchDetail = "a training run is already in progress",
  fields = STAGE_B_FIELDS,
}: {
  launchStatus?: number;
  /** Error `detail` for a non-202 launch; an array mimics FastAPI's 422 shape. */
  launchDetail?: unknown;
  /** The fields the primary series trains (Stage-A series train no p or T). */
  fields?: string[];
} = {}) {
  const posts: unknown[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: RequestInfo | URL, options?: RequestInit) => {
      const path = String(url);
      if (options?.method === "POST" && path.endsWith("/api/runs")) {
        posts.push(JSON.parse(String(options.body)));
        return launchStatus === 202
          ? json(LAUNCHED)
          : json({ detail: launchDetail }, launchStatus);
      }
      if (path.includes("/api/physics/")) {
        return json({
          dataset: path.split("/").pop(),
          equations: [],
          fields,
          groups: {},
          sharp_interface: fields.includes("p"),
        });
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
          artifacts: {
            checkpoint: true,
            metrics: true,
            groups: false,
            video: false,
            figures: [],
          },
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
  vi.stubGlobal(
    "EventSource",
    FakeEventSource as unknown as typeof EventSource,
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("SolverView", () => {
  it("renders the run configuration at the pipeline's defaults", async () => {
    stubApi();
    renderStage(<SolverView />);
    expect(await screen.findByLabelText("highest_t")).toBeChecked();
    expect(screen.getByLabelText(/Steps/)).toHaveValue(1500);
    expect(screen.getByLabelText(/Learning rate/)).toHaveValue(0.002);
    expect(screen.getByLabelText(/w\s*data/)).toHaveValue(10);
    // The recommended recipe is the rung the ladder starts on.
    expect(screen.getByRole("radio", { name: /Sharp front/ })).toBeChecked();
    const idleLine = "[naviernet] solver idle; configure a run and press Run";
    expect(screen.getByText(idleLine)).toBeInTheDocument();
  });

  it("states what Run will do, and why it cannot be pressed", async () => {
    stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    expect(
      screen.getByText(/highest_t · 1500 steps · sharp front/),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Clear all" }));
    expect(
      screen.getByText("Select at least one series to train on."),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
  });

  it("launches a run from the form and follows it live over SSE", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    fireEvent.change(screen.getByLabelText(/Steps/), {
      target: { value: "40" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.datasets).toEqual(["highest_t"]);
    expect(body.steps).toBe(40);
    expect(body.weights).toEqual({ data: 10, vof: 1, div: 1, src: 0.1, bc: 5 });
    expect(body.render).toBe(true);

    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    const stream = FakeEventSource.instances[0];
    expect(stream.url).toContain("/api/runs/run-test/stream");

    const record = {
      step: 10,
      lr: 0.002,
      data: 0.5,
      vof: 0.04,
      div: 0.01,
      src: 0.001,
      bc: 0.02,
    };
    act(() => {
      stream.emit("log", {
        line: "[naviernet] starting run run-test",
        tone: "dim",
      });
      stream.emit("hist", record);
    });
    expect(
      screen.getByText("[naviernet] starting run run-test"),
    ).toBeInTheDocument();
    expect(screen.getByText("5.00e-1")).toBeInTheDocument(); // latest data loss
    // Step progress advances from hist records alone (status only changes per stage).
    expect(screen.getByRole("progressbar")).toHaveAttribute(
      "aria-valuenow",
      "10",
    );
    expect(screen.getByText("running · train")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();

    act(() => {
      stream.emit("status", {
        ...LAUNCHED,
        state: "done",
        stage: null,
        steps_done: 40,
      });
    });
    expect(await screen.findByText("done")).toBeInTheDocument();
    expect(stream.closed).toBe(true);
    // The holdout IoU arrives from the finished run's metrics.
    expect(await screen.findByText("0.901")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();
  });

  function stubMultiDataset(
    entries: { id: string; processed: boolean }[],
    posts: unknown[],
  ) {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: RequestInfo | URL, options?: RequestInit) => {
        const path = String(url);
        if (options?.method === "POST" && path.endsWith("/api/runs")) {
          posts.push(JSON.parse(String(options.body)));
          return json(LAUNCHED);
        }
        if (path.endsWith("/api/datasets"))
          return json(entries.map((e) => ({ ...e, n_frames: 11 })));
        if (path.endsWith("/api/runs/active")) return json(null);
        if (path.endsWith("/api/runs")) return json([]);
        return json({ detail: "not found" }, 404);
      }),
    );
  }

  it("multi-selects the project's datasets and launches one joint run", async () => {
    const posts: unknown[] = [];
    stubMultiDataset(
      [
        { id: "ds_a", processed: true },
        { id: "ds_b", processed: true },
        { id: "ds_raw", processed: false }, // not preprocessed: never offered
      ],
      posts,
    );
    renderStage(<SolverView />);

    // Every processed series is selected by default; the unprocessed one is absent.
    expect(await screen.findByLabelText("ds_a")).toBeChecked();
    expect(screen.getByLabelText("ds_b")).toBeChecked();
    expect(screen.queryByLabelText("ds_raw")).toBeNull();

    // Clearing the selection disables Run; selecting all re-enables it.
    fireEvent.click(screen.getByRole("button", { name: "Clear all" }));
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Select all" }));
    expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    // Both datasets go in one request: the API trains them jointly.
    expect((posts[0] as Record<string, unknown>).datasets).toEqual([
      "ds_a",
      "ds_b",
    ]);
  });

  it("offers a validation split (off by default) and posts the chosen fraction", async () => {
    const posts: unknown[] = [];
    stubMultiDataset(
      [
        { id: "ds_a", processed: true },
        { id: "ds_b", processed: true },
      ],
      posts,
    );
    renderStage(<SolverView />);
    await screen.findByLabelText("ds_a");

    // Off by default (train on every frame); the user opts in. No strategy toggle.
    expect(screen.getByLabelText(/Validation split/)).toHaveValue("0");
    expect(screen.queryByLabelText(/Split strategy/)).toBeNull();
    fireEvent.change(screen.getByLabelText(/Validation split/), {
      target: { value: "0.2" },
    });

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.val_fraction).toBe(0.2);
    expect(body.val_strategy).toBe("tail");
    expect(body.heldout_datasets).toBeUndefined();
  });

  it("a single series offers no hold-out toggle and no frame holdout", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    // Hold-out is a series concept: with one series there's nothing to hold out.
    expect(screen.queryByRole("button", { name: /Hold out/ })).toBeNull();
    // The legacy per-frame "Holdout frame" control is gone from the Solver.
    expect(screen.queryByLabelText(/Holdout frame/)).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.val_fraction).toBe(0);
    expect(body.holdout_frame).toBe(-1); // no single-frame holdout
    expect(body.heldout_datasets).toBeUndefined();
  });

  it("starts at today's recipe: gradnorm, no causal, no adaptive", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    expect(screen.getByLabelText(/Weighting/)).toHaveValue("gradnorm");
    // The causal mode select only appears once causal weighting is on.
    expect(screen.queryByLabelText(/Causal mode/)).toBeNull();
    // Adaptive collocation refreshes the RBA pool, so it is gated off under gradnorm.
    expect(
      screen.getByRole("switch", { name: "Adaptive collocation" }),
    ).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.weighting).toBe("gradnorm");
    expect(body.causal_weighting).toBe(false);
    expect(body.adaptive_collocation).toBe(false);
  });

  it("enables causal time-marching and posts the chosen mode", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    fireEvent.click(screen.getByRole("switch", { name: "Causal weighting" }));
    // The mode select appears, defaulting to soft weighting.
    expect(screen.getByLabelText(/Causal mode/)).toHaveValue("weight");
    fireEvent.change(screen.getByLabelText(/Causal mode/), {
      target: { value: "march" },
    });

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.causal_weighting).toBe(true);
    expect(body.causal_mode).toBe("march");
  });

  it("gates RBA / causal / adaptive valid-by-construction in both directions", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    // Turn causal ON under gradnorm, then switch to RBA: causal is forced off and
    // disabled (RBA and causal are not composable yet).
    fireEvent.click(screen.getByRole("switch", { name: "Causal weighting" }));
    expect(
      screen.getByRole("switch", { name: "Causal weighting" }),
    ).toBeChecked();
    fireEvent.change(screen.getByLabelText(/Weighting/), {
      target: { value: "rba" },
    });
    const causal = screen.getByRole("switch", { name: "Causal weighting" });
    expect(causal).toBeDisabled();
    expect(causal).not.toBeChecked();

    // Under RBA, adaptive is available; turn it on, then switch back to gradnorm:
    // adaptive is forced off and disabled (it requires RBA).
    const adaptive = screen.getByRole("switch", {
      name: "Adaptive collocation",
    });
    expect(adaptive).toBeEnabled();
    fireEvent.click(adaptive);
    expect(adaptive).toBeChecked();
    fireEvent.change(screen.getByLabelText(/Weighting/), {
      target: { value: "gradnorm" },
    });
    expect(
      screen.getByRole("switch", { name: "Adaptive collocation" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("switch", { name: "Adaptive collocation" }),
    ).not.toBeChecked();

    // Finally launch an RBA + adaptive run and confirm the posted combo is valid.
    fireEvent.change(screen.getByLabelText(/Weighting/), {
      target: { value: "rba" },
    });
    fireEvent.click(
      screen.getByRole("switch", { name: "Adaptive collocation" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.weighting).toBe("rba");
    expect(body.adaptive_collocation).toBe(true);
    expect(body.causal_weighting).toBe(false);
  });

  it("starts with the pin and growth constraints off and posts them so", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    // The pin belongs to the diffuse rung alone: the explicit front pins the
    // root by construction, so under the default there is no pin to offer.
    expect(screen.queryByRole("switch", { name: "Hard root pin" })).toBeNull();
    expect(
      screen.getByRole("switch", { name: "Growth constraints" }),
    ).not.toBeChecked();
    // The scale/weight fields only appear once their feature is on.
    expect(screen.queryByLabelText(/Pin gate scale/)).toBeNull();
    expect(screen.queryByLabelText(/Growth margin/)).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.hard_pin).toBe(false);
    expect(body.kinematics).toBe(false);
    expect(body.kin_weight_evap).toBe(0); // the bench-rejected floor stays off
  });

  it("enables the pin + growth constraints and posts the tuned values", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    // The explicit front pins the root exactly, so a run that wants the SOFT
    // pin has to step down to the diffuse treatment first.
    fireEvent.click(screen.getByRole("radio", { name: /Diffuse/ }));
    fireEvent.click(screen.getByRole("switch", { name: "Hard root pin" }));
    expect(screen.getByLabelText(/Pin gate scale/)).toHaveValue(0.1);
    fireEvent.click(screen.getByRole("switch", { name: "Growth constraints" }));
    // Growth fields appear at the recorded defaults; evap floor starts at 0.
    expect(screen.getByLabelText(/Growth margin/)).toHaveValue(0.3);
    expect(screen.getByLabelText(/Evap-floor weight/)).toHaveValue(0);
    fireEvent.change(screen.getByLabelText(/Growth margin/), {
      target: { value: "0.65" },
    });

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.hard_pin).toBe(true);
    expect(body.pin_d_ref).toBe(0.1);
    expect(body.kinematics).toBe(true);
    expect(body.kin_margin_frac).toBe(0.65);
    expect(body.kin_weight_mono).toBe(1);
    expect(body.kin_weight_balance).toBe(1);
    expect(body.kin_weight_evap).toBe(0);
  });

  it("makes the pin and the explicit front mutually exclusive by construction", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    // Set the pin on the rung that offers it, then climb back: the pin is not
    // merely disabled, it is not part of this formulation at all -- the same
    // combination the trainer rejects, made unreachable rather than refused.
    fireEvent.click(screen.getByRole("radio", { name: /Diffuse/ }));
    fireEvent.click(screen.getByRole("switch", { name: "Hard root pin" }));
    fireEvent.click(screen.getByRole("radio", { name: /Explicit front/ }));
    expect(screen.queryByRole("switch", { name: "Hard root pin" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.front_geometry).toBe(true);
    expect(body.hard_pin).toBe(false);
  });

  it("offers each treatment only the options that treatment admits", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    // Sharp: the film-pressure correction exists because the jump does.
    expect(
      screen.getByRole("switch", { name: "Film pressure" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("switch", { name: "Allow pinch-off" }),
    ).toBeInTheDocument();

    // Explicit front: the front options remain, the jump's correction does not.
    fireEvent.click(screen.getByRole("radio", { name: /Explicit front/ }));
    expect(screen.queryByRole("switch", { name: "Film pressure" })).toBeNull();
    expect(
      screen.getByRole("switch", { name: "Allow pinch-off" }),
    ).toBeInTheDocument();

    // Diffuse: no front at all, so nothing that reads one is offered.
    fireEvent.click(screen.getByRole("radio", { name: /Diffuse/ }));
    expect(
      screen.queryByRole("switch", { name: "Allow pinch-off" }),
    ).toBeNull();
    expect(
      screen.getByRole("switch", { name: "Hard root pin" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    // Every flag below the chosen rung is cleared, not left dangling.
    expect(body.front_geometry).toBe(false);
    expect(body.sharp_interface).toBe(false);
    expect(body.film_pressure).toBe(false);
    expect(body.allow_pinch).toBe(false);
    expect(body.front_velocity).toBe(false);
  });

  it("offers the free end caps on both front rungs, and never on the diffuse one", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    // Sharp and explicit-front both have caps to free.
    expect(
      screen.getByRole("switch", { name: "Free the end caps" }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: /Explicit front/ }));
    expect(
      screen.getByRole("switch", { name: "Free the end caps" }),
    ).toBeInTheDocument();

    // A diffuse run has no capsule, so it has no caps to free.
    fireEvent.click(screen.getByRole("radio", { name: /Diffuse/ }));
    expect(
      screen.queryByRole("switch", { name: "Free the end caps" }),
    ).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    expect((posts[0] as Record<string, unknown>).cap_freedom).toBe(false);
  });

  it("sends the freed caps and their bound with the run", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    fireEvent.click(screen.getByRole("switch", { name: "Free the end caps" }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.cap_freedom).toBe(true);
    // The bound rides along at its default; it is tuned from the CLI/API, so the
    // aside offers the decision rather than every dial.
    expect(body.cap_delta).toBe(0.2);
  });

  it("names the momentum closure each treatment implies", async () => {
    stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    // The thing that actually changes between the rungs, said out loud.
    expect(
      screen.getByText(/Darcy drag, imposed ON the front/),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: /Explicit front/ }));
    expect(
      screen.getByText(/2-D depth-averaged momentum \(Hele-Shaw\)/),
    ).toBeInTheDocument();
  });

  it("climbs back to the sharp rung with its recipe intact", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    fireEvent.click(screen.getByRole("radio", { name: /Diffuse/ }));
    fireEvent.click(screen.getByRole("radio", { name: /Sharp front/ }));
    fireEvent.click(screen.getByRole("switch", { name: "Allow pinch-off" }));

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.front_geometry).toBe(true);
    expect(body.sharp_interface).toBe(true);
    expect(body.film_pressure).toBe(true);
    expect(body.allow_pinch).toBe(true);
  });

  it("reveals the sharpening fields only once sharpening is switched on", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    // Off by default: the interface holds its width unless asked otherwise.
    expect(screen.queryByLabelText(/Anneal over/)).toBeNull();
    fireEvent.click(
      screen.getByRole("switch", { name: "Sharpen the interface" }),
    );
    expect(screen.getByLabelText(/Anneal over/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Final/)).toHaveValue(0.02);

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    // A target of 0 with a non-zero schedule is a 422; the switch cannot make one.
    expect(body.alpha_eps_anneal_steps).toBeGreaterThan(0);
    expect(body.alpha_eps_final).toBe(0.02);
  });

  it("toggling the pin and growth constraints off hides their fields again", async () => {
    stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    fireEvent.click(screen.getByRole("radio", { name: /Diffuse/ }));
    fireEvent.click(screen.getByRole("switch", { name: "Hard root pin" }));
    fireEvent.click(screen.getByRole("switch", { name: "Growth constraints" }));
    expect(screen.getByLabelText(/Pin gate scale/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Growth margin/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("switch", { name: "Hard root pin" }));
    expect(screen.queryByLabelText(/Pin gate scale/)).toBeNull();
    expect(screen.getByLabelText(/Growth margin/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("switch", { name: "Growth constraints" }));
    expect(screen.queryByLabelText(/Growth margin/)).toBeNull();
  });

  it("refuses a treatment the series cannot support, and says why", async () => {
    const posts = stubApi({ fields: STAGE_A_FIELDS });
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    // A Stage-A series trains no pressure, so the sharp rung is unreachable --
    // and the API would reject an explicit ask for it. The form steps down to
    // the explicit front and states the reason at the control.
    await waitFor(() =>
      expect(
        screen.getByRole("radio", { name: /Explicit front/ }),
      ).toBeChecked(),
    );
    expect(screen.getByRole("radio", { name: /Sharp front/ })).toHaveAttribute(
      "aria-disabled",
      "true",
    );
    expect(
      screen.getByText(/does not\s+train\. Enable Momentum/),
    ).toBeInTheDocument();
    // Same for the superheat, which reads a temperature field this series has not got.
    expect(
      screen.getByRole("switch", { name: "Depletable superheat" }),
    ).not.toBeChecked();

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.sharp_interface).toBe(false);
    expect(body.film_pressure).toBe(false);
    expect(body.depletable_superheat).toBe(false);
    expect(body.front_geometry).toBe(true); // the capsule needs no Stage-B field
  });

  it("holds out a whole series from its row and posts it as a transfer condition", async () => {
    const posts: unknown[] = [];
    stubMultiDataset(
      [
        { id: "ds_a", processed: true },
        { id: "ds_b", processed: true },
      ],
      posts,
    );
    renderStage(<SolverView />);
    await screen.findByLabelText("ds_a");

    // Two series -> each row gets a hold-out toggle; hold ds_b out, ds_a trains.
    fireEvent.click(screen.getByRole("button", { name: "Hold out ds_b" }));
    expect(
      screen.getByRole("button", { name: "Return to training ds_b" }),
    ).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.datasets).toEqual(["ds_a", "ds_b"]);
    expect(body.heldout_datasets).toEqual(["ds_b"]);
  });

  it("holding out a different series moves the single hold-out", async () => {
    const posts: unknown[] = [];
    stubMultiDataset(
      [
        { id: "ds_a", processed: true },
        { id: "ds_b", processed: true },
        { id: "ds_c", processed: true },
      ],
      posts,
    );
    renderStage(<SolverView />);
    await screen.findByLabelText("ds_a");

    fireEvent.click(screen.getByRole("button", { name: "Hold out ds_b" }));
    // Marking ds_c clears ds_b — only one series is ever held out.
    fireEvent.click(screen.getByRole("button", { name: "Hold out ds_c" }));

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.heldout_datasets).toEqual(["ds_c"]);
  });

  it("clears the held-out series when it is deselected from the run", async () => {
    const posts: unknown[] = [];
    stubMultiDataset(
      [
        { id: "ds_a", processed: true },
        { id: "ds_b", processed: true },
      ],
      posts,
    );
    renderStage(<SolverView />);
    await screen.findByLabelText("ds_a");

    // Hold out ds_b, then uncheck ds_a — ds_b must not be left as the sole,
    // held-out series (which the API would reject on submit).
    fireEvent.click(screen.getByRole("button", { name: "Hold out ds_b" }));
    fireEvent.click(screen.getByLabelText("ds_a"));

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.datasets).toEqual(["ds_b"]);
    expect(body.heldout_datasets).toBeUndefined();
  });

  it("a finished joint run surfaces its validation and transfer IoU in the monitor", async () => {
    // End-to-end wiring: launch → SSE done → useSolverRun fetches the finished
    // run's metrics.json v2 and MonitorPanel renders both generalization axes.
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: RequestInfo | URL, options?: RequestInit) => {
        const path = String(url);
        if (options?.method === "POST" && path.endsWith("/api/runs")) {
          return json(LAUNCHED);
        }
        if (path.endsWith("/api/datasets"))
          return json([
            { id: "ds_a", n_frames: 11, processed: true },
            { id: "ds_b", n_frames: 11, processed: true },
          ]);
        if (path.endsWith("/api/runs/active")) return json(null);
        if (path.endsWith("/api/runs/run-test"))
          return json({
            id: "run-test",
            dataset: "ds_a",
            status: "trained",
            steps: 40,
            metrics: { val_iou_mean: 0.86, transfer: { mean: 0.71 } },
            config: null,
            artifacts: {
              checkpoint: true,
              metrics: true,
              groups: false,
              video: false,
              figures: [],
            },
          });
        if (path.endsWith("/api/runs")) return json([]);
        return json({ detail: "not found" }, 404);
      }),
    );
    renderStage(<SolverView />);
    await screen.findByLabelText("ds_a");

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    act(() => {
      FakeEventSource.instances[0].emit("status", {
        ...LAUNCHED,
        state: "done",
        stage: null,
        steps_done: 40,
      });
    });

    // Both numbers come from the finished run's metrics via the real hook.
    expect(await screen.findByText("0.860")).toBeInTheDocument();
    expect(screen.getByText("Validation IoU")).toBeInTheDocument();
    expect(await screen.findByText("0.710")).toBeInTheDocument();
    expect(screen.getByText("Transfer IoU")).toBeInTheDocument();
    expect(screen.queryByText("Holdout IoU")).toBeNull();
  });

  it("shows a series by its display label while still launching by id", async () => {
    const posts: unknown[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: RequestInfo | URL, options?: RequestInit) => {
        const path = String(url);
        if (options?.method === "POST" && path.endsWith("/api/runs")) {
          posts.push(JSON.parse(String(options.body)));
          return json(LAUNCHED);
        }
        if (path.endsWith("/api/datasets"))
          return json([
            {
              id: "ds_a",
              label: "High-T FC-72",
              n_frames: 11,
              processed: true,
            },
          ]);
        if (path.endsWith("/api/runs/active")) return json(null);
        if (path.endsWith("/api/runs")) return json([]);
        return json({ detail: "not found" }, 404);
      }),
    );
    renderStage(<SolverView />);

    // The row is labelled by the display name, not the raw id.
    expect(await screen.findByLabelText("High-T FC-72")).toBeChecked();

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    // ...but the launch still targets the immutable id.
    expect((posts[0] as Record<string, unknown>).datasets).toEqual(["ds_a"]);
  });

  it("scopes the dataset list to the open project", async () => {
    const posts: unknown[] = [];
    stubMultiDataset(
      [
        { id: "ds_a", processed: true },
        { id: "other", processed: true }, // belongs to a different project
      ],
      posts,
    );
    const project = {
      id: "a".repeat(32),
      name: "P",
      description: "",
      datasets: ["ds_a"],
      created_at: "2026-07-24T00:00:00+00:00",
    };
    renderStage(<SolverView project={project} />);

    expect(await screen.findByLabelText("ds_a")).toBeInTheDocument();
    // A processed dataset from another project must not appear here.
    expect(screen.queryByLabelText("other")).toBeNull();
  });

  it("surfaces a rejected launch as an alert", async () => {
    stubApi({ launchStatus: 409 });
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("a training run is already in progress");
  });

  it("flattens a 422 validation error into a readable field-level alert", async () => {
    // FastAPI reports request-body validation failures as an array of issues;
    // rendering that array raw gives the useless "[object Object]".
    stubApi({
      launchStatus: 422,
      launchDetail: [
        {
          type: "greater_than_equal",
          loc: ["body", "steps"],
          msg: "Input should be greater than or equal to 1",
        },
      ],
    });
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      "steps: Input should be greater than or equal to 1",
    );
    expect(alert).not.toHaveTextContent("[object Object]");
  });

  it("resume locks the config to the original run and posts run_id", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    fireEvent.click(screen.getByRole("radio", { name: "Resume" }));
    // The series list is replaced by the checkpoint it continues, and the note
    // says what a resume actually keeps.
    expect(screen.queryByLabelText("highest_t")).toBeNull();
    expect(screen.getByText(/Only/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Learning rate/)).toBeDisabled();
    // The accuracy controls are fixed by the original run too (resume ignores them).
    expect(screen.getByLabelText(/Weighting/)).toBeDisabled();
    expect(
      screen.getByRole("switch", { name: "Causal weighting" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("switch", { name: "Adaptive collocation" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("switch", { name: "Growth constraints" }),
    ).toBeDisabled();
    expect(screen.getByRole("radio", { name: /Diffuse/ })).toHaveAttribute(
      "aria-disabled",
      "true",
    );
    // Steps is the one field a resume still honours.
    expect(screen.getByLabelText(/Steps/)).toBeEnabled();
    const resumeSelect = await screen.findByLabelText(/Run to resume/);
    expect(resumeSelect).toHaveValue("highest_t");

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.resume).toBe(true);
    expect(body.run_id).toBe("highest_t");
    expect(body.datasets).toBeUndefined();
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
              {
                run_id: "sweep-test-s3",
                dataset: "highest_t",
                state: "queued",
                stage: null,
                message: null,
                steps_done: 0,
                steps_total: 0,
              },
              {
                run_id: "sweep-test-s4",
                dataset: "highest_t",
                state: "queued",
                stage: null,
                message: null,
                steps_done: 0,
                steps_total: 0,
              },
            ],
          });
        }
        if (path.includes("/api/physics/")) {
          return json({
            dataset: "highest_t",
            equations: [],
            fields: STAGE_B_FIELDS,
            groups: {},
            sharp_interface: true,
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
              {
                run_id: "sweep-test-s3",
                dataset: "highest_t",
                state: "running",
                stage: "train",
                message: null,
                steps_done: 0,
                steps_total: 40,
              },
              {
                run_id: "sweep-test-s4",
                dataset: "highest_t",
                state: "queued",
                stage: null,
                message: null,
                steps_done: 0,
                steps_total: 0,
              },
            ],
          });
        }
        if (path.endsWith("/api/runs")) return json(TRAINED_RUNS);
        return json({ detail: "not found" }, 404);
      }),
    );

    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");
    fireEvent.click(screen.getByRole("radio", { name: "Seed sweep" }));
    fireEvent.change(screen.getByLabelText(/Seeds/), {
      target: { value: "3, 4" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.seeds).toEqual([3, 4]);
    expect(body.datasets).toEqual(["highest_t"]);
    expect(body.resume).toBeUndefined();

    // The sweep panel lists both children (queued until each starts).
    expect(
      await screen.findByText("Seed sweep", { selector: "h2" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/seed 3/)).toBeInTheDocument();
    expect(screen.getByText(/seed 4/)).toBeInTheDocument();
    // The running child gets streamed once polling notices it.
    await waitFor(
      () => expect(FakeEventSource.instances.length).toBeGreaterThan(0),
      {
        timeout: 3000,
      },
    );
    expect(FakeEventSource.instances[0].url).toContain(
      "/api/runs/sweep-test-s3/stream",
    );
  });

  it("rejects an unparseable seed list by disabling Run", async () => {
    stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");
    fireEvent.click(screen.getByRole("radio", { name: "Seed sweep" }));
    fireEvent.change(screen.getByLabelText(/Seeds/), {
      target: { value: "1, 1, x" },
    });
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
    expect(screen.getByLabelText(/Seeds/)).toHaveAttribute(
      "aria-invalid",
      "true",
    );
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
    expect(
      screen.getByRole("log", { name: "Solver console" }),
    ).toBeInTheDocument();
    expect(container.querySelector(".ok")).toHaveTextContent("good");
    expect(container.querySelector(".dim")).toHaveTextContent("note");
  });

  it("Switch toggles aria-checked", () => {
    const onChange = vi.fn();
    render(
      <Switch
        label="Render deliverables"
        checked={false}
        onChange={onChange}
      />,
    );
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

  const monitorStatus = {
    run_id: "r",
    dataset: "d",
    state: "done" as const,
    stage: null,
    message: null,
    steps_done: 40,
    steps_total: 40,
  };

  it("MonitorPanel shows the holdout IoU when no split or hold-out is set", () => {
    render(
      <MonitorPanel
        status={monitorStatus}
        latest={null}
        showVal={false}
        showTransfer={false}
        holdoutIou={0.9}
        valIou={null}
        transferIou={null}
      />,
    );
    expect(screen.getByText("Holdout IoU")).toBeInTheDocument();
    expect(screen.getByText("0.900")).toBeInTheDocument();
    expect(screen.queryByText("Transfer IoU")).toBeNull();
    expect(screen.queryByText("Validation IoU")).toBeNull();
  });

  it("MonitorPanel surfaces validation and transfer IoU distinctly", () => {
    render(
      <MonitorPanel
        status={monitorStatus}
        latest={null}
        showVal
        showTransfer
        holdoutIou={null}
        valIou={0.86}
        transferIou={0.71}
      />,
    );
    // The two generalization numbers answer different questions — both shown.
    expect(screen.getByText("Validation IoU")).toBeInTheDocument();
    expect(screen.getByText("0.860")).toBeInTheDocument();
    expect(screen.getByText("Transfer IoU")).toBeInTheDocument();
    expect(screen.getByText("0.710")).toBeInTheDocument();
    // Holdout is retired once the two-axis metrics are shown.
    expect(screen.queryByText("Holdout IoU")).toBeNull();
  });

  it("MonitorPanel omits transfer when no series is held out", () => {
    render(
      <MonitorPanel
        status={monitorStatus}
        latest={null}
        showVal
        showTransfer={false}
        holdoutIou={null}
        valIou={0.86}
        transferIou={null}
      />,
    );
    expect(screen.getByText("Validation IoU")).toBeInTheDocument();
    expect(screen.queryByText("Transfer IoU")).toBeNull();
  });

  it("MonitorPanel labels a configured run before its metrics load", () => {
    // A live run with a validation split has no metrics yet (val null) — it must
    // still be labelled by its config, not fall back to the 'Holdout IoU'.
    render(
      <MonitorPanel
        status={{ ...monitorStatus, state: "running", steps_done: 5 }}
        latest={null}
        showVal
        showTransfer={false}
        holdoutIou={null}
        valIou={null}
        transferIou={null}
      />,
    );
    expect(screen.getByText("Validation IoU")).toBeInTheDocument();
    expect(screen.queryByText("Holdout IoU")).toBeNull();
    // With no value yet, the stat must explain the wait rather than read as broken.
    expect(screen.getByText("known after evaluation")).toBeInTheDocument();
    expect(screen.queryByText("held-out frames · in-distribution")).toBeNull();
  });

  it("LossChart draws one line per loss term once two records exist", () => {
    const records = [
      {
        step: 10,
        lr: 2e-3,
        data: 0.5,
        vof: 0.04,
        div: 0.01,
        src: 1e-3,
        bc: 0.02,
      },
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
    const { container } = render(
      <LossChart records={records} rebalanceSteps={[10]} />,
    );
    expect(container.querySelectorAll("path.chart-line")).toHaveLength(3);
    expect(container.querySelectorAll("line.chart-marker")).toHaveLength(1);
    const empty = render(<LossChart records={[]} />);
    expect(empty.container.querySelectorAll("path.chart-line")).toHaveLength(0);
  });
});

describe("SolverView film pressure", () => {
  it("exists only where the jump it corrects does", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    // It corrects the Young-Laplace jump, so it belongs to the sharp rung and
    // nowhere else: stepping down does not disable it, it removes it.
    expect(screen.getByRole("switch", { name: "Film pressure" })).toBeChecked();
    fireEvent.click(screen.getByRole("radio", { name: /Explicit front/ }));
    expect(screen.queryByRole("switch", { name: "Film pressure" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    expect((posts[0] as Record<string, unknown>).film_pressure).toBe(false);
  });
});

describe("SolverView defaults", () => {
  it("posts the recommended physics recipe without the user touching anything", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    // Hitting Run straight away must give the best-known recipe. Every measured
    // gain in this line of work was built on it.
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));

    const body = posts[0] as Record<string, unknown>;
    expect(body.front_geometry).toBe(true);
    expect(body.sharp_interface).toBe(true);
    expect(body.film_pressure).toBe(true);
    expect(body.depletable_superheat).toBe(true);
    expect(body.evap_closure_two_way).toBe(true);
    // Not in the recipe: these traded one number for another when measured.
    expect(body.allow_pinch).toBe(false);
    expect(body.alpha_eps_anneal_steps).toBe(0);
    expect(body.hard_pin).toBe(false);
  });
});

describe("SolverView two-way closure", () => {
  it("is switchable, not just silently sent", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    // It ships on by default; the point of this test is that the panel can turn
    // it OFF -- a knob that is sent but has no control is not configurable.
    const closure = screen.getByRole("switch", {
      name: "Two-way evaporation",
    });
    expect(closure).toBeChecked();
    fireEvent.click(closure);
    expect(closure).not.toBeChecked();

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    expect((posts[0] as Record<string, unknown>).evap_closure_two_way).toBe(
      false,
    );
  });
});

describe("SolverView evolving width", () => {
  it("is offered only with an explicit front, and reaches the request", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    fireEvent.click(screen.getByRole("radio", { name: /Diffuse/ }));
    expect(screen.queryByRole("switch", { name: "Evolving width" })).toBeNull();
    fireEvent.click(screen.getByRole("radio", { name: /Sharp front/ }));
    fireEvent.click(screen.getByRole("switch", { name: "Evolving width" }));

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    expect((posts[0] as Record<string, unknown>).evolving_width).toBe(true);
  });
});

describe("SolverView measured front velocity", () => {
  it("is unavailable without a front, and reaches the request with one", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    // The switch stays on screen under the diffuse treatment, carrying the only
    // explanation there is for why it cannot be used.
    fireEvent.click(screen.getByRole("radio", { name: /Diffuse/ }));
    expect(
      screen.getByRole("switch", { name: "Measured front velocity" }),
    ).toBeDisabled();
    expect(
      screen.getByText(/no front to give a normal speed/),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: /Sharp front/ }));
    fireEvent.click(
      screen.getByRole("switch", { name: "Measured front velocity" }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    const body = posts[0] as Record<string, unknown>;
    expect(body.front_velocity).toBe(true);
    expect(body.fv_weight).toBe(10);
    expect(body.fv_apex_weight).toBe(10);
  });

  it("will not launch a measurement it then ignores", async () => {
    stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    fireEvent.click(
      screen.getByRole("switch", { name: "Measured front velocity" }),
    );
    for (const field of [/Normal-speed weight/, /Apex weight/]) {
      fireEvent.change(screen.getByLabelText(field), {
        target: { value: "0" },
      });
    }

    // The API rejects this combination; the dock says so before the request.
    expect(
      screen.getByText(/needs a weight on the profile or the apex/),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
  });

  it("reveals its weight fields only while it is on", async () => {
    stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    expect(screen.queryByLabelText(/Normal-speed weight/)).toBeNull();
    fireEvent.click(
      screen.getByRole("switch", { name: "Measured front velocity" }),
    );
    expect(screen.getByLabelText(/Normal-speed weight/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Apex weight/)).toBeInTheDocument();
  });

  it("stepping down to the diffuse treatment takes it with it", async () => {
    const posts = stubApi();
    renderStage(<SolverView />);
    await screen.findByLabelText("highest_t");

    fireEvent.click(
      screen.getByRole("switch", { name: "Measured front velocity" }),
    );
    fireEvent.click(screen.getByRole("radio", { name: /Diffuse/ }));

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts).toHaveLength(1));
    expect((posts[0] as Record<string, unknown>).front_velocity).toBe(false);
  });
});
