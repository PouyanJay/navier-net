import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
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
    // A legacy CLI run auto-named after its dataset: it must display the
    // series' current label, never the stale id.
    id: "highest_t",
    dataset: "highest_t",
    status: "trained",
    steps: 800,
    iou_holdout: 0.96,
    datasets: ["highest_t"],
    heldout_datasets: [],
    val_iou_mean: null,
    date: "2026-07-23T17:00:00+00:00",
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
  // The series was renamed: the id stays the immutable key, the label is
  // what every surface must show.
  { id: "highest_t", label: "series-1", frames: 11, processed: true },
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
      iou_per_frame: { "1": 0.973, "6": 0.968, "10": 0.921 },
      holdout_frame: run.iou_holdout != null ? 6 : null,
      validation_frames: [10],
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
      const groupsMatch = u.match(/\/api\/datasets\/([^/]+)\/groups$/);
      if (groupsMatch) {
        const heldout = groupsMatch[1] === "low_t";
        return json({
          Re: heldout ? 265 : 320,
          We: heldout ? 1.4 : 1.8,
          Ca: heldout ? 0.0044 : 0.0056,
          Ja: heldout ? 0.22 : 0.31,
          Pr: 9.4,
          hele_shaw: heldout ? 5.2 : 1.9,
        });
      }
      if (u.includes("/api/datasets/highest_t/qc-data")) {
        return json({
          dataset: "highest_t",
          n_frames_event: 10,
          kinematics: {
            t_ms: [0],
            length_um: [0],
            fit_slope_mm_s: 180,
            fit_intercept_um: 0,
          },
          interface: {
            x_pin_star: 0.1,
            x_range: [0, 5.6],
            y_range: [0, 1.2],
            l_ref_um: 300,
            y_roi_top: 40,
            frames: [1, 6, 10].map((n) => ({
              index: n - 1,
              camera_frame: n,
              t_ms: (n - 1) * 0.5,
              rings: [
                [
                  [0.1, 0.2],
                  [0.5, 0.2],
                  [0.5, 0.9],
                ],
              ],
            })),
          },
          sdf: {
            frame_index: 0,
            t_ms: 0,
            x_range: [0, 5.6],
            y_range: [0, 1.2],
            values: [[0]],
          },
        });
      }
      if (u.match(/\/api\/datasets\/highest_t$/)) {
        return json({
          id: "highest_t",
          n_frames: 12,
          processed: true,
          conditions_set: true,
          label: "highest_t",
          frame_px: [1024, 256],
          dt_frame_ms: 0.5,
          has_qc: true,
          conditions: {
            fluid: "fc72",
            T_sat_C: 56,
            q_wall_W_cm2: 2,
            flow_rate_mL_hr: 10,
            channel_width_um: 300,
            channel_height_um: 300,
            dt_frame_ms: 0.5,
            flow_direction: "left",
            n_frames_raw: 12,
            n_frames_usable: 11,
            n_frames_event: 10,
            U_ref_m_s: 0.2,
          },
          holdout_frame: 6,
          um_per_px: 1.685,
          notes: null,
          excluded_frames: [],
          exclusions_applied: true,
          conditions_applied: true,
        });
      }
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
      if (u.includes("/loss-history")) {
        // Like a real CLI-era checkpoint: no lr key at all (regression: this
        // used to crash the whole page).
        return json([
          { step: 100, data: 5e-3, vof: 4e-2, div: 4e-3, src: 2e-3, bc: 2e-3 },
          { step: 200, data: 3e-3, vof: 2e-2, div: 3e-3, src: 1e-3, bc: 1e-3 },
        ]);
      }
      if (u.includes("/field?")) {
        return json({
          run_id: "demo_run",
          dataset: null,
          name: "u",
          unit: "mm·s⁻¹",
          t_star: 0.34,
          t_min_star: 0,
          t_max_star: 1.0,
          x_um: [0, 100, 200],
          y_um: [0, 50],
          values: [
            [0, 40, 80],
            [10, 50, 90],
          ],
          vmin: 0,
          vmax: 90,
          fields_available: [
            "alpha",
            "u",
            "v",
            "umag",
            "s",
            "res_vof",
            "res_div",
          ],
        });
      }
      if (u.includes("/interface")) {
        return json({
          run_id: "demo_run",
          domain: { x_um: [0, 1700], y_um: [0, 360], x_pin_um: 120 },
          frames: [
            {
              t_ms: 0,
              contours: [
                [
                  [100, 100],
                  [300, 100],
                  [300, 250],
                ],
              ],
            },
            {
              t_ms: 2.5,
              contours: [
                [
                  [100, 100],
                  [700, 100],
                  [700, 250],
                ],
              ],
            },
          ],
          measured: [
            {
              t_ms: 0,
              contours: [
                [
                  [100, 100],
                  [300, 100],
                  [300, 250],
                ],
              ],
            },
          ],
        });
      }
      if (u.includes("/trajectory")) {
        return json({
          t_ms: [0, 1, 2, 3],
          nose_um: [0, 220, 440, 660],
          area_um2: [0, 50_000, 100_000, 150_000],
          measured: {
            t_ms: [0, 1.5, 3],
            nose_um: [0, 330, 664],
            area_um2: [0, 76_000, 152_000],
          },
        });
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
          transfer_per_frame: joint
            ? { low_t: { "1": 0.91, "2": 0.9, "3": 0.899 } }
            : null,
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
  it("an empty project shows the no-runs state with tabs disabled", async () => {
    const calls = mockApi();
    // The project-scoped listing returns nothing for this project.
    const original = globalThis.fetch;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL, opts?: RequestInit) => {
        const u = String(url);
        if (u.includes("/api/runs?project=")) {
          calls.runListCalls.push(u);
          return json([]);
        }
        return original(url as never, opts);
      }),
    );
    renderAt(`/projects/${PID}/results`);

    expect(await screen.findByText(/no runs yet/i)).toBeInTheDocument();
    expect(
      screen.getByText(/launch the first training run/i),
    ).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /overview/i })).toBeDisabled();
  });

  it("an unknown run id in the URL falls back to the default run", async () => {
    mockApi();
    renderAt(`/projects/${PID}/results/does-not-exist/overview`);

    // First trained run selected instead of a crash or blank page.
    expect(
      await screen.findByRole("option", { name: /demo_run/i }),
    ).toHaveAttribute("aria-selected", "true");
  });

  it("deletes the selected run after confirmation and drops it from the list", async () => {
    mockApi();
    const base = globalThis.fetch as typeof fetch;
    const deleted = new Set<string>();
    let deleteCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL, opts?: RequestInit) => {
        const u = String(url);
        const target = u.match(/\/api\/runs\/([^/?]+)$/);
        if (opts?.method === "DELETE" && target) {
          deleteCalls += 1;
          deleted.add(target[1]);
          return new Response(null, { status: 204 });
        }
        // The listing reflects the deletion (the server would too).
        if (/\/api\/runs(\?|$)/.test(u)) {
          return json(RUNS.filter((r) => !deleted.has(r.id)));
        }
        return base(url as never, opts);
      }),
    );

    renderAt(`/projects/${PID}/results/demo_run`);
    await screen.findByRole("option", { name: /demo_run/i });

    // Open the confirm from the run header, then approve the destructive action.
    fireEvent.click(screen.getByRole("button", { name: /delete run/i }));
    expect(
      await screen.findByText("This cannot be undone."),
    ).toBeInTheDocument();
    fireEvent.click(
      within(screen.getByRole("alertdialog")).getByRole("button", {
        name: /delete run/i,
      }),
    );

    // deleteRun was called, and the list refreshed without the deleted run --
    // selection falls back to another run rather than crashing on the removed id.
    await waitFor(() =>
      expect(screen.queryByRole("option", { name: /demo_run/i })).toBeNull(),
    );
    expect(deleteCalls).toBe(1);
    expect(
      screen.getByRole("option", { name: /second_run/i }),
    ).toBeInTheDocument();
  });

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
    expect(panel).toHaveTextContent(/validation iou/i);
    // Axis B — transfer to the held-out condition.
    await waitFor(() =>
      expect(screen.getByText(/transfer iou/i)).toBeInTheDocument(),
    );
    // In the stat and again in the verdict, which tags its own numbers.
    expect(screen.getAllByText("0.903").length).toBeGreaterThan(0);

    // The verdict narrative states what the numbers argue.
    expect(
      screen.getByText(/learned physics, not footage/i),
    ).toBeInTheDocument();
  });

  it("names what an unmeasured axis would take, instead of printing a dash", async () => {
    mockApi();
    renderAt(`/projects/${PID}/results/demo_run/overview`);

    // demo_run is a single condition with no transfer and no measured nose
    // speed: three of the old scorecard's four slots were em dashes.
    await waitFor(() =>
      expect(
        screen.getByText(/hold one out in the solver/i),
      ).toBeInTheDocument(),
    );
    // And nothing in the scorecard is an em dash: every slot it keeps is a
    // measurement, every slot it drops is an action.
    expect(screen.getByTestId("overview-scorecard")).not.toHaveTextContent("—");
  });

  it("leads the overview with the evidence, and opens the worst frame", async () => {
    mockApi();
    renderAt(`/projects/${PID}/results/demo_run/overview`);

    // The per-frame agreement is the answer to this stage's question, so it is
    // on the landing tab rather than one click away.
    const worst = await screen.findByRole("button", { name: /Frame 6/ });
    fireEvent.click(worst);
    await waitFor(() =>
      expect(
        screen.getByRole("tab", { name: /reconstruction/i }),
      ).toHaveAttribute("aria-selected", "true"),
    );
  });

  it("reconstruction tab plays the interface and charts the kinematics", async () => {
    mockApi();
    renderAt(`/projects/${PID}/results/demo_run/recon`);

    // The continuous player is real (its scrubber exists), with its caption.
    expect(
      await screen.findByRole("heading", {
        name: /continuous reconstruction/i,
      }),
    ).toBeInTheDocument();
    expect(
      await screen.findByText(/continuous pinn interface reconstruction/i),
    ).toBeInTheDocument();

    // Frame matching: per-frame thumbnails with IoU + role, layer toggles.
    const strip = await screen.findByRole("group", { name: /camera frames/i });
    expect(strip).toHaveTextContent("f01 · 0.973");
    expect(strip).toHaveTextContent(/HOLDOUT/);
    const layerGroup = screen.getByRole("group", { name: /overlay layers/i });
    for (const layer of ["camera", "detected", "PINN"]) {
      expect(
        screen.getByRole("button", { name: new RegExp(`^${layer}$`) }),
      ).toBeInTheDocument();
    }
    expect(layerGroup).toBeInTheDocument();

    // Both kinematics quantities chart side by side — never a dual axis.
    expect(
      await screen.findByRole("heading", { name: /nose position/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /vapor area/i }),
    ).toBeInTheDocument();
  });

  it("export tab links artifacts and CSV exports, and copies a citation", async () => {
    mockApi();
    const writeText = vi.fn<(text: string) => Promise<void>>(async () => {});
    Object.assign(navigator, { clipboard: { writeText } });
    renderAt(`/projects/${PID}/results/demo_run/export`);

    expect(
      await screen.findByRole("heading", { name: /run artifacts/i }),
    ).toBeInTheDocument();
    // CSV exports point at the real export endpoints.
    const links = await screen.findAllByRole("link", { name: /download/i });
    expect(
      links.some((a) => a.getAttribute("href")?.includes("/export/iou.csv")),
    ).toBe(true);
    expect(
      links.some((a) => a.getAttribute("href")?.includes("/export/loss.csv")),
    ).toBe(true);

    fireEvent.click(screen.getByRole("button", { name: /copy/i }));
    await waitFor(() => expect(writeText).toHaveBeenCalledOnce());
    expect(writeText.mock.calls[0]?.[0]).toContain("demo_run");

    // The report bundle is declared planned, not faked.
    expect(
      screen.getByText(/planned \(follow-up feature\)/i),
    ).toBeInTheDocument();
  });

  it("compare tab tables the selected runs with the best cell marked", async () => {
    mockApi();
    renderAt(`/projects/${PID}/results/demo_run/compare`);

    // Both trained runs preselected: their columns + the pick chips.
    const picks = await screen.findByRole("group", {
      name: /runs to compare/i,
    });
    expect(picks).toHaveTextContent("demo_run");
    expect(picks).toHaveTextContent("second_run");
    expect(
      await screen.findByRole("columnheader", { name: "second_run" }),
    ).toBeInTheDocument();
    // The joint run's axis-A number lands in its table column (the run
    // browser also shows it, so scope to the table).
    const table = screen.getByRole("table");
    await waitFor(() => expect(table).toHaveTextContent("0.941"));
  });

  it("physics and training tabs read real validation, groups and losses", async () => {
    mockApi();
    renderAt(`/projects/${PID}/results/demo_run/physics`);

    expect(
      await screen.findByRole("heading", { name: /physics validation/i }),
    ).toBeInTheDocument();
    // Wait for the validation payload to land before reading any verdict: the
    // tab states none until it does, and the tolerance is what says it has.
    expect(await screen.findByText(/tol 10 %/)).toBeInTheDocument();
    expect(screen.getByText(/nose-speed agreement/i)).toBeInTheDocument();
    // The method's own open question is stated once, as a note, not failed as
    // a per-run check on every run ever trained.
    expect(
      screen.getByText(/global mass closure is not yet quantitative/i),
    ).toBeInTheDocument();
    // Group tiles from the condition's own groups endpoint.
    expect(await screen.findByText("320")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: /training/i }));
    expect(
      await screen.findByRole("heading", { name: /training diagnostics/i }),
    ).toBeInTheDocument();
    expect(await screen.findByText(/final ·/)).toBeInTheDocument();
    expect(screen.getByText(/vof · transport/)).toBeInTheDocument();
    // No lr recorded → no lr readout, and above all no crash.
    expect(screen.queryByText(/· lr/)).not.toBeInTheDocument();
  });

  it("fields tab evaluates the checkpoint and marks Stage B honestly", async () => {
    mockApi();
    renderAt(`/projects/${PID}/results/demo_run/fields`);

    // Field chips from the checkpoint's own availability; p/T disabled.
    const chips = await screen.findByTestId("field-chips");
    expect(chips).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "p" })).toBeDisabled(),
    );
    expect(screen.getByRole("button", { name: "T" })).toBeDisabled();
    expect(screen.getByText(/stage a only/i)).toBeInTheDocument();

    // The map is real: unit + time readout from the payload.
    expect(screen.getByText("mm·s⁻¹")).toBeInTheDocument();
    expect(screen.getByLabelText(/field time/i)).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /speed .u. field map/i }),
    ).toBeInTheDocument();

    // Residual maps: Stage-A equations render, Stage-B ones say why not.
    expect(
      await screen.findByRole("img", {
        name: /vof · interface transport residual/i,
      }),
    ).toBeInTheDocument();
    expect((await screen.findAllByText(/stage b off/i)).length).toBe(2);
    // And the field animation has a play control.
    expect(
      screen.getByRole("button", { name: /play field animation/i }),
    ).toBeInTheDocument();
  });

  it("agreement tab shows per-condition dots, transfer and the envelope", async () => {
    mockApi();
    renderAt(`/projects/${PID}/results/second_run/agreement`);

    const grid = await screen.findByTestId("agreement-grid");
    expect(grid).toHaveTextContent(/held-out condition/i);
    expect(grid).toHaveTextContent(/mean 0\.958/);

    // Transfer panel: the axis-B number and the credibility argument.
    expect(
      await screen.findByText(/transfer iou · all frames/i),
    ).toBeInTheDocument();
    // In the stat and again in the verdict, which tags its own numbers.
    expect(screen.getAllByText("0.903").length).toBeGreaterThan(0);
    // With one training dataset the envelope is a point: groups where the
    // held-out value differs read extrapolated; the matching one (Pr) reads
    // inside. Both states must be present and text-labeled.
    expect(
      (await screen.findAllByText(/extrapolated/i)).length,
    ).toBeGreaterThan(1);
    expect(screen.getByText(/inside envelope/i)).toBeInTheDocument();
  });

  it("single runs show the transfer empty state", async () => {
    mockApi();
    renderAt(`/projects/${PID}/results/demo_run/agreement`);

    expect(
      await screen.findByText(/no condition was held out/i),
    ).toBeInTheDocument();
    // The single condition still charts, holdout flagged in text.
    expect(await screen.findByTestId("agreement-grid")).toHaveTextContent(
      /mean/,
    );
  });

  it("joint runs state per-condition reconstruction honestly", async () => {
    mockApi();
    renderAt(`/projects/${PID}/results/second_run/recon`);

    expect(
      await screen.findByText(/per-condition reconstruction/i),
    ).toBeInTheDocument();
    // Kinematics ARE per-condition now: the joint run charts the viewing
    // condition's trajectory (served with ?dataset=).
    expect(
      await screen.findByRole("heading", { name: /nose position/i }),
    ).toBeInTheDocument();
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
    // What the run was actually scored on, from the frames the evaluator wrote
    // — the config's val_fraction says how many were meant to be held, not which.
    expect(header).toHaveTextContent(/validated on/i);
    // And where it stands in the ranking it was selected from.
    expect(header).toHaveTextContent(/rank \d+ of \d+/);

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

  it("a run auto-named after its series shows the series' current label", async () => {
    mockApi();
    renderAt(`/projects/${PID}/results/highest_t/overview`);

    // The run browser row and the header lead with the display label…
    const row = await screen.findByRole("option", { name: /series-1/i });
    expect(row).toHaveAttribute("aria-selected", "true");
    const header = await screen.findByTestId("run-header");
    expect(header).toHaveTextContent("series-1");
    expect(header).not.toHaveTextContent("outputs/highest_t");
    // …while the immutable id stays discoverable as hover provenance.
    expect(
      within(header).getByTitle(/stored in outputs\/highest_t/i),
    ).toBeInTheDocument();
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
