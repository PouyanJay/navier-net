import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ProjectSummary } from "../src/lib/api";
import { DatasetsView } from "../src/views/DatasetsView";

/** Owns project state like App does, so attach flows re-render the view. */
function Harness({ onProjectChanged = vi.fn() }: { onProjectChanged?: (p: ProjectSummary) => void }) {
  const [project, setProject] = useState<ProjectSummary>(PROJECT);
  return (
    <DatasetsView
      project={project}
      onProjectChanged={(updated) => {
        setProject(updated);
        onProjectChanged(updated);
      }}
    />
  );
}

const PROJECT = {
  id: "a".repeat(32),
  name: "Microchannel FC-72",
  description: "",
  datasets: ["sample"],
  created_at: "2026-07-24T00:00:00+00:00",
};

const DETAIL = {
  id: "sample",
  n_frames: 3,
  processed: false,
  conditions_set: false,
  frame_px: [16, 12] as [number, number],
  has_qc: false,
  holdout_frame: 6,
  um_per_px: null,
  notes: "Frames 1-10 continuous growth.",
  conditions: {
    fluid: "FC-72",
    T_sat_C: 56.6,
    q_wall_W_cm2: 2.0,
    flow_rate_mL_hr: 5.0,
    channel_width_um: 300,
    channel_height_um: 150,
    dt_frame_ms: 0.5,
    flow_direction: "right_to_left",
    n_frames_raw: 12,
    n_frames_usable: 11,
    n_frames_event: 10,
    U_ref_m_s: 0.2,
  },
};
const GROUPS = {
  Re: 215.5,
  We: 2.302,
  Ca: 0.01068,
  Pr: 9.411,
  Bond: 0.0746,
  hele_shaw: 0.2228,
  bretherton_film_um: 4.875,
  Dh_um: 200,
};
const QC = {
  dataset: "sample",
  n_frames_event: 3,
  kinematics: {
    t_ms: [0, 0.5, 1.0],
    length_um: [600, 800, 1000],
    fit_slope_mm_s: 180,
    fit_intercept_um: 610,
  },
  interface: {
    x_pin_star: 0.1,
    x_range: [0, 5.7] as [number, number],
    y_range: [0, 1] as [number, number],
    frames: [{ index: 0, t_ms: 0, contours: [[[0.2, 0.3], [0.4, 0.5]]] }],
  },
  sdf: {
    frame_index: 1,
    t_ms: 0.5,
    x_range: [0, 5.7] as [number, number],
    y_range: [0, 1] as [number, number],
    values: [
      [-0.5, 0.5],
      [0.2, 0.9],
    ],
  },
};
const IDLE = { dataset: "sample", state: "idle", message: null, has_qc: false };

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200 });
}

interface Calls {
  upload: string[];
  startPreprocess: string[];
  conditionPatches: Record<string, unknown>[];
  projectPatches: Record<string, unknown>[];
}

function mockApi({ processed = false, holdout = 6 } = {}): Calls {
  const calls: Calls = {
    upload: [],
    startPreprocess: [],
    conditionPatches: [],
    projectPatches: [],
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL, opts?: RequestInit) => {
      const u = String(url);
      const post = opts?.method === "POST";
      if (u.endsWith("/api/datasets")) {
        const rows = [
          { id: "sample", n_frames: 3, processed, conditions_set: false, frame_px: [16, 12], dt_frame_ms: 0.5 },
        ];
        if (calls.upload.includes("mid_T")) {
          rows.push({
            id: "mid_T",
            n_frames: 5,
            processed: false,
            conditions_set: false,
            frame_px: [16, 12],
            dt_frame_ms: 0.5,
          });
        }
        return json(rows);
      }
      if (u.endsWith("/api/runs")) return json([]);
      if (u.endsWith("/groups")) return json(GROUPS);
      if (u.endsWith("/qc-data")) {
        return processed ? json(QC) : new Response("{}", { status: 404 });
      }
      if (u.endsWith("/conditions") && opts?.method === "PATCH") {
        const body = JSON.parse(String(opts.body));
        calls.conditionPatches.push(body);
        return json({
          conditions: { ...DETAIL.conditions, ...body },
          groups: { ...GROUPS, Re: 431.0 },
        });
      }
      const upload = u.match(/\/api\/datasets\/([^/]+)\/upload$/);
      if (upload && post) {
        calls.upload.push(upload[1]);
        return json({
          id: upload[1],
          n_frames: 5,
          processed: false,
          conditions_set: false,
          frame_px: [16, 12],
        });
      }
      const pre = u.match(/\/api\/datasets\/([^/]+)\/preprocess$/);
      if (pre) {
        if (post) {
          calls.startPreprocess.push(pre[1]);
          return json({ ...IDLE, dataset: pre[1], state: "running" });
        }
        // Once started, the job settles immediately so polling flows finish.
        const done = calls.startPreprocess.includes(pre[1]);
        return json({ ...IDLE, dataset: pre[1], state: done ? "done" : "idle" });
      }
      const patch = u.match(/\/api\/projects\/([0-9a-f]{32})$/);
      if (patch && opts?.method === "PATCH") {
        const body = JSON.parse(String(opts.body));
        calls.projectPatches.push(body);
        return json({ ...PROJECT, ...body });
      }
      if (/\/api\/datasets\/[^/]+$/.test(u)) {
        return json({ ...DETAIL, processed, holdout_frame: holdout });
      }
      return new Response("not found", { status: 404 });
    }),
  );
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

const noop = vi.fn();

describe("DatasetsView", () => {
  it("shows the series library, frame strip, conditions, and groups", async () => {
    mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);

    expect(await screen.findByText("Series library")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /sample/ })).toBeInTheDocument();
    expect(await screen.findByText(/Image sequence — sample/)).toBeInTheDocument();
    expect(screen.getAllByAltText(/frame \d+/).length).toBe(3);
    // Conditions are editable inputs with unit suffixes.
    expect(screen.getByLabelText(/Frame interval/)).toHaveValue(0.5);
    expect(screen.getByLabelText(/Reference velocity/)).toHaveValue(0.2);
    // Groups render as tiles.
    expect(await screen.findByText("215.5")).toBeInTheDocument();
  });

  it("saves an edited condition on blur and applies the recomputed groups", async () => {
    const calls = mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);

    const dt = await screen.findByLabelText(/Frame interval/);
    fireEvent.change(dt, { target: { value: "0.4" } });
    fireEvent.blur(dt);

    await waitFor(() => expect(calls.conditionPatches).toEqual([{ dt_frame_ms: 0.4 }]));
    expect(await screen.findByText("431.0")).toBeInTheDocument(); // live Re
  });

  it("uploads a new series through the modal and preprocesses it", async () => {
    const calls = mockApi();
    const onProjectChanged = vi.fn();
    render(<Harness onProjectChanged={onProjectChanged} />);

    fireEvent.click(await screen.findByRole("button", { name: /Upload new series/ }));
    const dialog = await screen.findByRole("dialog", { name: "Upload new series" });
    expect(dialog).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Series name"), { target: { value: "mid_T" } });
    const file = new File([new Uint8Array([0x49, 0x49, 0x2a, 0x00])], "1.tif", {
      type: "image/tiff",
    });
    fireEvent.change(screen.getByLabelText(/Image sequence/), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "Upload & preprocess" }));

    await waitFor(() =>
      expect(calls.projectPatches).toEqual([{ datasets: ["sample", "mid_T"] }]),
    );
    expect(calls.upload).toEqual(["mid_T"]);
    // The pipeline runs right after the upload…
    await waitFor(() => expect(calls.startPreprocess).toEqual(["mid_T"]));
    // …and the modal closes on completion (polling saw state: done).
    await waitFor(
      () => expect(screen.queryByRole("dialog", { name: "Upload new series" })).toBeNull(),
      { timeout: 4000 },
    );
    expect(onProjectChanged).toHaveBeenCalledWith(
      expect.objectContaining({ datasets: ["sample", "mid_T"] }),
    );
    // The refreshed library shows the new series without a reload.
    expect(await screen.findByRole("button", { name: /mid_T/ })).toBeInTheDocument();
  });

  it("starts preprocessing when the button is clicked", async () => {
    const calls = mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    const button = await screen.findByRole("button", { name: /Run preprocessing/ });

    fireEvent.click(button);
    await waitFor(() => expect(calls.startPreprocess).toEqual(["sample"]));
  });

  it("shows the interactive QC checks once the series is preprocessed", async () => {
    mockApi({ processed: true });
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);

    expect(await screen.findByText("Preprocessing QC")).toBeInTheDocument();
    // Three checks behind one switch; kinematics is the default tab.
    expect(screen.getByRole("tab", { name: /Growth kinematics/ })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(
      screen.getByRole("img", { name: /Bubble length per frame/ }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: /Interface evolution/ }));
    expect(screen.getByRole("img", { name: /Interface contours/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: /Signed distance/ }));
    expect(screen.getByRole("img", { name: /Signed distance field/ })).toBeInTheDocument();
  });

  it("marks the holdout frame in the strip", async () => {
    mockApi({ holdout: 2 });
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);

    expect(await screen.findByAltText("frame 2 (holdout)")).toBeInTheDocument();
    expect(screen.getByAltText("frame 1")).toBeInTheDocument();
  });
});

describe("DatasetsView with several series", () => {
  it("scopes the library to the project's series and switches between them", async () => {
    mockApi();
    const twoSeries = { ...PROJECT, datasets: ["sample", "mid_T"] };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/api/datasets")) {
          return json([
            { id: "sample", n_frames: 3, processed: true, conditions_set: true, frame_px: [16, 12], dt_frame_ms: 0.5 },
            { id: "mid_T", n_frames: 2, processed: false, conditions_set: false, frame_px: [16, 12], dt_frame_ms: 0.5 },
            { id: "foreign", n_frames: 9, processed: false, conditions_set: false, frame_px: null, dt_frame_ms: null },
          ]);
        }
        if (u.endsWith("/api/runs")) return json([]);
        if (u.endsWith("/groups")) return json(GROUPS);
        if (u.endsWith("/qc-data")) return new Response("{}", { status: 404 });
        if (u.endsWith("/preprocess")) return json(IDLE);
        const detail = u.match(/\/api\/datasets\/([^/]+)$/);
        if (detail) return json({ ...DETAIL, id: detail[1], processed: false });
        return new Response("not found", { status: 404 });
      }),
    );
    render(<DatasetsView project={twoSeries} onProjectChanged={noop} />);

    // Both project series are listed; the foreign dataset is not.
    expect(await screen.findByRole("button", { name: /sample/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /mid_T/ })).toBeInTheDocument();
    expect(screen.queryByText("foreign")).not.toBeInTheDocument();
    // Per-series chips are independent: processed+conditions vs bare upload.
    expect(screen.getByText("tensors ready")).toBeInTheDocument();
    expect(screen.getByText("needs conditions")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /mid_T/ }));
    expect(await screen.findByText(/Image sequence — mid_T/)).toBeInTheDocument();
  });
});

describe("NewSeriesModal failure paths", () => {
  function openFormAndFill(name = "mid_T") {
    fireEvent.click(screen.getByRole("button", { name: /Upload new series/ }));
    fireEvent.change(screen.getByLabelText("Series name"), { target: { value: name } });
    const file = new File([new Uint8Array([0x49, 0x49, 0x2a, 0x00])], "1.tif", {
      type: "image/tiff",
    });
    fireEvent.change(screen.getByLabelText(/Image sequence/), { target: { files: [file] } });
  }

  it("reports a failed upload and never attempts the attach", async () => {
    const calls = mockApi();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const real = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (url: string | URL, opts?: RequestInit) => {
      if (String(url).endsWith("/upload")) {
        return new Response(JSON.stringify({ detail: "too many frames" }), { status: 400 });
      }
      return real(url, opts);
    });
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText("Series library");
    openFormAndFill();
    fireEvent.click(screen.getByRole("button", { name: "Upload & preprocess" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/Upload failed: too many/);
    expect(calls.projectPatches).toEqual([]);
  });

  it("distinguishes an attach failure from an upload failure", async () => {
    const calls = mockApi();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const real = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (url: string | URL, opts?: RequestInit) => {
      if (/\/api\/projects\//.test(String(url)) && opts?.method === "PATCH") {
        return new Response(JSON.stringify({ detail: "disk full" }), { status: 500 });
      }
      return real(url, opts);
    });
    const onProjectChanged = vi.fn();
    render(<DatasetsView project={PROJECT} onProjectChanged={onProjectChanged} />);
    await screen.findByText("Series library");
    openFormAndFill();
    fireEvent.click(screen.getByRole("button", { name: "Upload & preprocess" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Uploaded, but linking the series failed: disk full/,
    );
    expect(calls.upload).toEqual(["mid_T"]); // frames did land on disk
    expect(onProjectChanged).not.toHaveBeenCalled();
  });

  it("rejects a duplicate series name before any request", async () => {
    mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText("Series library");
    openFormAndFill("sample"); // already in the project

    expect(
      screen.getByText(/A series with this name already exists/),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upload & preprocess" })).toBeDisabled();
  });
});

describe("ConditionsPanel behavior", () => {
  it("surfaces the API's rejection of a bad value", async () => {
    mockApi();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const real = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (url: string | URL, opts?: RequestInit) => {
      if (String(url).endsWith("/conditions")) {
        return new Response(
          JSON.stringify({ detail: "dt_frame_ms must be a positive number" }),
          { status: 400 },
        );
      }
      return real(url, opts);
    });
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);

    const dt = await screen.findByLabelText(/Frame interval/);
    fireEvent.change(dt, { target: { value: "3" } });
    fireEvent.blur(dt);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Frame interval: dt_frame_ms must be a positive number/,
    );
  });

  it("does not PATCH when the value is unchanged", async () => {
    const calls = mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);

    const dt = await screen.findByLabelText(/Frame interval/);
    fireEvent.change(dt, { target: { value: "0.5" } }); // same as current
    fireEvent.blur(dt);

    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(calls.conditionPatches).toEqual([]);
  });
});

describe("DatasetsView preprocess polling", () => {
  it("polls a running job to completion and refreshes the series", async () => {
    let started = false;
    let polls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL, opts?: RequestInit) => {
        const u = String(url);
        const post = opts?.method === "POST";
        const done = polls >= 2;
        if (u.endsWith("/api/datasets")) {
          return json([
            { id: "sample", n_frames: 3, processed: done, conditions_set: false, frame_px: [16, 12], dt_frame_ms: 0.5 },
          ]);
        }
        if (u.endsWith("/api/runs")) return json([]);
        if (u.endsWith("/groups")) return json(GROUPS);
        if (u.endsWith("/qc-data")) {
          return done ? json(QC) : new Response("{}", { status: 404 });
        }
        if (u.endsWith("/preprocess")) {
          if (post) {
            started = true;
            return json({ ...IDLE, state: "running" });
          }
          if (!started) return json(IDLE);
          polls += 1;
          return json({ ...IDLE, state: polls >= 2 ? "done" : "running" });
        }
        if (/\/api\/datasets\/[^/]+$/.test(u)) return json({ ...DETAIL, processed: done });
        return new Response("not found", { status: 404 });
      }),
    );

    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    fireEvent.click(await screen.findByRole("button", { name: /Run preprocessing/ }));

    // Real timers: the hook polls every 1s until done, then refreshes —
    // proven by the QC panel appearing once the series reports processed.
    await waitFor(
      () => expect(screen.getByText("Preprocessing QC")).toBeInTheDocument(),
      { timeout: 5000 },
    );
    expect(polls).toBeGreaterThanOrEqual(2);
  });
});

describe("DatasetsView failure paths", () => {
  it("shows an error instead of an endless spinner when the API is down", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/Could not load datasets/);
    expect(screen.queryByText("Loading datasets…")).not.toBeInTheDocument();
  });
});
