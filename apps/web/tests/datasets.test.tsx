import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ProjectSummary } from "../src/lib/api";
import { DatasetsView } from "../src/views/DatasetsView";

/** Owns project state like App does, so attach flows re-render the view. */
function Harness({
  onProjectChanged = vi.fn(),
}: {
  onProjectChanged?: (p: ProjectSummary) => void;
}) {
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
  excluded_frames: [] as number[],
  exclusions_applied: false,
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

const FLUIDS = [
  {
    id: "fc72",
    name: "FC-72",
    T_sat_C: 56.6,
    rho_l: 1601.6,
    rho_v: 13.39,
    mu_l: 4.46e-4,
    mu_v: 1.2e-5,
    k_l: 0.0522,
    k_v: 0.0124,
    cp_l: 1101.5,
    cp_v: 885,
    sigma: 8.35e-3,
    h_lv: 84500,
  },
  {
    id: "water",
    name: "Water",
    T_sat_C: 100,
    rho_l: 958.4,
    rho_v: 0.5982,
    mu_l: 2.82e-4,
    mu_v: 1.23e-5,
    k_l: 0.679,
    k_v: 0.0251,
    cp_l: 4217,
    cp_v: 2080,
    sigma: 5.89e-2,
    h_lv: 2.257e6,
  },
];
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
    l_ref_um: 300,
    y_roi_top: 2,
    frames: [
      {
        index: 0,
        camera_frame: 1,
        t_ms: 0,
        rings: [
          [
            [0.2, 0.3],
            [0.4, 0.3],
            [0.4, 0.5],
            [0.2, 0.5],
            [0.2, 0.3],
          ],
        ],
      },
    ],
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
  exclusionPuts: number[][];
}

function mockApi({
  processed = false,
  holdout = 6,
  excluded = [] as number[],
  exclusionStatus = 200,
  umPerPx = null as number | null,
} = {}): Calls {
  const calls: Calls = {
    upload: [],
    startPreprocess: [],
    conditionPatches: [],
    projectPatches: [],
    exclusionPuts: [],
  };
  // The endpoint replaces the set, so the mock keeps the series' current one.
  let excludedFrames = excluded;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL, opts?: RequestInit) => {
      const u = String(url);
      const post = opts?.method === "POST";
      if (u.endsWith("/api/datasets")) {
        const rows = [
          {
            id: "sample",
            n_frames: 3,
            processed,
            conditions_set: false,
            frame_px: [16, 12],
            dt_frame_ms: 0.5,
          },
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
      if (u.endsWith("/api/fluids")) return json(FLUIDS);
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
        return json({
          ...IDLE,
          dataset: pre[1],
          state: done ? "done" : "idle",
        });
      }
      const patch = u.match(/\/api\/projects\/([0-9a-f]{32})$/);
      if (patch && opts?.method === "PATCH") {
        const body = JSON.parse(String(opts.body));
        calls.projectPatches.push(body);
        return json({ ...PROJECT, ...body });
      }
      if (u.endsWith("/excluded-frames") && opts?.method === "PUT") {
        const body = JSON.parse(String(opts.body)) as {
          excluded_frames: number[];
        };
        calls.exclusionPuts.push(body.excluded_frames);
        if (exclusionStatus !== 200) {
          return new Response(
            JSON.stringify({ detail: "frame 9 is outside the sequence" }),
            {
              status: exclusionStatus,
            },
          );
        }
        excludedFrames = body.excluded_frames;
        return json({
          ...DETAIL,
          processed,
          holdout_frame: holdout,
          excluded_frames: excludedFrames,
        });
      }
      if (/\/api\/datasets\/[^/]+$/.test(u)) {
        return json({
          ...DETAIL,
          processed,
          holdout_frame: holdout,
          excluded_frames: excludedFrames,
          um_per_px: umPerPx ?? DETAIL.um_per_px,
        });
      }
      return new Response("not found", { status: 404 });
    }),
  );
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

const noop = vi.fn();

describe("DatasetsView", () => {
  it("shows the series library, frame strip, QC, and groups", async () => {
    mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);

    expect(await screen.findByText("Series library")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /sample/ })).toBeInTheDocument();
    expect(
      await screen.findByText(/Image sequence · sample/),
    ).toBeInTheDocument();
    expect(screen.getAllByAltText(/Frame \d+ preview/).length).toBe(3);
    // QC is a sub-section of the same card; before preprocessing it prompts.
    expect(screen.getByText("Preprocessing QC")).toBeInTheDocument();
    expect(
      screen.getByText(/Run preprocessing to compute the QC checks/),
    ).toBeInTheDocument();
    // Operating conditions are set in the upload modal, not edited here.
    expect(screen.queryByLabelText(/Flow rate/)).toBeNull();
    expect(screen.queryByLabelText(/Frame interval/)).toBeNull();
    // Groups render as tiles.
    expect(await screen.findByText("215.5")).toBeInTheDocument();
  });

  it("shows the selected series' inputs read-only in the library card", async () => {
    mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);

    // The inputs set in the upload modal are surfaced for the selected series.
    const inputs = within(
      await screen.findByRole("region", { name: /sample conditions/ }),
    );
    expect(inputs.getByText("Working fluid")).toBeInTheDocument();
    expect(inputs.getByText("FC-72")).toBeInTheDocument();
    // The unit rides in the label, so the value column stays clean numbers.
    expect(inputs.getByText("Channel height (µm)")).toBeInTheDocument();
    expect(inputs.getByText("150")).toBeInTheDocument();
    expect(inputs.getByText("Reference velocity (m·s⁻¹)")).toBeInTheDocument();
  });

  it("defines a dimensionless group and reads back its value on selection", async () => {
    mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);

    // Capillary number is the default selection: it explains the group and the
    // regime its value puts the run in.
    const ca = within(
      await screen.findByRole("region", { name: /Capillary number definition/ }),
    );
    expect(ca.getByText(/Viscous drag ÷ surface tension/)).toBeInTheDocument();
    expect(ca.getByText(/Bretherton film/)).toBeInTheDocument();

    // Selecting the Reynolds tile switches the definition and read-back.
    fireEvent.click(screen.getByRole("button", { name: /RE 215\.5/ }));
    const re = within(
      await screen.findByRole("region", { name: /Reynolds number definition/ }),
    );
    expect(re.getByText(/Inertial forces ÷ viscous forces/)).toBeInTheDocument();
    expect(re.getByText(/Laminar/)).toBeInTheDocument();
  });

  it("uploads a new series through the modal and preprocesses it", async () => {
    const calls = mockApi();
    const onProjectChanged = vi.fn();
    render(<Harness onProjectChanged={onProjectChanged} />);

    fireEvent.click(
      await screen.findByRole("button", { name: /Upload new series/ }),
    );
    const dialog = await screen.findByRole("dialog", {
      name: "Upload new series",
    });
    expect(dialog).toBeInTheDocument();

    const form = within(dialog);
    fireEvent.change(form.getByLabelText("Series name"), {
      target: { value: "mid_T" },
    });
    const file = new File([new Uint8Array([0x49, 0x49, 0x2a, 0x00])], "1.tif", {
      type: "image/tiff",
    });
    fireEvent.change(form.getByLabelText(/Image sequence/), {
      target: { files: [file] },
    });
    fireEvent.change(form.getByLabelText(/Frame interval/), {
      target: { value: "0.25" },
    });
    fireEvent.change(form.getByLabelText(/Channel width/), {
      target: { value: "400" },
    });
    fireEvent.change(form.getByLabelText(/Channel height/), {
      target: { value: "150" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Upload & preprocess" }),
    );

    await waitFor(() =>
      expect(calls.projectPatches).toEqual([{ datasets: ["sample", "mid_T"] }]),
    );
    expect(calls.upload).toEqual(["mid_T"]);
    // The pipeline runs right after the upload…
    await waitFor(() => expect(calls.startPreprocess).toEqual(["mid_T"]));
    // …and the modal closes on completion (polling saw state: done).
    await waitFor(
      () =>
        expect(
          screen.queryByRole("dialog", { name: "Upload new series" }),
        ).toBeNull(),
      { timeout: 4000 },
    );
    expect(onProjectChanged).toHaveBeenCalledWith(
      expect.objectContaining({ datasets: ["sample", "mid_T"] }),
    );
    // The refreshed library shows the new series without a reload.
    expect(
      await screen.findByRole("button", { name: /mid_T/ }),
    ).toBeInTheDocument();
  });

  it("starts preprocessing when the button is clicked", async () => {
    const calls = mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    const button = await screen.findByRole("button", {
      name: /Run preprocessing/,
    });

    fireEvent.click(button);
    await waitFor(() => expect(calls.startPreprocess).toEqual(["sample"]));
  });

  it("shows the interactive QC checks once the series is preprocessed", async () => {
    mockApi({ processed: true });
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);

    // "Preprocessing QC" shows immediately; the tabs appear once tensors load.
    const kinematics = await screen.findByRole("tab", {
      name: /Growth kinematics/,
    });
    expect(screen.getByText("Preprocessing QC")).toBeInTheDocument();
    // Three checks behind one switch; kinematics is the default tab.
    expect(kinematics).toHaveAttribute("aria-selected", "true");
    expect(
      screen.getByRole("img", {
        name: /Bubble length in micrometres against time/,
      }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: /Interface evolution/ }));
    expect(
      screen.getByRole("img", { name: /Bubble outline for \d+ frames/ }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: /Signed distance/ }));
    expect(
      screen.getByRole("img", { name: /Signed distance field/ }),
    ).toBeInTheDocument();
  });

  it("marks the holdout frame in the strip", async () => {
    mockApi({ holdout: 2 });
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);

    // The tile is a toggle; its accessible name carries the state, so the
    // holdout is announced rather than only being amber.
    expect(
      await screen.findByRole("button", { name: /Frame 2,.*holdout frame/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Frame 1, included in training$/ }),
    );
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
            {
              id: "sample",
              n_frames: 3,
              processed: true,
              conditions_set: true,
              frame_px: [16, 12],
              dt_frame_ms: 0.5,
            },
            {
              id: "mid_T",
              n_frames: 2,
              processed: false,
              conditions_set: false,
              frame_px: [16, 12],
              dt_frame_ms: 0.5,
            },
            {
              id: "foreign",
              n_frames: 9,
              processed: false,
              conditions_set: false,
              frame_px: null,
              dt_frame_ms: null,
            },
          ]);
        }
        if (u.endsWith("/api/runs")) return json([]);
        if (u.endsWith("/api/fluids")) return json(FLUIDS);
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
    expect(
      await screen.findByRole("button", { name: /sample/ }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /mid_T/ })).toBeInTheDocument();
    expect(screen.queryByText("foreign")).not.toBeInTheDocument();
    // Per-series chips are independent: processed+conditions vs bare upload.
    expect(screen.getByText("tensors ready")).toBeInTheDocument();
    expect(screen.getByText("needs conditions")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /mid_T/ }));
    expect(
      await screen.findByText(/Image sequence · mid_T/),
    ).toBeInTheDocument();
  });
});

describe("NewSeriesModal failure paths", () => {
  async function openFormAndFill(name = "mid_T") {
    fireEvent.click(screen.getByRole("button", { name: /Upload new series/ }));
    // Scope to the dialog so the query is unambiguous.
    const form = within(
      screen.getByRole("dialog", { name: "Upload new series" }),
    );
    fireEvent.change(form.getByLabelText("Series name"), {
      target: { value: name },
    });
    const file = new File([new Uint8Array([0x49, 0x49, 0x2a, 0x00])], "1.tif", {
      type: "image/tiff",
    });
    fireEvent.change(form.getByLabelText(/Image sequence/), {
      target: { files: [file] },
    });
    fireEvent.change(form.getByLabelText(/Frame interval/), {
      target: { value: "0.25" },
    });
    fireEvent.change(form.getByLabelText(/Channel width/), {
      target: { value: "400" },
    });
    fireEvent.change(form.getByLabelText(/Channel height/), {
      target: { value: "150" },
    });
    // Submit is gated on the fluid catalogue; wait for it to load (the selector
    // enables once the fetch resolves), independent of name validity.
    await waitFor(() =>
      expect(form.getByLabelText("Working fluid")).toBeEnabled(),
    );
  }

  it("reports a failed upload and never attempts the attach", async () => {
    const calls = mockApi();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const real = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(
      async (url: string | URL, opts?: RequestInit) => {
        if (String(url).endsWith("/upload")) {
          return new Response(JSON.stringify({ detail: "too many frames" }), {
            status: 400,
          });
        }
        return real(url, opts);
      },
    );
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText("Series library");
    await openFormAndFill();
    fireEvent.click(
      screen.getByRole("button", { name: "Upload & preprocess" }),
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Upload failed: too many/,
    );
    expect(calls.projectPatches).toEqual([]);
  });

  it("distinguishes an attach failure from an upload failure", async () => {
    const calls = mockApi();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const real = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(
      async (url: string | URL, opts?: RequestInit) => {
        if (/\/api\/projects\//.test(String(url)) && opts?.method === "PATCH") {
          return new Response(JSON.stringify({ detail: "disk full" }), {
            status: 500,
          });
        }
        return real(url, opts);
      },
    );
    const onProjectChanged = vi.fn();
    render(
      <DatasetsView project={PROJECT} onProjectChanged={onProjectChanged} />,
    );
    await screen.findByText("Series library");
    await openFormAndFill();
    fireEvent.click(
      screen.getByRole("button", { name: "Upload & preprocess" }),
    );

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
    await openFormAndFill("sample"); // already in the project

    expect(
      screen.getByText(/A series with this name already exists/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Upload & preprocess" }),
    ).toBeDisabled();
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
            {
              id: "sample",
              n_frames: 3,
              processed: done,
              conditions_set: false,
              frame_px: [16, 12],
              dt_frame_ms: 0.5,
            },
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
        if (/\/api\/datasets\/[^/]+$/.test(u))
          return json({ ...DETAIL, processed: done });
        return new Response("not found", { status: 404 });
      }),
    );

    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    fireEvent.click(
      await screen.findByRole("button", { name: /Run preprocessing/ }),
    );

    // Real timers: the hook polls every 1s until done, then refreshes; proven by
    // the QC charts (their tabs) appearing once the series reports processed.
    await waitFor(
      () =>
        expect(
          screen.getByRole("tab", { name: /Growth kinematics/ }),
        ).toBeInTheDocument(),
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

describe("frame exclusion", () => {
  /** The tile toggle for a 1-based camera frame. */
  const tile = (n: number) =>
    screen.getByRole("button", { name: new RegExp(`^Frame ${n},`) });

  it("excludes a frame on click and sends the whole new set", async () => {
    const calls = mockApi({ excluded: [1] });
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText(/Image sequence · sample/);

    fireEvent.click(tile(3));

    // The set is replaced, not appended to; frame 1 must survive the round trip.
    await waitFor(() => expect(calls.exclusionPuts).toEqual([[1, 3]]));
    await waitFor(() =>
      expect(tile(3)).toHaveAttribute("aria-pressed", "true"),
    );
    expect(await screen.findByText(/2 excluded \(1, 3\)/)).toBeInTheDocument();
  });

  it("includes an excluded frame again on a second click", async () => {
    const calls = mockApi({ excluded: [3] });
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText(/Image sequence · sample/);
    await waitFor(() =>
      expect(tile(3)).toHaveAttribute("aria-pressed", "true"),
    );

    fireEvent.click(tile(3));

    await waitFor(() => expect(calls.exclusionPuts).toEqual([[]]));
    await waitFor(() =>
      expect(tile(3)).toHaveAttribute("aria-pressed", "false"),
    );
  });

  it("refuses to exclude the holdout frame without calling the API", async () => {
    const calls = mockApi({ holdout: 2 });
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText(/Image sequence · sample/);

    fireEvent.click(tile(2));

    expect(
      await screen.findByText(/only unsupervised check/),
    ).toBeInTheDocument();
    expect(calls.exclusionPuts).toEqual([]);
    expect(tile(2)).toHaveAttribute("aria-pressed", "false");
  });

  it("reverts the tile and surfaces the reason when the API rejects the edit", async () => {
    mockApi({ exclusionStatus: 400 });
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText(/Image sequence · sample/);

    fireEvent.click(tile(3));

    // The border must never claim an exclusion the server did not accept.
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /outside the sequence/,
    );
    await waitFor(() =>
      expect(tile(3)).toHaveAttribute("aria-pressed", "false"),
    );
  });

  it("warns that saved exclusions need a preprocessing re-run", async () => {
    mockApi({ processed: true, excluded: [3] });
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);

    // DETAIL carries exclusions_applied: false, so the tensors are out of date.
    expect(
      await screen.findByText(
        /excluded frames have changed since the tensors were built/,
      ),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Re-run" })).toBeInTheDocument();
  });

  it("opens the frame full size on double-click without toggling it", async () => {
    const calls = mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText(/Image sequence · sample/);

    fireEvent.doubleClick(tile(2));

    const dialog = await screen.findByRole("dialog", {
      name: /Frame 2 of sample/,
    });
    expect(dialog).toBeInTheDocument();
    // The two clicks a double-click also delivers must not have excluded it.
    await waitFor(() => expect(calls.exclusionPuts).toEqual([]));
  });

  it("steps frames with the arrow keys and closes on Escape", async () => {
    mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText(/Image sequence · sample/);
    fireEvent.doubleClick(tile(2));
    await screen.findByRole("dialog", { name: /Frame 2 of sample/ });

    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(
      await screen.findByRole("dialog", { name: /Frame 3 of sample/ }),
    ).toBeInTheDocument();

    // Frame 3 is the last of the 3-frame series; the axis must not run past it.
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(
      await screen.findByRole("dialog", { name: /Frame 3 of sample/ }),
    ).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
  });

  it("offers an enlarge control so the lightbox is reachable by keyboard", async () => {
    mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText(/Image sequence · sample/);

    fireEvent.click(screen.getByRole("button", { name: "Enlarge frame 2" }));

    expect(
      await screen.findByRole("dialog", { name: /Frame 2 of sample/ }),
    ).toBeInTheDocument();
  });
});

describe("frame strip scrolling", () => {
  /** jsdom lays nothing out, so the strip has to be told it overflows. */
  function makeScrollable(width = 200, content = 1200) {
    const strip = document.querySelector<HTMLElement>(".strip");
    if (!strip) throw new Error("no frame strip rendered");
    Object.defineProperty(strip, "clientWidth", {
      value: width,
      configurable: true,
    });
    Object.defineProperty(strip, "scrollWidth", {
      value: content,
      configurable: true,
    });
    fireEvent.scroll(strip); // re-measure through the component's own listener
    return strip;
  }

  it("shows a scrollbar only when the frames overflow", async () => {
    mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText(/Image sequence · sample/);

    // Nothing overflows by default in jsdom, so nothing to scroll.
    expect(screen.queryByRole("scrollbar")).not.toBeInTheDocument();

    makeScrollable();

    const bar = await screen.findByRole("scrollbar", { name: /frame strip/i });
    expect(bar).toHaveAttribute("aria-orientation", "horizontal");
    expect(bar).toHaveAttribute("aria-valuenow", "0");
  });

  it("scrolls the strip horizontally from a plain vertical mouse wheel", async () => {
    mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText(/Image sequence · sample/);
    const strip = makeScrollable();

    // A wheel only emits deltaY; without the mapping the strip never moves and
    // the page scrolls past it instead.
    fireEvent.wheel(strip, { deltaY: 120, deltaX: 0 });

    expect(strip.scrollLeft).toBe(120);
  });

  it("hands the wheel back to the page at the end of the strip", async () => {
    mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText(/Image sequence · sample/);
    const strip = makeScrollable();
    strip.scrollLeft = 1000; // scrollWidth - clientWidth

    const event = new WheelEvent("wheel", {
      deltaY: 120,
      cancelable: true,
      bubbles: true,
    });
    strip.dispatchEvent(event);

    // Not swallowed, so the page scrolls rather than the pointer being trapped.
    expect(event.defaultPrevented).toBe(false);
    expect(strip.scrollLeft).toBe(1000);
  });

  it("moves the strip with the arrow keys on the scrollbar", async () => {
    mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText(/Image sequence · sample/);
    const strip = makeScrollable();
    strip.scrollBy = vi.fn(({ left }: ScrollToOptions = {}) => {
      strip.scrollLeft += left ?? 0;
    }) as never;

    fireEvent.keyDown(await screen.findByRole("scrollbar"), {
      key: "ArrowRight",
    });

    expect(strip.scrollLeft).toBeGreaterThan(0);
  });
});

describe("new series conditions", () => {
  const tiff = () =>
    new File([new Uint8Array([0x49, 0x49, 0x2a, 0x00])], "1.tif", {
      type: "image/tiff",
    });

  /** Queries scoped to the dialog so condition lookups are unambiguous. */
  async function openModal() {
    fireEvent.click(
      await screen.findByRole("button", { name: /Upload new series/ }),
    );
    const dialog = await screen.findByRole("dialog", {
      name: "Upload new series",
    });
    const form = within(dialog);
    fireEvent.change(form.getByLabelText("Series name"), {
      target: { value: "mid_T" },
    });
    fireEvent.change(form.getByLabelText(/Image sequence/), {
      target: { files: [tiff()] },
    });
    return form;
  }

  const submit = () =>
    screen.getByRole("button", { name: "Upload & preprocess" });

  it("will not upload until the frame interval, channel width, and channel height are given", async () => {
    mockApi();
    render(<Harness />);
    const form = await openModal();

    // Name and frames alone are not enough: preprocessing bakes these in.
    expect(submit()).toBeDisabled();

    fireEvent.change(form.getByLabelText(/Frame interval/), {
      target: { value: "0.25" },
    });
    expect(submit()).toBeDisabled();

    fireEvent.change(form.getByLabelText(/Channel width/), {
      target: { value: "400" },
    });
    expect(submit()).toBeDisabled();

    fireEvent.change(form.getByLabelText(/Channel height/), {
      target: { value: "150" },
    });
    expect(submit()).toBeEnabled();
  });

  it("rejects a frame interval outside the server's bounds", async () => {
    mockApi();
    render(<Harness />);
    const form = await openModal();
    fireEvent.change(form.getByLabelText(/Channel width/), {
      target: { value: "400" },
    });

    fireEvent.change(form.getByLabelText(/Frame interval/), {
      target: { value: "-1" },
    });

    expect(form.getByLabelText(/Frame interval/)).toHaveAttribute(
      "aria-invalid",
      "true",
    );
    expect(submit()).toBeDisabled();
  });

  it("saves the conditions before preprocessing, not after", async () => {
    const calls = mockApi();
    const order: string[] = [];
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const original = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(
      async (url: string | URL, opts?: RequestInit) => {
        const u = String(url);
        if (u.endsWith("/conditions")) order.push("conditions");
        if (u.endsWith("/preprocess") && opts?.method === "POST")
          order.push("preprocess");
        return original(url, opts);
      },
    );

    render(<Harness />);
    const form = await openModal();
    fireEvent.change(form.getByLabelText(/Frame interval/), {
      target: { value: "0.25" },
    });
    fireEvent.change(form.getByLabelText(/Channel width/), {
      target: { value: "400" },
    });
    fireEvent.change(form.getByLabelText(/Channel height/), {
      target: { value: "150" },
    });
    fireEvent.click(submit());

    await waitFor(() => expect(calls.startPreprocess).toEqual(["mid_T"]));
    // Tensors built before the conditions land would carry the wrong time axis
    // and the wrong µm/px, and nothing downstream would say so. All conditions
    // are sent at once: the filled measurements and the operating defaults.
    expect(order).toEqual(["conditions", "preprocess"]);
    expect(calls.conditionPatches).toEqual([
      {
        dt_frame_ms: 0.25,
        channel_width_um: 400,
        channel_height_um: 150,
        fluid: "fc72",
        q_wall_W_cm2: 2,
        flow_rate_mL_hr: 5,
        U_ref: 0.2,
      },
    ]);
  });

  it("derives the saturation temperature from the fluid, read-only", async () => {
    mockApi();
    render(<Harness />);
    const form = await openModal();

    // FC-72 is the default; its saturation temperature is shown, not typed.
    const fluid = await form.findByLabelText("Working fluid");
    expect(form.getByText(/56\.6/)).toBeInTheDocument();
    // There is no editable T_sat field any more.
    expect(
      form.queryByRole("spinbutton", { name: /saturation/i }),
    ).toBeNull();

    // Choosing water updates the read-only saturation temperature.
    fireEvent.change(fluid, { target: { value: "water" } });
    expect(await form.findByText(/100/)).toBeInTheDocument();
  });

  it("sends the chosen fluid with the conditions", async () => {
    const calls = mockApi();
    render(<Harness />);
    const form = await openModal();
    fireEvent.change(await form.findByLabelText("Working fluid"), {
      target: { value: "water" },
    });
    fireEvent.change(form.getByLabelText(/Frame interval/), {
      target: { value: "0.25" },
    });
    fireEvent.change(form.getByLabelText(/Channel width/), {
      target: { value: "400" },
    });
    fireEvent.change(form.getByLabelText(/Channel height/), {
      target: { value: "150" },
    });
    fireEvent.click(submit());

    await waitFor(() => expect(calls.conditionPatches).toHaveLength(1));
    const patch = calls.conditionPatches[0] as Record<string, unknown>;
    expect(patch.fluid).toBe("water");
    expect(patch).not.toHaveProperty("T_sat_C");
  });

  it("does not preprocess when the conditions could not be saved", async () => {
    const calls = mockApi();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const original = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(
      async (url: string | URL, opts?: RequestInit) => {
        if (String(url).endsWith("/conditions")) {
          return new Response(
            JSON.stringify({ detail: "dt_frame_ms out of range" }),
            {
              status: 400,
            },
          );
        }
        return original(url, opts);
      },
    );

    render(<Harness />);
    const form = await openModal();
    fireEvent.change(form.getByLabelText(/Frame interval/), {
      target: { value: "0.25" },
    });
    fireEvent.change(form.getByLabelText(/Channel width/), {
      target: { value: "400" },
    });
    fireEvent.change(form.getByLabelText(/Channel height/), {
      target: { value: "150" },
    });
    fireEvent.click(submit());

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /operating conditions could not be saved/,
    );
    // Silently preprocessing with another series' values is the failure mode
    // this whole flow exists to prevent.
    expect(calls.startPreprocess).toEqual([]);
  });

  it("surfaces a failed fluid-catalogue load and blocks upload", async () => {
    mockApi();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const original = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(
      async (url: string | URL, opts?: RequestInit) => {
        if (String(url).endsWith("/api/fluids")) {
          return new Response("upstream down", { status: 503 });
        }
        return original(url, opts);
      },
    );

    render(<Harness />);
    const form = await openModal();
    fireEvent.change(form.getByLabelText(/Frame interval/), {
      target: { value: "0.25" },
    });
    fireEvent.change(form.getByLabelText(/Channel width/), {
      target: { value: "400" },
    });
    fireEvent.change(form.getByLabelText(/Channel height/), {
      target: { value: "150" },
    });

    // The failure is shown, not swallowed, and upload stays blocked.
    expect(await form.findByRole("alert")).toHaveTextContent(
      /Could not load the fluid catalogue/,
    );
    expect(submit()).toBeDisabled();
  });

  it("blocks upload when no fluids are characterised", async () => {
    mockApi();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const original = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(
      async (url: string | URL, opts?: RequestInit) => {
        if (String(url).endsWith("/api/fluids")) return json([]);
        return original(url, opts);
      },
    );

    render(<Harness />);
    const form = await openModal();
    fireEvent.change(form.getByLabelText(/Frame interval/), {
      target: { value: "0.25" },
    });
    fireEvent.change(form.getByLabelText(/Channel width/), {
      target: { value: "400" },
    });
    fireEvent.change(form.getByLabelText(/Channel height/), {
      target: { value: "150" },
    });

    // No fluid to commit to → the flow cannot proceed (no silent default).
    // Wait a tick for the (empty) catalogue fetch to resolve before asserting.
    await waitFor(() => expect(form.getByLabelText("Working fluid")).toBeEnabled());
    expect(submit()).toBeDisabled();
  });
});

describe("QC chart axes", () => {
  async function showCheck(name: RegExp) {
    mockApi({ processed: true });
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    // The QC tabs appear once the tensors load (before that the section prompts).
    fireEvent.click(await screen.findByRole("tab", { name }));
  }

  it("names the quantity and unit on both axes of the kinematics chart", async () => {
    mockApi({ processed: true });
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);

    // Tick numbers alone do not say what is being measured.
    expect(await screen.findByText(/t \(ms\)/)).toBeInTheDocument();
    expect(screen.getByText(/L \(µm\)/)).toBeInTheDocument();
  });

  it("names both axes of the interface chart in physical units", async () => {
    await showCheck(/Interface evolution/);

    expect(screen.getByText(/x \(µm\), downstream/)).toBeInTheDocument();
    expect(screen.getByText(/y \(µm\), across channel/)).toBeInTheDocument();
  });

  it("names both axes of the signed-distance chart", async () => {
    await showCheck(/Signed distance/);

    expect(screen.getByText(/x \(µm\), downstream/)).toBeInTheDocument();
    expect(screen.getByText("y (µm)")).toBeInTheDocument();
  });

  it("draws each frame as one closed silhouette, not loose arcs", async () => {
    await showCheck(/Interface evolution/);

    const shapes = document.querySelectorAll(".qc-silhouette");
    expect(shapes).toHaveLength(1); // the fixture has one frame
    // A ring that does not close leaves the bubble looking broken.
    expect(shapes[0].getAttribute("d")).toMatch(/Z$/);
    expect(shapes[0].getAttribute("fill")).toBeTruthy();
  });

  it("keeps chart text off SVG's black default on the dark canvas", async () => {
    await showCheck(/Interface evolution/);

    // The class has to land where the CSS can reach it; a bare <text> with a
    // descendant-only rule falls back to black and is invisible here.
    for (const label of document.querySelectorAll("text.chart-axis-title")) {
      expect(label.classList.contains("chart-axis-title")).toBe(true);
    }
    expect(
      document.querySelectorAll("text.chart-axis-title").length,
    ).toBeGreaterThan(0);
  });
});

describe("frame boundary overlay", () => {
  // A calibrated, processed series is what makes the overlay possible: the
  // rings need µm/px and the raw frame size to land on real pixels.
  const calibrated = () =>
    mockApi({ processed: true, umPerPx: 100 });

  async function enlarge(frame: number) {
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    await screen.findByText("Preprocessing QC");
    fireEvent.click(
      screen.getByRole("button", { name: `Enlarge frame ${frame}` }),
    );
    return within(await screen.findByRole("dialog", { name: /Frame/ }));
  }

  it("overlays the detected boundary on the enlarged frame by default", async () => {
    calibrated();
    await enlarge(1);

    const path = document.querySelector(".frame-overlay path");
    expect(path).not.toBeNull();
    // Flip (col = W − 0.5 − x*·L/µm) and ROI shift (row = y*·L/µm − 0.5 + y0)
    // with W=16, L/µm = 300/100 = 3, y0 = 2: [0.2, 0.3] → (14.90, 2.40).
    expect(path?.getAttribute("d")).toContain("M14.90 2.40");
    expect(path?.getAttribute("d")).toMatch(/Z$/); // a closed silhouette
  });

  it("toggles the boundary off and back on", async () => {
    calibrated();
    const dialog = await enlarge(1);
    expect(document.querySelector(".frame-overlay")).not.toBeNull();

    const toggle = dialog.getByRole("button", { name: "Boundary" });
    fireEvent.click(toggle); // off
    expect(document.querySelector(".frame-overlay")).toBeNull();
    expect(toggle).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(toggle); // back on
    expect(document.querySelector(".frame-overlay")).not.toBeNull();
    expect(toggle).toHaveAttribute("aria-pressed", "true");
  });

  it("says so, and draws nothing, when a frame has no silhouette", async () => {
    calibrated(); // the fixture only has a ring for frame 1
    const dialog = await enlarge(2);

    expect(document.querySelector(".frame-overlay")).toBeNull();
    expect(dialog.getByText(/no boundary this frame/)).toBeInTheDocument();
    // The control is still there so the user can turn it off.
    expect(
      dialog.getByRole("button", { name: "Boundary" }),
    ).toBeInTheDocument();
  });

  it("offers no overlay control for an uncalibrated series", async () => {
    mockApi({ processed: true }); // um_per_px stays null
    const dialog = await enlarge(1);

    expect(
      dialog.queryByRole("button", { name: "Boundary" }),
    ).toBeNull();
    expect(document.querySelector(".frame-overlay")).toBeNull();
  });
});
