import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DatasetsView } from "../src/views/DatasetsView";

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
  startPreprocess: number;
  conditionPatches: Record<string, unknown>[];
  projectPatches: Record<string, unknown>[];
}

function mockApi({ processed = false, holdout = 6 } = {}): Calls {
  const calls: Calls = {
    upload: [],
    startPreprocess: 0,
    conditionPatches: [],
    projectPatches: [],
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL, opts?: RequestInit) => {
      const u = String(url);
      const post = opts?.method === "POST";
      if (u.endsWith("/api/datasets")) {
        return json([
          { id: "sample", n_frames: 3, processed, conditions_set: false, frame_px: [16, 12] },
        ]);
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
      if (u.endsWith("/preprocess")) {
        if (post) {
          calls.startPreprocess += 1;
          return json({ ...IDLE, state: "running" });
        }
        return json(IDLE);
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

  it("uploads replacement frames for the selected series", async () => {
    const calls = mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    const input = (await screen.findByLabelText(
      /Image sequence \(TIFF frames\)/,
    )) as HTMLInputElement;

    const file = new File([new Uint8Array([0x49, 0x49, 0x2a, 0x00])], "1.tif", {
      type: "image/tiff",
    });
    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => expect(calls.upload).toEqual(["sample"]));
  });

  it("uploads a new series and attaches it to the project", async () => {
    const calls = mockApi();
    const onProjectChanged = vi.fn();
    render(<DatasetsView project={PROJECT} onProjectChanged={onProjectChanged} />);

    fireEvent.click(await screen.findByRole("button", { name: /Upload new series/ }));
    fireEvent.change(screen.getByLabelText("Series name"), { target: { value: "mid_T" } });
    const file = new File([new Uint8Array([0x49, 0x49, 0x2a, 0x00])], "1.tif", {
      type: "image/tiff",
    });
    fireEvent.change(screen.getByLabelText("Frames"), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "Upload" }));

    await waitFor(() =>
      expect(calls.projectPatches).toEqual([{ datasets: ["sample", "mid_T"] }]),
    );
    expect(calls.upload).toEqual(["mid_T"]);
    expect(onProjectChanged).toHaveBeenCalledWith(
      expect.objectContaining({ datasets: ["sample", "mid_T"] }),
    );
  });

  it("starts preprocessing when the button is clicked", async () => {
    const calls = mockApi();
    render(<DatasetsView project={PROJECT} onProjectChanged={noop} />);
    const button = await screen.findByRole("button", { name: /Run preprocessing/ });

    fireEvent.click(button);
    await waitFor(() => expect(calls.startPreprocess).toBe(1));
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
