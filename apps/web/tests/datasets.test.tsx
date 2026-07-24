import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DatasetsView } from "../src/views/DatasetsView";
import { ProjectsView } from "../src/views/ProjectsView";

const DETAIL = {
  id: "sample",
  n_frames: 3,
  processed: false,
  has_qc: false,
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
  },
};
const GROUPS = { Re: 215.5, We: 2.302, Ca: 0.01068, Pr: 9.411, Bond: 0.0746, hele_shaw: 0.2228, bretherton_film_um: 4.875, Dh_um: 200 };
const IDLE = { dataset: "sample", state: "idle", message: null, has_qc: false };

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200 });
}

interface Calls {
  upload: number;
  startPreprocess: number;
}

function mockApi(): Calls {
  const calls: Calls = { upload: 0, startPreprocess: 0 };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL, opts?: RequestInit) => {
      const u = String(url);
      const post = opts?.method === "POST";
      if (u.endsWith("/api/datasets")) return json([{ id: "sample", n_frames: 3, processed: false }]);
      if (u.endsWith("/api/runs")) return json([]);
      if (u.endsWith("/groups")) return json(GROUPS);
      if (u.endsWith("/upload")) {
        calls.upload += 1;
        return json({ id: "sample", n_frames: 5, processed: false });
      }
      if (u.endsWith("/preprocess")) {
        if (post) {
          calls.startPreprocess += 1;
          return json({ ...IDLE, state: "running" });
        }
        return json(IDLE);
      }
      if (/\/api\/datasets\/[^/]+$/.test(u)) return json(DETAIL);
      return new Response("not found", { status: 404 });
    }),
  );
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

describe("DatasetsView", () => {
  it("shows operating conditions, live groups, and the image sequence", async () => {
    mockApi();
    render(<DatasetsView datasetId="sample" />);

    expect(await screen.findByText("Operating conditions")).toBeInTheDocument();
    expect(screen.getByText("FC-72")).toBeInTheDocument();
    expect(screen.getByText("300×150")).toBeInTheDocument();
    expect(screen.getByText("215.5")).toBeInTheDocument(); // live Reynolds
    expect(screen.getAllByAltText(/frame \d+/).length).toBe(3); // image sequence
  });

  it("uploads selected frames through the API", async () => {
    const calls = mockApi();
    render(<DatasetsView datasetId="sample" />);
    const input = (await screen.findByLabelText(/Image sequence \(TIFF frames\)/)) as HTMLInputElement;

    const file = new File([new Uint8Array([0x49, 0x49, 0x2a, 0x00])], "1.tif", { type: "image/tiff" });
    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => expect(calls.upload).toBe(1));
  });

  it("starts preprocessing when the button is clicked", async () => {
    const calls = mockApi();
    render(<DatasetsView datasetId="sample" />);
    const button = await screen.findByRole("button", { name: /Run preprocessing/ });

    fireEvent.click(button);
    await waitFor(() => expect(calls.startPreprocess).toBe(1));
  });

  it("polls a running job to completion and shows the QC preview", async () => {
    let started = false;
    let polls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL, opts?: RequestInit) => {
        const u = String(url);
        const post = opts?.method === "POST";
        if (u.endsWith("/api/datasets")) {
          return json([{ id: "sample", n_frames: 3, processed: polls >= 2 }]);
        }
        if (u.endsWith("/groups")) return json(GROUPS);
        if (u.endsWith("/preprocess")) {
          if (post) {
            started = true;
            return json({ ...IDLE, state: "running" });
          }
          if (!started) return json(IDLE); // idle until a POST starts it
          polls += 1;
          const done = polls >= 2;
          return json({ ...IDLE, state: done ? "done" : "running", has_qc: done });
        }
        if (/\/api\/datasets\/[^/]+$/.test(u)) return json({ ...DETAIL, has_qc: polls >= 2 });
        return new Response("not found", { status: 404 });
      }),
    );

    render(<DatasetsView datasetId="sample" />);
    const button = await screen.findByRole("button", { name: /Run preprocessing/ });
    fireEvent.click(button);

    // The hook polls every 1s (real timers); wait until the job completes and the
    // QC preview appears — proving the running->done transition + refresh + cleanup.
    await waitFor(
      () => expect(screen.getByRole("img", { name: /quality-control/ })).toBeInTheDocument(),
      { timeout: 4000 },
    );
    expect(polls).toBeGreaterThanOrEqual(2);
  });
});

describe("ProjectsView", () => {
  it("renders dataset cards and opens one", async () => {
    mockApi();
    const onOpen = vi.fn();
    render(<ProjectsView onOpen={onOpen} />);

    const card = await screen.findByRole("button", { name: /sample/ });
    fireEvent.click(card);
    expect(onOpen).toHaveBeenCalledWith("sample");
  });
});
