import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ProjectsView } from "../src/views/ProjectsView";

const LINKED = {
  id: "a".repeat(32),
  name: "Microchannel FC-72",
  description: "Reconstruct hidden fields from high-speed imaging.",
  datasets: ["sample"],
  created_at: "2026-07-24T00:00:00+00:00",
};
const EMPTY = {
  id: "b".repeat(32),
  name: "Film boiling sweep",
  description: "",
  datasets: [],
  created_at: "2026-07-24T00:01:00+00:00",
};

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200 });
}

interface Calls {
  created: { name: string; description: string }[];
  patched: { id: string; body: Record<string, unknown> }[];
  uploads: string[];
}

function mockApi(): Calls {
  const calls: Calls = { created: [], patched: [], uploads: [] };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL, opts?: RequestInit) => {
      const u = String(url);
      if (u.endsWith("/api/projects") && opts?.method === "POST") {
        const body = JSON.parse(String(opts.body));
        calls.created.push(body);
        return new Response(
          JSON.stringify({ ...EMPTY, id: "c".repeat(32), ...body }),
          { status: 201 },
        );
      }
      if (u.endsWith("/api/projects")) return json([LINKED, EMPTY]);
      const patch = u.match(/\/api\/projects\/([0-9a-f]{32})$/);
      if (patch && opts?.method === "PATCH") {
        const body = JSON.parse(String(opts.body));
        calls.patched.push({ id: patch[1], body });
        const base = patch[1] === LINKED.id ? LINKED : EMPTY;
        return json({ ...base, ...body });
      }
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
        ]);
      }
      if (u.endsWith("/api/runs")) return json([]);
      const upload = u.match(/\/api\/datasets\/([^/]+)\/upload$/);
      if (upload && opts?.method === "POST") {
        calls.uploads.push(upload[1]);
        return json({ id: upload[1], n_frames: 2, processed: false });
      }
      return new Response("not found", { status: 404 });
    }),
  );
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

const noCreate = { creating: false, onCreatingChange: vi.fn() };

describe("ProjectsView", () => {
  it("renders projects joined with their dataset facts", async () => {
    mockApi();
    render(<ProjectsView onOpen={vi.fn()} {...noCreate} />);

    expect(await screen.findByText("Microchannel FC-72")).toBeInTheDocument();
    expect(screen.getByText(/Reconstruct hidden fields/)).toBeInTheDocument();
    // The empty project renders too, with its awaiting-data state.
    expect(screen.getByText("Film boiling sweep")).toBeInTheDocument();
    expect(screen.getByText("Awaiting data")).toBeInTheDocument();
  });

  it("opens a project through its Open button", async () => {
    mockApi();
    const onOpen = vi.fn();
    render(<ProjectsView onOpen={onOpen} {...noCreate} />);

    await screen.findByText("Microchannel FC-72");
    fireEvent.click(screen.getAllByRole("button", { name: /Open/ })[0]);
    expect(onOpen).toHaveBeenCalledWith(
      expect.objectContaining({ id: LINKED.id }),
    );
  });

  it("edits a project's name and description in place", async () => {
    const calls = mockApi();
    render(<ProjectsView onOpen={vi.fn()} {...noCreate} />);

    await screen.findByText("Microchannel FC-72");
    fireEvent.click(screen.getAllByRole("button", { name: "Edit" })[0]);

    const name = screen.getByLabelText("Name");
    fireEvent.change(name, { target: { value: "FC-72 bubble growth" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(calls.patched).toEqual([
        {
          id: LINKED.id,
          body: {
            name: "FC-72 bubble growth",
            description: LINKED.description,
          },
        },
      ]),
    );
    expect(await screen.findByText("FC-72 bubble growth")).toBeInTheDocument();
  });

  it("creates a project and shows its card", async () => {
    const calls = mockApi();
    const onCreatingChange = vi.fn();
    render(
      <ProjectsView
        onOpen={vi.fn()}
        creating
        onCreatingChange={onCreatingChange}
      />,
    );

    await screen.findByText("Microchannel FC-72");
    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "Condensing slug" },
    });
    fireEvent.change(screen.getByLabelText("Description"), {
      target: { value: "Inverse reconstruction." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() =>
      expect(calls.created).toEqual([
        { name: "Condensing slug", description: "Inverse reconstruction." },
      ]),
    );
    expect(await screen.findByText("Condensing slug")).toBeInTheDocument();
    expect(onCreatingChange).toHaveBeenCalledWith(false);
  });

  it("shows the API's rejection instead of silently failing", async () => {
    mockApi();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    render(<ProjectsView onOpen={vi.fn()} {...noCreate} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Is the API running/,
    );
  });
});
