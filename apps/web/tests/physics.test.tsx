import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { EquationBlock, TopologyChart } from "../src/components";
import { PhysicsModelView } from "../src/views/PhysicsModelView";

const MODEL = {
  fields: ["phi", "u", "v", "s"],
  hidden: 96,
  layers: 4,
  fourier_feats: 64,
  fourier_scale: 3.0,
  alpha_eps: 0.05,
  nodewise_activation: true,
};

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200 });
}

function mockApi() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL) => {
      const u = String(url);
      if (u.endsWith("/api/datasets"))
        return json([{ id: "sample", n_frames: 3, processed: false }]);
      if (u.includes("/api/model/")) return json(MODEL);
      return new Response("not found", { status: 404 });
    }),
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("EquationBlock", () => {
  it("renders LaTeX via KaTeX", () => {
    const { container } = render(
      <EquationBlock tex="\\alpha = \\sigma(\\phi/\\varepsilon)" />,
    );
    expect(container.querySelector(".katex")).toBeInTheDocument();
  });
});

describe("TopologyChart", () => {
  it("draws field-output nodes, connecting edges, and labels from the model", () => {
    const { container } = render(<TopologyChart model={MODEL} />);
    // 4 field outputs are drawn as .field nodes.
    expect(container.querySelectorAll(".topo-node.field")).toHaveLength(4);
    // Edges wire the columns together (the "topology").
    expect(
      container.querySelectorAll(".topo-edges line").length,
    ).toBeGreaterThan(0);
    // Column labels reflect the model's real dimensions.
    expect(container.textContent).toContain("hidden · 96 × 4");
    expect(container.textContent).toContain("Fourier · 128");
  });
});

describe("PhysicsModelView", () => {
  it("shows governing equations, live topology, and per-field architecture", async () => {
    mockApi();
    render(<PhysicsModelView />);

    expect(
      screen.getByRole("heading", { name: "Governing equations" }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("heading", { name: "Model topology · live" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Advanced; per-field architecture" }),
    ).toBeInTheDocument();
    expect(screen.getByText("phi, u, v, s")).toBeInTheDocument();
    expect(screen.getByText("96")).toBeInTheDocument(); // hidden width
  });

  it("shows an error state when the model fails to load", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/api/datasets")) {
          return json([{ id: "sample", n_frames: 3, processed: false }]);
        }
        return new Response("not found", { status: 404 }); // getModel fails
      }),
    );
    render(<PhysicsModelView />);
    // Equations always render; the model panels are replaced by an alert.
    expect(
      screen.getByRole("heading", { name: "Governing equations" }),
    ).toBeInTheDocument();
    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });

  it("shows only the equations when there are no datasets yet", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => json([])),
    );
    render(<PhysicsModelView />);
    expect(
      screen.getByRole("heading", { name: "Governing equations" }),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(
        screen.queryByRole("heading", { name: "Model topology · live" }),
      ).not.toBeInTheDocument(),
    );
  });
});
