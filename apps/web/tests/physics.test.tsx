import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { EquationBlock } from "../src/components";
import type { DatasetSummary } from "../src/lib/api";
import { PhysicsModelView } from "../src/views/PhysicsModelView";

const DATASETS = [{ id: "sample", n_frames: 3, processed: true }] as unknown as DatasetSummary[];

const MODEL = {
  fields: ["phi", "u", "v", "s"],
  hidden: 96,
  layers: 4,
  fourier_feats: 64,
  fourier_scale: 3.0,
  alpha_eps: 0.05,
  nodewise_activation: true,
  per_field: {},
};

function eq(
  id: string,
  name: string,
  stage: "A" | "B",
  extra: Partial<Record<string, unknown>> = {},
) {
  return {
    id,
    name,
    stage,
    tex: `\\mathrm{${id}}`,
    weight_key: id,
    fields_required: [],
    fields_added: [],
    groups: [],
    core: stage === "A",
    enabled: stage === "A",
    weight: 1.0,
    ...extra,
  };
}

const PHYSICS = {
  dataset: "sample",
  fields: ["phi", "u", "v", "s"],
  groups: { Re: 215.5, We: 2.302, Pe: 2028, hele_shaw: 0.2228, dT_ref: 28.74 },
  equations: [
    eq("vof", "Interface transport", "A"),
    eq("div", "Continuity", "A"),
    eq("src", "Source penalty", "A"),
    eq("bc", "Boundary conditions", "A"),
    eq("mom", "Momentum", "B", { fields_added: ["p"], groups: ["Re", "We", "hele_shaw"] }),
    eq("energy", "Energy + evaporation", "B", { fields_added: ["T"], groups: ["Pe"] }),
    eq("evap", "Evaporation mass closure", "B"),
  ],
};

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200 });
}

function mockApi(model = MODEL, physics = PHYSICS) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL) => {
      const u = String(url);
      if (u.includes("/api/model/")) return json(model);
      if (u.includes("/api/physics/")) return json(physics);
      return new Response("not found", { status: 404 });
    }),
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("EquationBlock", () => {
  it("renders LaTeX via KaTeX", () => {
    const { container } = render(<EquationBlock tex="\\alpha = \\sigma(\\phi/\\varepsilon)" />);
    expect(container.querySelector(".katex")).toBeInTheDocument();
  });
});

describe("PhysicsModelView", () => {
  it("renders the equations, model builder, ensemble, table and run bar", async () => {
    mockApi();
    render(<PhysicsModelView datasets={DATASETS} />);

    expect(
      await screen.findByRole("heading", { name: "Governing equations" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Momentum")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Model builder" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /Medium/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Per-field architecture" })).toBeInTheDocument();
    // The run bar shows the exact Hydra command.
    expect(screen.getByText(/naviernet train dataset=sample/)).toBeInTheDocument();
    // Stage A only, so four networks.
    expect(screen.getByText(/4 networks/)).toBeInTheDocument();
  });

  it("enabling Momentum unlocks the pressure field", async () => {
    mockApi();
    render(<PhysicsModelView datasets={DATASETS} />);
    const momentum = await screen.findByRole("switch", { name: "Momentum" });
    expect(momentum).toHaveAttribute("aria-checked", "false");

    fireEvent.click(momentum);

    expect(momentum).toHaveAttribute("aria-checked", "true");
    // Pressure joins the ensemble: five networks now.
    await waitFor(() => expect(screen.getByText(/5 networks/)).toBeInTheDocument());
    // And the config is dirty, so it reports unsaved changes.
    expect(screen.getByText("unsaved changes")).toBeInTheDocument();
  });

  it("core equations are locked on and cannot be toggled off", async () => {
    mockApi();
    render(<PhysicsModelView datasets={DATASETS} />);
    const vof = await screen.findByRole("switch", { name: "Interface transport" });
    expect(vof).toBeDisabled();
    expect(vof).toHaveAttribute("aria-checked", "true");
  });

  it("shows an error state when the config fails to load", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("nope", { status: 500 })),
    );
    render(<PhysicsModelView datasets={DATASETS} />);
    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });

  it("prompts to upload when there are no datasets", () => {
    render(<PhysicsModelView datasets={[]} />);
    expect(screen.getByText(/No datasets yet/)).toBeInTheDocument();
  });

  it("applying the Large preset grows the parameter budget", async () => {
    mockApi();
    render(<PhysicsModelView datasets={DATASETS} />);
    const before = (await screen.findByText(/networks · /)).textContent ?? "";

    fireEvent.click(screen.getByRole("radio", { name: /Large/ }));

    await waitFor(() =>
      expect(screen.getByText(/networks · /).textContent).not.toEqual(before),
    );
    expect(screen.getByText("unsaved changes")).toBeInTheDocument();
  });

  it("editing a per-field width marks it overridden with a reset control", async () => {
    mockApi();
    render(<PhysicsModelView datasets={DATASETS} />);
    const widthInput = (await screen.findByLabelText("φ→α width")) as HTMLInputElement;

    fireEvent.change(widthInput, { target: { value: "200" } });

    // The override chip appears with a reset-all control.
    expect(await screen.findByText(/overridden/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "reset all" })).toBeInTheDocument();
  });

  it("flags an interface epsilon that is too small", async () => {
    mockApi();
    render(<PhysicsModelView datasets={DATASETS} />);
    const eps = (await screen.findByLabelText(/Interface ε/)) as HTMLInputElement;

    fireEvent.change(eps, { target: { value: "0.001" } });

    expect(await screen.findByRole("alert")).toHaveTextContent(/ε/);
  });
});
