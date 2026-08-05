import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { EquationBlock } from "../src/components";
import type { DatasetSummary } from "../src/lib/api";
import { PhysicsModelView } from "../src/views/PhysicsModelView";
import { renderStage } from "./stageHarness";

const DATASETS = [
  { id: "sample", n_frames: 3, processed: true },
] as unknown as DatasetSummary[];

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
  fields_required: string[],
  extra: Partial<Record<string, unknown>> = {},
) {
  return {
    id,
    name,
    stage,
    tex: `\\mathrm{${id}}`,
    weight_key: id,
    fields_required,
    fields_added: [],
    groups: [],
    core: stage === "A",
    enabled: stage === "A",
    weight: 1.0,
    mode: "any",
    ...extra,
  };
}

const PHYSICS = {
  dataset: "sample",
  fields: ["phi", "u", "v", "s"],
  // Resolved at run launch from whether the series trains `p`; Stage A does not.
  sharp_interface: false,
  groups: { Re: 215.5, We: 2.302, Pe: 2028, hele_shaw: 0.2228, dT_ref: 28.74 },
  equations: [
    eq("vof", "Interface transport", "A", ["phi", "u", "v"]),
    eq("div", "Continuity", "A", ["u", "v", "s"]),
    eq("src", "Source penalty", "A", ["s"]),
    eq("bc", "Boundary conditions", "A", ["u", "v"]),
    eq("mom", "Momentum", "B", ["phi", "u", "v", "p"], {
      fields_added: ["p"],
      groups: ["Re", "We", "hele_shaw"],
    }),
    eq("energy", "Energy + evaporation", "B", ["u", "v", "T"], {
      fields_added: ["T"],
      groups: ["Pe"],
    }),
    eq("evap", "Evaporation mass closure", "B", ["s", "T"]),
  ],
};

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200 });
}

function mockApi(model = MODEL, physics = PHYSICS) {
  // Writes echo the request back merged onto the stored shape, so a save
  // settles the same way the real API does.
  const fetchMock = vi.fn(async (url: string | URL, init?: RequestInit) => {
    const u = String(url);
    const sent = init?.body ? JSON.parse(String(init.body)) : {};
    if (u.includes("/api/model/")) {
      return json(init?.method ? { ...model, ...sent } : model);
    }
    if (u.includes("/api/physics/")) {
      return json(init?.method ? { ...physics, ...sent } : physics);
    }
    return new Response("not found", { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
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

describe("PhysicsModelView", () => {
  it("sets the physics in the aside and derives everything on the canvas", async () => {
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);

    // Config: the physics that defines the objective, and the capacity that
    // has to carry it. Both in bands, both in the aside.
    expect(await screen.findByText("Core physics")).toBeInTheDocument();
    expect(screen.getByText("Optional physics")).toBeInTheDocument();
    expect(screen.getByText("Capacity")).toBeInTheDocument();
    expect(screen.getByText("Momentum")).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /Medium/ })).toBeInTheDocument();

    // Derived: the price, the loss it buys, the ensemble, the command.
    expect(
      screen.getByRole("status", { name: "Model budget" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "The objective" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("status", { name: "Interface formulation" }),
    ).toBeInTheDocument();
    // The exact command is one press away on the run bar; it mounts when
    // opened rather than sitting hidden in the DOM.
    fireEvent.click(screen.getByRole("button", { name: "The exact command" }));
    expect(
      await screen.findByText(/naviernet train dataset=sample/),
    ).toBeInTheDocument();
    // Stage A only, so four networks.
    expect(screen.getByText(/4 networks/)).toBeInTheDocument();
  });

  it("enabling Momentum unlocks the pressure field", async () => {
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    const momentum = await screen.findByRole("switch", { name: "Momentum" });
    expect(momentum).toHaveAttribute("aria-checked", "false");

    fireEvent.click(momentum);

    expect(momentum).toHaveAttribute("aria-checked", "true");
    // Pressure joins the ensemble: five networks now.
    await waitFor(() =>
      expect(screen.getByText(/5 networks/)).toBeInTheDocument(),
    );
    // And the config is dirty, so it reports unsaved changes.
    expect(screen.getByText("unsaved changes")).toBeInTheDocument();
  });

  it("holds Save until there is an edit to commit", async () => {
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);

    const save = await screen.findByRole("button", { name: "Save" });
    expect(save).toBeDisabled();

    fireEvent.click(await screen.findByRole("switch", { name: "Momentum" }));

    await waitFor(() => expect(save).toBeEnabled());
  });

  it("Save writes the physics and the architecture, then goes clean", async () => {
    // Both write the same model.json, so the order is load-bearing: physics
    // first, then the architecture merged onto it.
    const fetchMock = mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);

    fireEvent.click(await screen.findByRole("switch", { name: "Momentum" }));
    expect(screen.getByText("unsaved changes")).toBeInTheDocument();

    fireEvent.click(await screen.findByRole("button", { name: "Save" }));

    await waitFor(() => expect(screen.getByText("saved")).toBeInTheDocument());

    const writes = fetchMock.mock.calls
      .map(([url, init]) => ({
        url: String(url),
        method: (init as RequestInit | undefined)?.method,
      }))
      .filter((call) => call.method && call.method !== "GET");
    expect(writes.map((w) => w.url.replace(/^.*\/api\//, "api/"))).toEqual([
      "api/physics/sample",
      "api/model/sample",
    ]);
    // The saved config is the edited one, not the loaded one: Momentum ("mom"
    // in the schema) is off in the fixture and on in what we just wrote.
    const body = JSON.parse(
      String(
        (fetchMock.mock.calls.find(
          ([url, init]) =>
            String(url).includes("/api/physics/") &&
            (init as RequestInit | undefined)?.method,
        )?.[1] as RequestInit)!.body,
      ),
    ) as { enabled: string[] };
    expect(PHYSICS.equations.find((e) => e.id === "mom")!.enabled).toBe(false);
    expect(body.enabled).toContain("mom");
  });

  it("enabling Energy lights up the coupled evaporation closure", async () => {
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    const energy = await screen.findByRole("switch", {
      name: "Energy + evaporation",
    });
    const evap = screen.getByRole("switch", {
      name: "Evaporation mass closure",
    });
    expect(evap).toHaveAttribute("aria-checked", "false");
    expect(evap).toBeDisabled(); // not independently toggleable; rides on Energy

    fireEvent.click(energy);

    // The coupled closure follows Energy on, even though it can't be toggled itself.
    await waitFor(() => expect(evap).toHaveAttribute("aria-checked", "true"));
  });

  it("states the core equations rather than offering a switch that cannot move", async () => {
    // They always train and their weights belong to the run-launch form, so
    // the row carried a switch that could never move and a field that could
    // never be typed in. Both were chrome.
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    await screen.findByText("Core physics");

    expect(
      screen.queryByRole("switch", { name: "Interface transport" }),
    ).not.toBeInTheDocument();
    expect(screen.getByText("Interface transport")).toBeInTheDocument();
    // Only the four toggleable Stage-B equations carry a switch.
    expect(screen.getAllByRole("switch")).toHaveLength(3);
  });

  it("shows an error state when the config fails to load", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("nope", { status: 500 })),
    );
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });

  it("prompts to upload when there are no datasets", () => {
    renderStage(<PhysicsModelView datasets={[]} />);
    expect(screen.getByText(/No datasets yet/)).toBeInTheDocument();
  });

  const paramK = (text: string) =>
    Number(/·\s*([\d.]+)\s*k params/.exec(text)?.[1] ?? "0");

  it("applying the Large preset grows the parameter budget", async () => {
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    const before = paramK(
      (await screen.findByText(/networks · /)).textContent ?? "",
    );

    fireEvent.click(screen.getByRole("radio", { name: /Large/ }));

    await waitFor(() =>
      expect(
        paramK(screen.getByText(/networks · /).textContent ?? ""),
      ).toBeGreaterThan(before),
    );
    expect(screen.getByText("unsaved changes")).toBeInTheDocument();
  });

  it("clicking a lane inspects that field", async () => {
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    // The u lane; the inspector starts on phi (φ→α).
    const uLane = await screen.findByRole("button", { name: /^u network,/ });

    fireEvent.click(uLane);

    // The inspector (aria-live) now names u and its transform.
    const inspector = document.querySelector(".inspector");
    expect(inspector?.textContent).toContain("identity");
  });

  it("editing a Stage-B weight is reflected in the reproducible command", async () => {
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    fireEvent.click(await screen.findByRole("switch", { name: "Momentum" }));

    const momWeight = screen
      .getByLabelText("Momentum")
      .parentElement?.querySelector("#w-mom") as HTMLInputElement;
    // The weight input is enabled for the now-on Stage-B equation.
    expect(momWeight).not.toBeDisabled();
    fireEvent.change(momWeight, { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: "The exact command" }));

    await waitFor(() =>
      expect(screen.getByText(/training\.weights\.mom=3/)).toBeInTheDocument(),
    );
  });

  it("core-equation weights are stated, not offered as dead inputs", async () => {
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    await screen.findByText("Core physics");

    // No input at all for a core weight: the Solver owns those values, and the
    // band header says so.
    expect(document.querySelector("#w-vof")).toBeNull();
    expect(screen.getByText(/weights set at launch/)).toBeInTheDocument();
  });

  it("locked field rows name the equation that unlocks them", async () => {
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    await screen.findByText("Capacity");
    expect(screen.getByText(/enable Momentum to unlock/)).toBeInTheDocument();
    expect(screen.getByText(/enable Energy to unlock/)).toBeInTheDocument();
  });

  it("editing a per-field width marks it overridden with a reset control", async () => {
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    const widthInput = (await screen.findByLabelText(
      "φ→α width",
    )) as HTMLInputElement;

    fireEvent.change(widthInput, { target: { value: "200" } });

    // The count and the way back ride the band's own header.
    expect(await screen.findByText(/overridden/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "reset" })).toBeInTheDocument();
  });

  it("writes the objective this configuration actually trains", async () => {
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    const objective = await screen.findByRole("heading", {
      name: "The objective",
    });
    const panel = objective.closest(".card") as HTMLElement;

    // Stage A: the supervised term plus the four core residuals, with the
    // launch-default weights substituted rather than implied.
    expect(within(panel).getByText("L_data")).toBeInTheDocument();
    expect(within(panel).getByText("r_vof")).toBeInTheDocument();
    expect(within(panel).getByText("L_bc")).toBeInTheDocument();

    // What is NOT trained is part of the objective too, so momentum is shown
    // ghosted rather than dropped.
    const mom = within(panel).getByText("r_mom").closest(".obj-term");
    expect(mom).toHaveClass("off");
  });

  it("un-ghosts a term in the objective when its equation is enabled", async () => {
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    const objective = (
      await screen.findByRole("heading", { name: "The objective" })
    ).closest(".card") as HTMLElement;
    fireEvent.click(screen.getByRole("switch", { name: "Momentum" }));

    // Scoped: the ensemble draws an `r_mom` hub of its own on the canvas.
    await waitFor(() =>
      expect(
        within(objective).getByText("r_mom").closest(".obj-term"),
      ).not.toHaveClass("off"),
    );
  });

  it("reports the interface formulation rather than owning a second copy", async () => {
    // model.sharp_interface is resolved at run launch, and the launcher appends
    // the series overrides AFTER its own list — so a control here would
    // silently outrank an explicit Solver choice. This page states the outcome.
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    const note = await screen.findByRole("status", {
      name: "Interface formulation",
    });

    expect(note).toHaveTextContent("diffuse");
    expect(note).toHaveTextContent(/no pressure field/i);
    // Stated, never set.
    expect(within(note).queryByRole("button")).not.toBeInTheDocument();
    expect(within(note).queryByRole("switch")).not.toBeInTheDocument();
  });

  it("swaps the physics that carries pressure when the front is explicit", async () => {
    // Both treatments are real physics the registry carries; the field set
    // decides which a launch gets. Momentum is the switch that ADDS pressure,
    // so it stays on screen either way — what changes is what carries it.
    mockApi(MODEL, {
      ...PHYSICS,
      fields: ["phi", "u", "v", "s", "p"],
      equations: [
        ...PHYSICS.equations.filter((e) => e.id !== "mom"),
        eq("mom", "Momentum", "B", ["phi", "u", "v", "p"], {
          fields_added: ["p"],
          mode: "diffuse",
          enabled: true,
        }),
        eq("darcy", "Darcy", "B", ["phi", "u", "v", "p"], { mode: "sharp" }),
      ],
    });
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    await screen.findByText("Optional physics");

    // Pressure is trained, so a launch takes the sharp front by default.
    expect(
      screen.getByRole("status", { name: "Interface formulation" }),
    ).toHaveTextContent("sharp front");
    expect(screen.getByText("Darcy")).toBeInTheDocument();
    // Momentum keeps its switch (it is the pressure switch) and says why it is
    // not itself the residual being trained.
    expect(screen.getByText("Momentum")).toBeInTheDocument();
    expect(
      screen.getByText(/carried by the front conditions/),
    ).toBeInTheDocument();
  });

  it("draws the same physics in the ensemble as the objective lists", async () => {
    // A diagram that disagrees with the loss above it is worse than no diagram.
    mockApi(MODEL, {
      ...PHYSICS,
      fields: ["phi", "u", "v", "s", "p"],
      equations: [
        ...PHYSICS.equations.filter((e) => e.id !== "mom"),
        eq("mom", "Momentum", "B", ["phi", "u", "v", "p"], {
          fields_added: ["p"],
          mode: "diffuse",
          enabled: true,
        }),
        eq("darcy", "Darcy", "B", ["phi", "u", "v", "p"], { mode: "sharp" }),
      ],
    });
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    const ensemble = await screen.findByRole("img", {
      name: /network ensemble/i,
    });

    // The sharp treatment is admitted, so its residual is a hub and the
    // diffuse one is not drawn at all.
    expect(within(ensemble).getByText("r_darcy")).toBeInTheDocument();
    expect(within(ensemble).queryByText("r_mom")).not.toBeInTheDocument();
  });

  it("opens an equation's detail outside the rail that scrolls it", async () => {
    // An absolutely-positioned panel inside the rail is clipped by its scroll
    // container — near the foot of the rail it disappears below the edge. The
    // panel is portalled to the body instead, so it holds wherever the row is.
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    await screen.findByText("Core physics");

    fireEvent.click(
      screen.getByRole("button", { name: "Interface transport detail" }),
    );

    const pop = await screen.findByRole("tooltip");
    expect(pop).toBeInTheDocument();
    // Outside the rail's subtree entirely: that is what escapes the clip.
    expect(pop.closest(".stage-aside")).toBeNull();
    expect(pop.parentElement).toBe(document.body);
  });

  it("names the fix on this page when the Solver would refuse the series", async () => {
    // The Solver sends sharp by default and the API rejects that ask without a
    // pressure field, so a Stage-A series would 422 at launch. The remedy is a
    // toggle in this rail, and the note says so.
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    const note = await screen.findByRole("status", {
      name: "Interface formulation",
    });

    expect(note).toHaveTextContent("diffuse");
    expect(note).toHaveTextContent(/Enable Momentum above/);
    expect(note).toHaveTextContent(/refuse this series/);
  });

  it("flags an interface epsilon that is too small", async () => {
    mockApi();
    renderStage(<PhysicsModelView datasets={DATASETS} />);
    const eps = (await screen.findByLabelText(
      /Interface ε/,
    )) as HTMLInputElement;

    fireEvent.change(eps, { target: { value: "0.001" } });

    expect(await screen.findByRole("alert")).toHaveTextContent(/very small/);
  });
});
