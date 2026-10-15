import { describe, expect, it } from "vitest";

import { FORM_DEFAULTS } from "../src/views/solver/form";
import { RECIPES, recipeDiff, recipeState } from "../src/views/solver/recipes";

describe("recipe stamped states", () => {
  it("makes 'Full physics' exactly the platform defaults", () => {
    // The recommended recipe and FORM_DEFAULTS are one configuration by
    // design; if they drift apart, the rail would open already "modified".
    expect(recipeState("full-physics")).toEqual(FORM_DEFAULTS);
  });

  it("stamps 'Best held-out IoU' as explicit front + causal, zero interface physics", () => {
    const state = recipeState("best-heldout");
    expect(state.front_geometry).toBe(true);
    expect(state.sharp_interface).toBe(false);
    expect(state.film_pressure).toBe(false);
    expect(state.liquid_film).toBe(false);
    expect(state.causal_weighting).toBe(true);
    expect(state.causal_mode).toBe("weight");
    expect(state.hard_pin).toBe(false);
  });

  it("stamps 'Legacy baseline' as the diffuse treatment with nothing structural", () => {
    const state = recipeState("legacy");
    expect(state.front_geometry).toBe(false);
    expect(state.sharp_interface).toBe(false);
    expect(state.film_pressure).toBe(false);
    expect(state.nucleation_root).toBe(false);
    expect(state.front_velocity).toBe(false);
    expect(state.causal_weighting).toBe(false);
  });

  it("stamps 'Trusted fields' as the audited chain: sharp front, physical walls, Brinkman, two-phase pressure, sharp thermal model, mass balance", () => {
    const state = recipeState("trusted");
    expect(state.front_geometry).toBe(true);
    expect(state.sharp_interface).toBe(true);
    expect(state.nucleation_root).toBe(true);
    expect(state.walls_at).toBe("physical");
    expect(state.wall_condition).toBe("no_slip");
    expect(state.momentum_closure).toBe("brinkman");
    expect(state.outlet_condition).toBe("zero_gradient");
    expect(state.pressure_representation).toBe("two_phase");
    expect(state.thermal_model).toBe("sharp");
    expect(state.source_mode).toBe("thermal");
    expect(state.collocation_time_bins).toBeGreaterThan(0);
    expect(state.mass_balance).toBe(true);
    expect(state.causal_weighting).toBe(true);
    expect(state.causal_gate_min_weight).toBe(0.99);
    expect(state.log_chunk_residuals).toBe(true);
    // Mirror symmetry is a safety net for AFTER the audit passes, never stamped.
    expect(state.mirror_symmetry).toBe(false);
    // What the YAMLs leave off stays off, even where the platform default is on:
    // the stamp is the CLI recipe, not the form's defaults with extras.
    expect(state.film_pressure).toBe(false);
    expect(state.depletable_superheat).toBe(false);
    expect(state.evap_closure_two_way).toBe(false);
    expect(state.steps).toBe(12000);
  });

  it("has a card for every recipe name", () => {
    expect(RECIPES.map((recipe) => recipe.name)).toEqual([
      "full-physics",
      "trusted",
      "best-heldout",
      "legacy",
    ]);
  });
});

describe("recipeDiff", () => {
  it("is empty at the stamped state, for every recipe", () => {
    for (const recipe of RECIPES) {
      expect(recipeDiff(recipeState(recipe.name), recipe.name)).toEqual([]);
    }
  });

  it("reports a toggled switch as one signed chip", () => {
    const form = { ...recipeState("full-physics"), nucleation_root: true };
    expect(recipeDiff(form, "full-physics")).toEqual(["+ nucleation root"]);
  });

  it("reports a treatment change as one chip, not a fan of flag chips", () => {
    // Stepping full-physics down to the explicit front switches film pressure
    // off BY CONSTRUCTION; that must not read as a user choice.
    const form = {
      ...recipeState("full-physics"),
      sharp_interface: false,
      film_pressure: false,
    };
    expect(recipeDiff(form, "full-physics")).toEqual([
      "treatment → explicit front",
    ]);
  });

  it("still reports a real choice made after a treatment change", () => {
    const form = {
      ...recipeState("full-physics"),
      sharp_interface: false,
      film_pressure: false,
      nose_shape: true,
    };
    expect(recipeDiff(form, "full-physics")).toEqual([
      "treatment → explicit front",
      "+ nose shape",
    ]);
  });

  it("reports a changed value with its new setting", () => {
    const form = { ...recipeState("full-physics"), n_coll: 4096 };
    expect(recipeDiff(form, "full-physics")).toEqual(["n_coll 4096"]);
  });

  it("formats the validation split as a percentage", () => {
    const form = { ...recipeState("full-physics"), val_fraction: 0.2 };
    expect(recipeDiff(form, "full-physics")).toEqual(["val split 20%"]);
  });

  it("ignores a sub-parameter whose governing switch is off", () => {
    // An inert value rides the request unchanged in effect; a chip for it
    // would claim a departure the run does not have.
    const form = { ...recipeState("full-physics"), kin_margin_frac: 0.5 };
    expect(recipeDiff(form, "full-physics")).toEqual([]);
  });

  it("reports the sub-parameter once its switch is on", () => {
    const form = {
      ...recipeState("full-physics"),
      kinematics: true,
      kin_margin_frac: 0.5,
    };
    expect(recipeDiff(form, "full-physics")).toEqual([
      "+ growth constraints",
      "growth margin 0.5",
    ]);
  });

  it("reports turning a stamped setting off against the held-out recipe", () => {
    const form = { ...recipeState("best-heldout"), causal_weighting: false };
    expect(recipeDiff(form, "best-heldout")).toEqual(["− causal weighting"]);
  });

  it("reports a physics choice as one 'label → option' chip", () => {
    const form = {
      ...recipeState("full-physics"),
      momentum_closure: "brinkman" as const,
      wall_condition: "no_penetration" as const,
    };
    expect(recipeDiff(form, "full-physics")).toEqual([
      "wall condition → no penetration",
      "momentum closure → brinkman",
    ]);
  });

  it("reports the causal gate as a switch, with its threshold gated on it", () => {
    const gated = {
      ...recipeState("best-heldout"),
      causal_gate_min_weight: 0.9,
    };
    expect(recipeDiff(gated, "best-heldout")).toEqual([
      "+ causal gate",
      "causal gate 0.9",
    ]);
    // Stepping the trusted recipe's gate off reads as one chip, not a value.
    const off = { ...recipeState("trusted"), causal_gate_min_weight: null };
    expect(recipeDiff(off, "trusted")).toEqual(["− causal gate"]);
  });

  it("does not re-report the choices a treatment change clears by construction", () => {
    // Stepping the trusted recipe down to the explicit front resets the
    // two-phase pressure and the sharp thermal model, which the rung cannot
    // carry; those are not user choices and get no chip.
    const form = {
      ...recipeState("trusted"),
      sharp_interface: false,
      film_pressure: false,
      root_attachment: false,
      receding_cap: false,
      pressure_representation: "single" as const,
      thermal_model: "legacy" as const,
      source_mode: "free" as const,
    };
    expect(recipeDiff(form, "trusted")).toEqual(["treatment → explicit front"]);
  });

  it("reports a weighting scheme change by name", () => {
    const form = {
      ...recipeState("full-physics"),
      weighting: "rba" as const,
    };
    expect(recipeDiff(form, "full-physics")).toEqual(["weighting → RBA"]);
  });

  it("reports sharpening as a switch, with its target gated on it", () => {
    const off = { ...recipeState("full-physics"), alpha_eps_final: 0.05 };
    expect(recipeDiff(off, "full-physics")).toEqual([]);
    const on = {
      ...recipeState("full-physics"),
      alpha_eps_anneal_steps: 600,
      alpha_eps_final: 0.05,
    };
    expect(recipeDiff(on, "full-physics")).toEqual([
      "+ sharpening",
      "final αε 0.05",
    ]);
  });
});

describe("SWITCH_LABELS completeness", () => {
  it("covers every modifier key, so no departure can silently drop its chip", async () => {
    const { MODIFIERS } = await import("../src/views/solver/formulation");
    const { SWITCH_LABELS } = await import("../src/views/solver/recipes");
    const labelled = new Set(SWITCH_LABELS.map(([key]) => key));
    for (const modifier of MODIFIERS) {
      expect(
        labelled.has(modifier.key),
        `no diff-chip label for ${modifier.key}`,
      ).toBe(true);
    }
  });
});
