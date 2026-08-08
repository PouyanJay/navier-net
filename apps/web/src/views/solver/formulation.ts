/** The interface formulation: which treatment a run gives the front, what that
 * costs in momentum, and which options each treatment admits.
 *
 * The trainer exposes this as four independent booleans (`front_geometry`,
 * `sharp_interface`, `film_pressure`, `hard_pin`) with a dependency tree between
 * them, which the form used to render as a flat pile of switches whose hints
 * said "needs front geometry". There are really only THREE coherent
 * formulations, and they form a ladder — each one a superset of the last. This
 * module is that ladder, so the aside can offer a choice instead of a puzzle.
 */

import type { SolverFormState } from "./form";

export type Treatment = "diffuse" | "front" | "sharp";

export interface TreatmentMeta {
  name: Treatment;
  label: string;
  /** The mono sub-note on the segment: the treatment in three words. */
  note: string;
  /** What closes momentum under this treatment — the thing that actually
   * changes between them, and the reason to pick one. */
  momentum: string;
  /** What the interface IS here, in one sentence. */
  detail: string;
  /** A model field this treatment reads, which the series must therefore train. */
  needs?: "p";
}

/** The ladder, in order of how much structure it imposes on the front. */
export const TREATMENTS: TreatmentMeta[] = [
  {
    name: "diffuse",
    label: "Diffuse",
    note: "α = σ(φ/ε)",
    momentum: "2-D depth-averaged momentum (Hele-Shaw), over the whole field",
    detail:
      "No explicit front: the interface is wherever the volume fraction crosses ½. Nothing holds the root, so pin it below if it drifts.",
  },
  {
    name: "front",
    label: "Explicit front",
    note: "capsule",
    momentum: "2-D depth-averaged momentum (Hele-Shaw), over the whole field",
    detail:
      "The front is a capsule: exact root, monotone nose, one connected shape at every t. The construction pins the root, so no pin term is needed.",
  },
  {
    name: "sharp",
    label: "Sharp front",
    note: "Darcy · Young-Laplace",
    momentum: "depth-averaged Darcy drag, imposed ON the front",
    detail:
      "The capsule front carrying the conditions that hold at the interface: the Young-Laplace pressure jump and the kinematic condition. Darcy replaces the 2-D momentum residual, so the pressure field is read at the front.",
    needs: "p",
  },
];

export const treatmentMeta = (name: Treatment): TreatmentMeta =>
  TREATMENTS.find((t) => t.name === name) ?? TREATMENTS[0];

/** Which treatment the current flags amount to. */
export function treatmentOf(form: SolverFormState): Treatment {
  if (form.sharp_interface) return "sharp";
  return form.front_geometry ? "front" : "diffuse";
}

/**
 * The flags one treatment implies, including the ones it must clear.
 *
 * Every option below the ladder rung is switched off rather than left dangling:
 * a diffuse run has no front to pinch off, and a run without the interface
 * conditions has no Young-Laplace jump for the film pressure to correct — the
 * same combinations the trainer and the API reject.
 */
export function treatmentPatch(name: Treatment): Partial<SolverFormState> {
  if (name === "diffuse") {
    return {
      front_geometry: false,
      sharp_interface: false,
      film_pressure: false,
      liquid_film: false,
      allow_pinch: false,
      evolving_width: false,
      cap_freedom: false,
      front_velocity: false,
    };
  }
  if (name === "front") {
    return {
      front_geometry: true,
      hard_pin: false,
      sharp_interface: false,
      film_pressure: false,
      liquid_film: false,
    };
  }
  // The sharp rung comes with its recommended companion: the film-pressure
  // correction is part of that recipe (and is the platform default), so
  // choosing the treatment chooses the recipe. It stays switchable below.
  // The liquid film is NOT switched on with the rung: it is opt-in until
  // benched, so choosing sharp does not silently change the reproducible recipe.
  return {
    front_geometry: true,
    hard_pin: false,
    sharp_interface: true,
    film_pressure: true,
  };
}

/** A boolean that refines the chosen treatment, offered only where it applies. */
export interface Modifier {
  key:
    | "hard_pin"
    | "allow_pinch"
    | "evolving_width"
    | "film_pressure"
    | "liquid_film"
    | "cap_freedom";
  label: string;
  hint: string;
  /** The rungs that admit it; on any other rung it is not shown at all. */
  on: Treatment[];
}

export const MODIFIERS: Modifier[] = [
  {
    key: "hard_pin",
    label: "Hard root pin",
    hint: "anchor the root at the measured cavity",
    on: ["diffuse"],
  },
  {
    key: "film_pressure",
    label: "Film pressure",
    hint: "Bretherton film offset on the bubble's sides",
    on: ["sharp"],
  },
  {
    key: "liquid_film",
    label: "Liquid film δ(x,t)",
    hint: "deposited · evaporating · dryout · its source and its pressure",
    on: ["sharp"],
  },
  {
    // First among the shape modifiers: it is the one that decides whether the
    // bubble's ends can have a shape at all.
    key: "cap_freedom",
    label: "Free the end caps",
    hint: "the nose is shaped, not a circle of fixed curvature",
    on: ["front", "sharp"],
  },
  {
    key: "evolving_width",
    label: "Evolving width",
    hint: "the profile keeps deforming past the data",
    on: ["front", "sharp"],
  },
  {
    key: "allow_pinch",
    label: "Allow pinch-off",
    hint: "signed radius · the bubble may detach",
    on: ["front", "sharp"],
  },
];

export const modifiersFor = (treatment: Treatment): Modifier[] =>
  MODIFIERS.filter((m) => m.on.includes(treatment));
