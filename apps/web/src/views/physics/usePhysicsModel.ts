import { useCallback, useEffect, useMemo, useState } from "react";

import {
  api,
  type EquationState,
  type ModelArchitecture,
  type PhysicsState,
} from "../../lib/api";
import { errorMessage } from "../../lib/errors";
import {
  ALL_FIELDS,
  type FieldArch,
  type FieldName,
  fieldParams,
  FIELDS,
  derivePerField,
  PRESETS,
  type PresetName,
} from "./model";

interface Globals {
  ff: number;
  ffScale: number;
  alphaEps: number;
  nodewise: boolean;
}

/** A physics caution, carried to the control that caused it. */
export interface Warning {
  /** The global this warning belongs beside, or null for a whole-model one. */
  at: "ffScale" | "alphaEps" | "width" | null;
  message: string;
}

/**
 * Which interface treatment the composed run gets, and why.
 *
 * NOT owned here. `model.sharp_interface` is resolved at run launch, where an
 * unset request takes the recipe wherever the series fits (`p` is trained).
 * The run launcher appends `series_overrides()` AFTER its own list and Hydra's
 * last override wins, so a per-series copy of this flag would silently
 * outrank the Solver's explicit choice. This page states what will happen and
 * names where to change it.
 */
export interface Formulation {
  sharp: boolean;
  /** The equation ids this treatment admits that the other would not. */
  admits: string[];
  reason: string;
}

interface Baseline {
  perField: Record<FieldName, FieldArch>;
  globals: Pick<Globals, "ff" | "ffScale">;
}

interface EditState {
  dataset: string;
  equations: EquationState[];
  groups: Record<string, number>;
  enabled: Record<string, boolean>; // toggleable equation id -> on
  weights: Record<string, number>; // weight_key -> value
  preset: PresetName | null; // the matching preset, or null when custom
  perField: Record<FieldName, FieldArch>;
  globals: Globals;
  // The width/depth/Fourier values an "override" is measured against: the
  // loaded config, or whatever a later preset re-derived. So a freshly loaded
  // config reads as 0 overrides even when it matches no preset.
  baseline: Baseline;
  dirty: boolean;
}

export type PhysicsLoad =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; model: PhysicsModel };

/** Everything the Physics & Model panels read and drive. */
export interface PhysicsModel {
  dataset: string;
  equations: EquationDisplay[];
  groups: Record<string, number>;
  preset: PresetName | null;
  perField: Record<FieldName, FieldArch>;
  globals: Globals;
  activeFields: FieldName[];
  fieldOn: (field: FieldName) => boolean;
  fieldParamCount: (field: FieldName) => number;
  fieldOverridden: (field: FieldName) => boolean;
  globalOverridden: (key: "ff" | "ffScale") => boolean;
  totalParams: number;
  overrideCount: number;
  formulation: Formulation;
  warnings: Warning[];
  dirty: boolean;
  saving: boolean;
  saveError: string | null;
  toggleEquation: (id: string) => void;
  setWeight: (weightKey: string, value: number) => void;
  applyPreset: (preset: PresetName) => void;
  setGlobal: <K extends keyof Globals>(key: K, value: Globals[K]) => void;
  setFieldArch: (field: FieldName, key: keyof FieldArch, value: number) => void;
  resetToPreset: () => void;
  hydraCommand: string;
  save: () => void;
}

export interface EquationDisplay extends EquationState {
  /** live enabled state (overlays the loaded value with local edits) */
  on: boolean;
  /** Whether the current interface treatment admits this equation at all. A
   * diffuse run never trains the sharp trio, however many fields exist. */
  admitted: boolean;
  /** live weight (overlays the loaded value) */
  liveWeight: number;
  toggleable: boolean;
}

const TOGGLEABLE = new Set(["mom", "energy"]);

function baseFrom(model: ModelArchitecture, field: FieldName): FieldArch {
  const over = model.per_field[field];
  return {
    width: over?.hidden ?? model.hidden,
    depth: over?.layers ?? model.layers,
  };
}

function detectPreset(model: ModelArchitecture): PresetName | null {
  return (
    (Object.keys(PRESETS) as PresetName[]).find((name) => {
      const p = PRESETS[name];
      return (
        model.hidden === p.width &&
        model.layers === p.depth &&
        model.fourier_feats === p.ff &&
        model.fourier_scale === p.ffScale
      );
    }) ?? null
  );
}

function buildEdit(model: ModelArchitecture, physics: PhysicsState): EditState {
  const enabled: Record<string, boolean> = {};
  const weights: Record<string, number> = {};
  for (const e of physics.equations) {
    weights[e.weight_key] = e.weight;
    if (TOGGLEABLE.has(e.id)) enabled[e.id] = e.enabled;
  }
  const perField = {} as Record<FieldName, FieldArch>;
  for (const f of ALL_FIELDS) perField[f] = baseFrom(model, f);
  return {
    dataset: physics.dataset,
    equations: physics.equations,
    groups: physics.groups,
    enabled,
    weights,
    preset: detectPreset(model),
    perField,
    globals: {
      ff: model.fourier_feats,
      ffScale: model.fourier_scale,
      alphaEps: model.alpha_eps,
      nodewise: model.nodewise_activation,
    },
    baseline: {
      perField: structuredClone(perField),
      globals: { ff: model.fourier_feats, ffScale: model.fourier_scale },
    },
    dirty: false,
  };
}

export function usePhysicsModel(dataset: string | null): PhysicsLoad {
  const [edit, setEdit] = useState<EditState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    if (dataset == null) return;
    let alive = true;
    setEdit(null);
    setError(null);
    Promise.all([api.getModel(dataset), api.getPhysics(dataset)])
      .then(([model, physics]) => alive && setEdit(buildEdit(model, physics)))
      .catch((err) => alive && setError(errorMessage(err)));
    return () => {
      alive = false;
    };
  }, [dataset]);

  const activeFields = useMemo<FieldName[]>(() => {
    if (!edit) return [];
    return ALL_FIELDS.filter(
      (f) =>
        FIELDS[f].needs === null || edit.enabled[FIELDS[f].needs as string],
    );
  }, [edit]);

  const patch = useCallback((next: Partial<EditState>) => {
    setEdit((cur) => (cur ? { ...cur, ...next, dirty: true } : cur));
  }, []);

  const save = useCallback(() => {
    if (!edit) return;
    setSaving(true);
    setSaveError(null);
    const enabledIds = [...TOGGLEABLE].filter((id) => edit.enabled[id]);
    // Only Stage-B weights are owned here; Stage-A weights are set at run launch.
    const weights: Record<string, number> = {};
    for (const e of edit.equations) {
      if (e.stage === "B")
        weights[e.weight_key] = edit.weights[e.weight_key] ?? e.weight;
    }
    const base = edit.perField.phi;
    const per_field: Record<string, { hidden: number; layers: number }> = {};
    for (const f of ALL_FIELDS) {
      const a = edit.perField[f];
      if (a.width !== base.width || a.depth !== base.depth) {
        per_field[f] = { hidden: a.width, layers: a.depth };
      }
    }
    // Save physics first, then the architecture: both write the same model.json,
    // so they must be sequential (the model save merges onto the physics save).
    api
      .updatePhysics(edit.dataset, { enabled: enabledIds, weights })
      .then((physics) =>
        api
          .updateModel(edit.dataset, {
            hidden: base.width,
            layers: base.depth,
            fourier_feats: edit.globals.ff,
            fourier_scale: edit.globals.ffScale,
            alpha_eps: edit.globals.alphaEps,
            per_field,
          })
          .then((model) => setEdit(buildEdit(model, physics))),
      )
      .catch((err) => setSaveError(errorMessage(err)))
      .finally(() => setSaving(false));
  }, [edit]);

  if (dataset == null || (!edit && !error)) return { status: "loading" };
  if (error) return { status: "error", message: error };
  if (!edit) return { status: "loading" };

  const model: PhysicsModel = {
    dataset: edit.dataset,
    groups: edit.groups,
    preset: edit.preset,
    perField: edit.perField,
    globals: edit.globals,
    activeFields,
    dirty: edit.dirty,
    saving,
    saveError,
    hydraCommand: buildHydra(edit, activeFields, edit.perField.phi),
    formulation: formulationOf(activeFields.includes("p")),
    warnings: validate(edit.globals, edit.perField, activeFields),
    save,
    ...deriveView(edit, activeFields),
    ...buildActions(edit, patch),
  };
  return { status: "ready", model };
}

/* Validation thresholds, named rather than inlined in the checks. */
const MIN_ALPHA_EPS = 0.005; // interface half-width the image supervision resolves
const SMALL_EPS = 0.03; // below this, sigma_B must be large enough to resolve it
const MIN_FF_SCALE_FOR_SMALL_EPS = 2.5;
const MAX_STABLE_WIDTH = 256; // beyond this, CPU step cost grows quadratically

/**
 * Physics cautions, each tagged with the control that caused it.
 *
 * Advice, never a block: these are judgements about resolvability, not invalid
 * values, and a researcher may well know better than the heuristic.
 */
function validate(
  globals: Globals,
  perField: Record<FieldName, FieldArch>,
  activeFields: FieldName[],
): Warning[] {
  const out: Warning[] = [];
  if (globals.alphaEps < MIN_ALPHA_EPS) {
    out.push({
      at: "alphaEps",
      message: `ε ${globals.alphaEps} is very small; the image supervision may not resolve it. Raise ε, or bin the frames.`,
    });
  }
  if (
    globals.ffScale < MIN_FF_SCALE_FOR_SMALL_EPS &&
    globals.alphaEps <= SMALL_EPS
  ) {
    out.push({
      at: "ffScale",
      message: `σ_B ${globals.ffScale} is likely too low to resolve ε ${globals.alphaEps}; the α network will smooth the interface. Raise σ_B toward ${MIN_FF_SCALE_FOR_SMALL_EPS}, or ε back above ${SMALL_EPS}.`,
    });
  }
  if (activeFields.some((f) => perField[f].width > MAX_STABLE_WIDTH)) {
    out.push({
      at: "width",
      message: `Width above ${MAX_STABLE_WIDTH}: CPU step cost grows quadratically. Plan for a GPU device in the Solver.`,
    });
  }
  return out;
}

/**
 * Which interface treatment a launched run gets, and why.
 *
 * Predicted from the field set, NOT from the composed `model.sharp_interface`:
 * that flag has no per-series override, so it always reads the schema default
 * here, while the launcher resolves it against what the series trains. The
 * Solver sends sharp by default and the API rejects that ask when there is no
 * pressure field, so a series without `p` is the case worth naming — the fix
 * for it lives on this page.
 */
function formulationOf(hasPressure: boolean): Formulation {
  if (hasPressure) {
    return {
      sharp: true,
      admits: ["darcy", "kinematic", "laplace"],
      reason:
        "Pressure is trained, so a launch imposes the interface conditions on the explicit front: Darcy drag, the kinematic condition, and the Young–Laplace jump.",
    };
  }
  return {
    sharp: false,
    admits: ["mom"],
    reason:
      "No pressure field, so there is no front pressure to impose a jump on. Enable Momentum above to reach the sharp treatment; until then the Solver's sharp-interface option will refuse this series.",
  };
}

/** The read-only derived values a view needs: field helpers, counts, and the
 * equation display list. Pure in ``edit`` + ``activeFields``. */
function deriveView(edit: EditState, activeFields: FieldName[]) {
  const fieldOn = (f: FieldName) =>
    FIELDS[f].needs === null || edit.enabled[FIELDS[f].needs as string];
  const fieldParamCount = (f: FieldName) =>
    fieldParams(edit.perField[f], edit.globals.ff, edit.globals.nodewise);
  const fieldOverridden = (f: FieldName) =>
    edit.perField[f].width !== edit.baseline.perField[f].width ||
    edit.perField[f].depth !== edit.baseline.perField[f].depth;
  const globalOverridden = (key: "ff" | "ffScale") =>
    edit.globals[key] !== edit.baseline.globals[key];
  const overrideCount =
    ALL_FIELDS.reduce(
      (n, f) =>
        n +
        (edit.perField[f].width !== edit.baseline.perField[f].width ? 1 : 0) +
        (edit.perField[f].depth !== edit.baseline.perField[f].depth ? 1 : 0),
      0,
    ) +
    (globalOverridden("ff") ? 1 : 0) +
    (globalOverridden("ffScale") ? 1 : 0);
  // An equation is active when the formulation admits it AND every field it
  // needs is present -- derived live from the toggles, so coupled terms (the
  // evaporation closure, which rides on temperature) light up with the
  // equation that unlocks their field, and the sharp trio appears only under
  // the sharp treatment.
  const activeSet = new Set<string>(activeFields);
  const predictedSharp = activeFields.includes("p");
  const admitted = (mode: EquationState["mode"] | undefined) =>
    mode == null ||
    mode === "any" ||
    mode === (predictedSharp ? "sharp" : "diffuse");
  const equations: EquationDisplay[] = edit.equations.map((e) => ({
    ...e,
    on: admitted(e.mode) && e.fields_required.every((f) => activeSet.has(f)),
    admitted: admitted(e.mode),
    liveWeight: edit.weights[e.weight_key] ?? e.weight,
    toggleable: TOGGLEABLE.has(e.id),
  }));
  return {
    fieldOn,
    fieldParamCount,
    fieldOverridden,
    globalOverridden,
    overrideCount,
    equations,
    totalParams: activeFields.reduce((sum, f) => sum + fieldParamCount(f), 0),
  };
}

/** The edit actions, each producing a dirty patch over ``edit``. */
function buildActions(
  edit: EditState,
  patch: (next: Partial<EditState>) => void,
) {
  return {
    toggleEquation: (id: string) => {
      if (!TOGGLEABLE.has(id)) return;
      patch({ enabled: { ...edit.enabled, [id]: !edit.enabled[id] } });
    },
    setWeight: (weightKey: string, value: number) =>
      patch({ weights: { ...edit.weights, [weightKey]: value } }),
    applyPreset: (preset: PresetName) => {
      const d = deriveFromPreset(preset);
      // The preset becomes the new baseline: re-derived values read as 0 overrides.
      patch({
        preset,
        perField: d.perField,
        globals: { ...edit.globals, ...d.globals },
        baseline: { perField: structuredClone(d.perField), globals: d.globals },
      });
    },
    setGlobal: <K extends keyof EditState["globals"]>(
      key: K,
      value: EditState["globals"][K],
    ) => patch({ globals: { ...edit.globals, [key]: value } }),
    setFieldArch: (field: FieldName, key: keyof FieldArch, value: number) =>
      patch({
        perField: {
          ...edit.perField,
          [field]: { ...edit.perField[field], [key]: value },
        },
      }),
    resetToPreset: () =>
      patch({
        perField: structuredClone(edit.baseline.perField),
        globals: { ...edit.globals, ...edit.baseline.globals },
      }),
  };
}

function deriveFromPreset(preset: PresetName): Baseline {
  const p = PRESETS[preset];
  return {
    perField: derivePerField(p),
    globals: { ff: p.ff, ffScale: p.ffScale },
  };
}

function buildHydra(
  edit: EditState,
  activeFields: FieldName[],
  baseArch: FieldArch,
): string {
  const parts = [
    `dataset=${edit.dataset}`,
    `model.fields=[${activeFields.join(",")}]`,
    `model.hidden=${baseArch.width}`,
    `model.layers=${baseArch.depth}`,
    `model.fourier_feats=${edit.globals.ff}`,
    `model.fourier_scale=${edit.globals.ffScale}`,
    `model.alpha_eps=${edit.globals.alphaEps}`,
  ];
  const perField: string[] = [];
  for (const f of activeFields) {
    const a = edit.perField[f];
    if (a.width !== baseArch.width || a.depth !== baseArch.depth) {
      perField.push(`${f}:{hidden:${a.width},layers:${a.depth}}`);
    }
  }
  if (perField.length) parts.push(`+model.per_field={${perField.join(",")}}`);
  for (const eq of edit.equations) {
    if (eq.stage === "B") {
      const w = edit.weights[eq.weight_key] ?? eq.weight;
      if (w !== 1.0) parts.push(`training.weights.${eq.weight_key}=${w}`);
    }
  }
  return `naviernet train ${parts.join(" ")}`;
}
