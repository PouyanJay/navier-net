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
  PRESETS,
  type PresetName,
  STAGE_A_FIELDS,
} from "./model";

interface Globals {
  ff: number;
  ffScale: number;
  alphaEps: number;
  nodewise: boolean;
}

interface EditState {
  dataset: string;
  equations: EquationState[];
  groups: Record<string, number>;
  enabled: Record<string, boolean>; // toggleable equation id -> on
  weights: Record<string, number>; // weight_key -> value
  preset: PresetName;
  perField: Record<FieldName, FieldArch>;
  globals: Globals;
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
  preset: PresetName;
  perField: Record<FieldName, FieldArch>;
  globals: Globals;
  activeFields: FieldName[];
  fieldOn: (field: FieldName) => boolean;
  fieldParamCount: (field: FieldName) => number;
  totalParams: number;
  overrideCount: number;
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

function detectPreset(model: ModelArchitecture): PresetName {
  const match = (Object.keys(PRESETS) as PresetName[]).find((name) => {
    const p = PRESETS[name];
    return (
      model.hidden === p.width &&
      model.layers === p.depth &&
      model.fourier_feats === p.ff &&
      model.fourier_scale === p.ffScale
    );
  });
  return match ?? "medium";
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
      (f) => FIELDS[f].needs === null || edit.enabled[FIELDS[f].needs as string],
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
    const base = edit.perField.phi;
    const per_field: Record<string, { hidden: number; layers: number }> = {};
    for (const f of ALL_FIELDS) {
      const a = edit.perField[f];
      if (a.width !== base.width || a.depth !== base.depth) {
        per_field[f] = { hidden: a.width, layers: a.depth };
      }
    }
    Promise.all([
      api.updatePhysics(edit.dataset, { enabled: enabledIds, weights: edit.weights }),
      api.updateModel(edit.dataset, {
        hidden: base.width,
        layers: base.depth,
        fourier_feats: edit.globals.ff,
        fourier_scale: edit.globals.ffScale,
        alpha_eps: edit.globals.alphaEps,
        per_field,
      }),
    ])
      .then(([physics, model]) => setEdit(buildEdit(model, physics)))
      .catch((err) => setSaveError(errorMessage(err)))
      .finally(() => setSaving(false));
  }, [edit]);

  if (dataset == null || (!edit && !error)) return { status: "loading" };
  if (error) return { status: "error", message: error };
  if (!edit) return { status: "loading" };

  const fieldOn = (f: FieldName) =>
    FIELDS[f].needs === null || edit.enabled[FIELDS[f].needs as string];
  const fieldParamCount = (f: FieldName) =>
    fieldParams(edit.perField[f], edit.globals.ff, edit.globals.nodewise);
  const totalParams = activeFields.reduce((sum, f) => sum + fieldParamCount(f), 0);

  const derived = deriveFromPreset(edit.preset);
  const overrideCount = countOverrides(edit, derived);

  const equations: EquationDisplay[] = edit.equations.map((e) => ({
    ...e,
    on: e.core ? true : (edit.enabled[e.id] ?? e.enabled),
    liveWeight: edit.weights[e.weight_key] ?? e.weight,
    toggleable: TOGGLEABLE.has(e.id),
  }));

  const model: PhysicsModel = {
    dataset: edit.dataset,
    equations,
    groups: edit.groups,
    preset: edit.preset,
    perField: edit.perField,
    globals: edit.globals,
    activeFields,
    fieldOn,
    fieldParamCount,
    totalParams,
    overrideCount,
    dirty: edit.dirty,
    saving,
    saveError,
    toggleEquation: (id) => {
      if (!TOGGLEABLE.has(id)) return;
      patch({ enabled: { ...edit.enabled, [id]: !edit.enabled[id] } });
    },
    setWeight: (weightKey, value) =>
      patch({ weights: { ...edit.weights, [weightKey]: value } }),
    applyPreset: (preset) => {
      const d = deriveFromPreset(preset);
      patch({ preset, perField: d.perField, globals: { ...edit.globals, ...d.globals } });
    },
    setGlobal: (key, value) => patch({ globals: { ...edit.globals, [key]: value } }),
    setFieldArch: (field, key, value) =>
      patch({ perField: { ...edit.perField, [field]: { ...edit.perField[field], [key]: value } } }),
    resetToPreset: () => {
      const d = deriveFromPreset(edit.preset);
      patch({ perField: d.perField, globals: { ...edit.globals, ...d.globals } });
    },
    hydraCommand: buildHydra(edit, activeFields, base(edit)),
    save,
  };
  return { status: "ready", model };
}

function base(edit: EditState): FieldArch {
  return edit.perField.phi;
}

function deriveFromPreset(preset: PresetName): {
  perField: Record<FieldName, FieldArch>;
  globals: Pick<Globals, "ff" | "ffScale">;
} {
  const p = PRESETS[preset];
  const perField = {} as Record<FieldName, FieldArch>;
  for (const f of ALL_FIELDS) {
    const stageB = FIELDS[f].stage === "B";
    perField[f] = {
      width: stageB ? Math.round(p.width * 1.33) : p.width,
      depth: stageB ? p.depth + 2 : p.depth,
    };
  }
  return { perField, globals: { ff: p.ff, ffScale: p.ffScale } };
}

function countOverrides(
  edit: EditState,
  derived: ReturnType<typeof deriveFromPreset>,
): number {
  let n = 0;
  if (edit.globals.ff !== derived.globals.ff) n++;
  if (edit.globals.ffScale !== derived.globals.ffScale) n++;
  for (const f of ALL_FIELDS) {
    if (edit.perField[f].width !== derived.perField[f].width) n++;
    if (edit.perField[f].depth !== derived.perField[f].depth) n++;
  }
  return n;
}

function buildHydra(edit: EditState, activeFields: FieldName[], baseArch: FieldArch): string {
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
  for (const [key, value] of Object.entries(edit.weights)) {
    if (value !== 1.0) parts.push(`training.weights.${key}=${value}`);
  }
  return `naviernet train ${parts.join(" ")}`;
}

export { STAGE_A_FIELDS };
