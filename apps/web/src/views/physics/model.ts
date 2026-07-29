/** Client-side model derivations for the Physics & Model builder.
 *
 * The backend owns persistence (fields, weights, per_field); these are the UI's
 * live derivations — capacity presets, parameter counts, and budget estimates —
 * ported from the platform mockup. Parameter counts are exact; the time and
 * memory figures are deliberately labelled estimates.
 */

export type FieldName = "phi" | "u" | "v" | "s" | "p" | "T";

export interface FieldMeta {
  label: string;
  hue: string; // a --f-* token, live on the dark canvas
  transform: string;
  stage: "A" | "B";
  /** The equation id that unlocks this field, or null for Stage-A fields. */
  needs: string | null;
}

export const FIELDS: Record<FieldName, FieldMeta> = {
  phi: {
    label: "φ→α",
    hue: "var(--f-phi)",
    transform: "sigmoid(φ/ε)",
    stage: "A",
    needs: null,
  },
  u: {
    label: "u",
    hue: "var(--f-u)",
    transform: "identity",
    stage: "A",
    needs: null,
  },
  v: {
    label: "v",
    hue: "var(--f-v)",
    transform: "identity",
    stage: "A",
    needs: null,
  },
  s: {
    label: "s",
    hue: "var(--f-s)",
    transform: "softplus ≥ 0",
    stage: "A",
    needs: null,
  },
  p: {
    label: "p",
    hue: "var(--f-p)",
    transform: "zero-mean gauge",
    stage: "B",
    needs: "mom",
  },
  T: {
    label: "T",
    hue: "var(--f-t)",
    transform: "bounded [T_in, T_wall]",
    stage: "B",
    needs: "energy",
  },
};

export const STAGE_A_FIELDS: FieldName[] = ["phi", "u", "v", "s"];
export const ALL_FIELDS: FieldName[] = ["phi", "u", "v", "s", "p", "T"];

export type PresetName = "small" | "medium" | "large";

export interface Preset {
  width: number;
  depth: number;
  ff: number; // Fourier feature pairs
  ffScale: number; // σ_B, the Fourier feature scale
}

/** Medium is the exact midpoint of Small and Large on every axis. */
export const PRESETS: Record<PresetName, Preset> = {
  small: { width: 64, depth: 3, ff: 32, ffScale: 2.0 },
  medium: { width: 112, depth: 6, ff: 80, ffScale: 3.0 },
  large: { width: 160, depth: 9, ff: 128, ffScale: 4.0 },
};

/** Stiffer Stage-B residuals get more capacity than the Stage-A fields. */
export const STAGE_B_SCALE = { width: 1.33, depth: 2 };

export interface FieldArch {
  width: number;
  depth: number;
}

/** The per-field width/depth a preset derives (Stage-B fields scaled up). */
export function derivePerField(preset: Preset): Record<FieldName, FieldArch> {
  const out = {} as Record<FieldName, FieldArch>;
  for (const f of ALL_FIELDS) {
    const stageB = FIELDS[f].stage === "B";
    out[f] = {
      width: stageB
        ? Math.round(preset.width * STAGE_B_SCALE.width)
        : preset.width,
      depth: stageB ? preset.depth + STAGE_B_SCALE.depth : preset.depth,
    };
  }
  return out;
}

/** Exact trainable-parameter count for one field network. */
export function fieldParams(
  arch: FieldArch,
  ff: number,
  nodewise: boolean,
): number {
  const { width: w, depth: d } = arch;
  let n = 2 * ff * w + w; // Fourier features -> first hidden
  for (let i = 1; i < d; i++) n += w * w + w; // hidden stack
  n += w + 1; // output head
  if (nodewise) n += w * d; // adaptive-tanh slopes
  return n;
}

// Measured Stage-A baseline, for the (clearly-labelled) budget estimates.
const BASE = { params: 163_000, msPerStep: 190 };

export function estMinutesPer1kSteps(params: number): number {
  return ((params / BASE.params) * BASE.msPerStep * 1000) / 60_000;
}

export function estMemoryMB(params: number): number {
  return (params * 8 * 3) / 1e6; // fp64 x (params + Adam m,v)
}

export function fmtCount(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(2)} M` : `${(n / 1e3).toFixed(0)} k`;
}
