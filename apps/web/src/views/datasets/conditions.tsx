import { useEffect, useId, useState } from "react";

import { Callout, Select } from "../../components";
import {
  api,
  type ConditionsUpdate,
  type Fluid,
  type OperatingConditions,
} from "../../lib/api";
import { errorMessage } from "../../lib/errors";
import {
  fieldByName,
  fieldOptions,
  fieldSpecs,
  type FieldSpec,
} from "../../lib/fields";

export const DEFAULT_FLUID_ID = "fc72";

/** A condition's short name: the field table's `name` (the last segment of
 * its Hydra key), which is also the key the conditions API takes. */
export type ConditionKey = string;

/** Condition fields baked into the preprocessed tensors: editing one on a
 * processed series needs a re-preprocess. Read off the field table, which the
 * pipeline's own baked-field list is generated from. */
export const BAKED_KEYS: ReadonlySet<ConditionKey> = new Set(
  fieldSpecs({ section: "conditions" })
    .filter((spec) => spec.baked)
    .map((spec) => spec.name),
);

export interface ConditionSpec {
  key: ConditionKey;
  label: string;
  hint: string;
  unit: string;
  step: number;
  /** Server-side bounds, from the field table. */
  min: number;
  max: number;
  /** Shown when empty (measurements) or pre-filled as the value (operating). */
  placeholder?: string;
  why: string;
  /** A number, or a choice among `options`. */
  kind: "number" | "select";
  options?: { value: string; label: string }[];
  /** Empty is a value: "not set", the derived default. */
  optional: boolean;
  /** What a new series starts with: "" must be entered (or stays not set). */
  initial: string;
}

/** The table's unit, in the interface's typographic register. */
const UNIT_DISPLAY: Record<string, string> = {
  "mL/h": "mL·hr⁻¹",
  "W/cm²": "W·cm⁻²",
  "m/s": "m·s⁻¹",
};

/** A sensible input step for a unit, where the table does not say. */
const STEP_BY_UNIT: Record<string, number> = {
  "°C": 0.1,
  "W/cm²": 0.1,
  µm: 10,
  ms: 0.1,
  K: 0.1,
  "": 0.05,
};

/** The form spec of one field-table condition, with the hints the table does
 * not carry (a step, a hint, a placeholder) overlaid. */
function fromTable(
  name: string,
  extra: Partial<ConditionSpec> = {},
): ConditionSpec {
  const spec = fieldByName(name);
  return {
    key: spec.name,
    label: spec.label,
    hint: spec.optional ? "optional" : "",
    unit: UNIT_DISPLAY[spec.unit] ?? spec.unit,
    step: STEP_BY_UNIT[spec.unit] ?? 0.1,
    min: spec.lo ?? -Infinity,
    max: spec.hi ?? Infinity,
    why: spec.help,
    kind: spec.kind === "enum" ? "select" : "number",
    options: spec.kind === "enum" ? fieldOptions(spec) : undefined,
    optional: spec.optional,
    initial: spec.default == null ? "" : String(spec.default),
    ...extra,
  };
}

/**
 * The fixed geometry and timing of a run. Frame interval and channel width are
 * baked into the tensors (time axis and µm/px); channel height only feeds the
 * dimensionless groups. They start empty -- there is no honest default for a
 * series nobody has described yet.
 */
export const REQUIRED_CONDITIONS: ConditionSpec[] = [
  fromTable("dt_frame_ms", {
    hint: "Δt between frames",
    step: 0.1,
    placeholder: "0.5",
  }),
  fromTable("channel_width_um", {
    hint: "imaged direction",
    placeholder: "300",
  }),
  fromTable("channel_height_um", {
    hint: "heated wall below",
    placeholder: "150",
  }),
];

/**
 * The operating conditions of the run. They have sensible defaults, so they
 * start pre-filled and are edited to override; they set the reference scales and
 * the dimensionless groups. Saturation temperature is not here: it is a property
 * of the selected fluid (derived, read-only).
 */
export const OPERATING_CONDITIONS: ConditionSpec[] = [
  fromTable("q_wall_W_cm2", { hint: "baseline", initial: "2.0" }),
  fromTable("flow_rate_mL_hr", { hint: "inlet", step: 0.5, initial: "5.0" }),
  fromTable("U_ref", { hint: "U_ref", step: 0.01, initial: "0.20" }),
];

const HAND_WRITTEN = new Set(
  [...REQUIRED_CONDITIONS, ...OPERATING_CONDITIONS].map((spec) => spec.key),
);

/**
 * Every further series condition the field table declares, rendered straight
 * from it so a new input is one table entry. Today these are the thermal
 * conditions and heaters the sharp thermal model reads, under one heading. An
 * optional one left empty stays "not set" and takes its derived default (the
 * inlet at saturation, no cavity heater, the heater on for the whole event).
 */
export const TABLE_CONDITIONS: ConditionSpec[] = fieldSpecs({
  section: "conditions",
  owner: "series",
  advanced: false,
})
  .filter((spec: FieldSpec) => !HAND_WRITTEN.has(spec.name))
  .map((spec) => fromTable(spec.name));

export const ALL_CONDITIONS: ConditionSpec[] = [
  ...REQUIRED_CONDITIONS,
  ...OPERATING_CONDITIONS,
  ...TABLE_CONDITIONS,
];

export type ConditionInputs = Record<ConditionKey, string>;

/** Every condition at its starting value, for a series not yet described. */
export function initialInputs(): ConditionInputs {
  return Object.fromEntries(
    ALL_CONDITIONS.map((spec) => [spec.key, spec.initial]),
  );
}

/** A series' saved conditions as form strings (unset reads as ""). U_ref is
 * reported under its unit, as `U_ref_m_s`. */
export function inputsFrom(conditions: OperatingConditions): ConditionInputs {
  const values = conditions as unknown as Record<string, unknown>;
  return Object.fromEntries(
    ALL_CONDITIONS.map((spec) => {
      const value =
        spec.key === "U_ref" ? conditions.U_ref_m_s : values[spec.key];
      return [spec.key, value == null ? "" : String(value)];
    }),
  );
}

/** Parse a condition value, or null when it is missing or out of range. */
export function parseCondition(
  raw: string,
  min: number,
  max: number,
): number | null {
  const value = Number(raw);
  if (raw.trim() === "" || !Number.isFinite(value)) return null;
  return value >= min && value <= max ? value : null;
}

/** The typed value of a form string: a number, a choice, or null for an
 * optional condition left empty ("not set"). Read it with `conditionValid`:
 * an invalid number is also null. */
export function conditionValue(
  spec: ConditionSpec,
  raw: string,
): number | string | null {
  if (spec.kind === "select") return raw;
  return parseCondition(raw, spec.min, spec.max);
}

/** Whether a form string is acceptable for its condition. */
export function conditionValid(spec: ConditionSpec, raw: string): boolean {
  if (spec.kind === "select") {
    return (spec.options ?? []).some((option) => option.value === raw);
  }
  if (raw.trim() === "") return spec.optional;
  return parseCondition(raw, spec.min, spec.max) !== null;
}

export interface ParsedCondition {
  spec: ConditionSpec;
  value: number | string | null;
  valid: boolean;
}

/** Every condition's typed value and validity, from the form strings. */
export function parseConditions(inputs: ConditionInputs): ParsedCondition[] {
  return ALL_CONDITIONS.map((spec) => ({
    spec,
    value: conditionValue(spec, inputs[spec.key]),
    valid: conditionValid(spec, inputs[spec.key]),
  }));
}

/**
 * The conditions edit to send: each entry's value under its key. With
 * `original`, only the entries whose value changed (so an edit to one
 * condition never re-sends, and re-bakes, the others); a cleared optional one
 * travels as null, which the API reads as "clear". Without it, every entry
 * with a value (a new series: an unset optional one is simply not sent).
 */
export function toConditionsUpdate(
  parsed: ParsedCondition[],
  original?: ParsedCondition[],
): ConditionsUpdate {
  const before = new Map(
    original?.map((entry) => [entry.spec.key, entry.value]),
  );
  const entries = parsed.filter(({ spec, value, valid }) =>
    original ? valid && value !== before.get(spec.key) : value !== null,
  );
  // The keys are the table's condition names, which the request type lists
  // field by field; this is the one place the generic row meets that type.
  return Object.fromEntries(
    entries.map(({ spec, value }) => [spec.key, value]),
  ) as ConditionsUpdate;
}

export type FluidsLoad =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; list: Fluid[] };

/** Load the fluid catalogue once; used by both the new-series and edit forms. */
export function useFluids(): FluidsLoad {
  const [load, setLoad] = useState<FluidsLoad>({ status: "loading" });
  useEffect(() => {
    let alive = true;
    api
      .listFluids()
      .then((list) => alive && setLoad({ status: "ready", list }))
      .catch(
        (err) =>
          alive && setLoad({ status: "error", message: errorMessage(err) }),
      );
    return () => {
      alive = false;
    };
  }, []);
  return load;
}

interface MeasurementFieldProps {
  spec: ConditionSpec;
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
  /** Note appended under the field (e.g. a re-preprocess warning for a baked
   * field on a processed series). */
  bakedNote?: string;
}

/** A single condition with its unit and bounds, in the form's field shell: a
 * number input, or a listbox for a condition that is a choice. */
export function MeasurementField({
  spec,
  value,
  disabled,
  onChange,
  bakedNote,
}: MeasurementFieldProps) {
  const id = useId();
  const out = value.trim() !== "" && !conditionValid(spec, value);
  return (
    <div className="fld">
      <label htmlFor={id}>
        {spec.label}
        {spec.hint && <span className="hint">{spec.hint}</span>}
      </label>
      {spec.kind === "select" ? (
        <Select
          id={id}
          value={value}
          options={spec.options ?? []}
          onChange={onChange}
          disabled={disabled}
          className="pick-field"
        />
      ) : (
        <span className="ug" data-disabled={disabled || undefined}>
          <input
            id={id}
            type="number"
            inputMode="decimal"
            value={value}
            step={spec.step}
            min={spec.min}
            max={spec.max}
            placeholder={
              spec.placeholder ?? (spec.optional ? "not set" : undefined)
            }
            disabled={disabled}
            aria-invalid={out || undefined}
            aria-describedby={`${id}-why`}
            onChange={(e) => onChange(e.target.value)}
          />
          <span className="sfx">{spec.unit}</span>
        </span>
      )}
      <span id={`${id}-why`} className="fld-why">
        {out
          ? `Must be between ${spec.min} and ${spec.max} ${spec.unit}.`
          : (bakedNote ?? spec.why)}
      </span>
    </div>
  );
}

/** The selected fluid's derived saturation temperature and saturated
 * properties, read-only: they come from the fluid, not the form. */
export function FluidProperties({ fluid }: { fluid: Fluid }) {
  const rows: { label: string; value: string }[] = [
    { label: "Saturation temperature", value: `${fluid.T_sat_C} °C` },
    { label: "Liquid density", value: `${fluid.rho_l.toFixed(0)} kg·m⁻³` },
    {
      label: "Liquid viscosity",
      value: `${(fluid.mu_l * 1e3).toPrecision(3)} mPa·s`,
    },
    {
      label: "Thermal conductivity",
      value: `${fluid.k_l.toPrecision(3)} W·m⁻¹·K⁻¹`,
    },
    {
      label: "Surface tension",
      value: `${(fluid.sigma * 1e3).toPrecision(3)} mN·m⁻¹`,
    },
    { label: "Latent heat", value: `${(fluid.h_lv / 1e3).toFixed(0)} kJ·kg⁻¹` },
  ];
  return (
    <dl className="fluid-props" aria-label={`${fluid.name} properties`}>
      {rows.map((row) => (
        <div className="fluid-prop" key={row.label}>
          <dt>{row.label}</dt>
          <dd>{row.value}</dd>
        </div>
      ))}
    </dl>
  );
}

interface FluidSelectProps {
  fluids: FluidsLoad;
  fluidId: string;
  disabled: boolean;
  onChange: (fluidId: string) => void;
}

/** The working-fluid selector plus the selected fluid's read-only properties. */
export function FluidSelect({
  fluids,
  fluidId,
  disabled,
  onChange,
}: FluidSelectProps) {
  const selectId = useId();
  const list = fluids.status === "ready" ? fluids.list : [];
  const selected = list.find((f) => f.id === fluidId) ?? list[0] ?? null;
  return (
    <>
      <div className="fld">
        <label htmlFor={selectId}>Working fluid</label>
        <Select
          id={selectId}
          value={fluidId}
          options={list.map((f) => ({ value: f.id, label: f.name }))}
          onChange={onChange}
          disabled={disabled || fluids.status !== "ready"}
          placeholder={
            fluids.status === "loading"
              ? "Loading fluids…"
              : "Fluids unavailable"
          }
          className="pick-field"
        />
        <span className="fld-why">
          sets the fluid properties and the derived saturation temperature
        </span>
      </div>
      {fluids.status === "error" && (
        <Callout tone="error">
          Could not load the fluid catalogue: {fluids.message}
        </Callout>
      )}
      {selected && <FluidProperties fluid={selected} />}
    </>
  );
}
