import { useEffect, useId, useState } from "react";

import { Callout } from "../../components";
import { api, type Fluid } from "../../lib/api";
import { errorMessage } from "../../lib/errors";

export const DEFAULT_FLUID_ID = "fc72";

export type ConditionKey =
  | "dt_frame_ms"
  | "channel_width_um"
  | "channel_height_um"
  | "q_wall_W_cm2"
  | "flow_rate_mL_hr"
  | "U_ref";

/** Condition fields baked into the preprocessed tensors: editing one on a
 * processed series needs a re-preprocess. Mirrors BAKED_CONDITION_FIELDS in the
 * datasets service. Everything else applies live. */
export const BAKED_KEYS: ReadonlySet<ConditionKey> = new Set<ConditionKey>([
  "dt_frame_ms",
  "channel_width_um",
  "U_ref",
]);

export interface ConditionSpec {
  key: ConditionKey;
  label: string;
  hint: string;
  unit: string;
  step: number;
  /** Server-side bounds (`CONDITION_FIELDS` in the datasets service). */
  min: number;
  max: number;
  /** Shown when empty (measurements) or pre-filled as the value (operating). */
  placeholder?: string;
  why: string;
}

/**
 * The fixed geometry and timing of a run. Frame interval and channel width are
 * baked into the tensors (time axis and µm/px); channel height only feeds the
 * dimensionless groups. They start empty -- there is no honest default for a
 * series nobody has described yet.
 */
export const REQUIRED_CONDITIONS: ConditionSpec[] = [
  {
    key: "dt_frame_ms",
    label: "Frame interval",
    hint: "Δt between frames",
    unit: "ms",
    step: 0.1,
    placeholder: "0.5",
    min: 1e-6,
    max: 1e4,
    why: "sets the time axis of the tensors",
  },
  {
    key: "channel_width_um",
    label: "Channel width",
    hint: "imaged direction",
    unit: "µm",
    step: 10,
    placeholder: "300",
    min: 1,
    max: 1e6,
    why: "calibrates µm/px from the detected walls",
  },
  {
    key: "channel_height_um",
    label: "Channel height",
    hint: "heated wall below",
    unit: "µm",
    step: 10,
    placeholder: "150",
    min: 1,
    max: 1e6,
    why: "sets the physical scale of the heated channel",
  },
];

/**
 * The operating conditions of the run. They have sensible defaults, so they
 * start pre-filled and are edited to override; they set the reference scales and
 * the dimensionless groups. Saturation temperature is not here: it is a property
 * of the selected fluid (derived, read-only).
 */
export const OPERATING_CONDITIONS: ConditionSpec[] = [
  {
    key: "q_wall_W_cm2",
    label: "Wall heat flux",
    hint: "baseline",
    unit: "W·cm⁻²",
    step: 0.1,
    min: 1e-3,
    max: 1e4,
    why: "bottom-wall heat flux setpoint",
  },
  {
    key: "flow_rate_mL_hr",
    label: "Flow rate",
    hint: "inlet",
    unit: "mL·hr⁻¹",
    step: 0.5,
    min: 1e-3,
    max: 1e6,
    why: "volumetric inlet flow rate",
  },
  {
    key: "U_ref",
    label: "Reference velocity",
    hint: "U_ref",
    unit: "m·s⁻¹",
    step: 0.01,
    min: 1e-6,
    max: 1e3,
    why: "measured nose-speed scale; sets the reference time",
  },
];

export const ALL_CONDITIONS: ConditionSpec[] = [
  ...REQUIRED_CONDITIONS,
  ...OPERATING_CONDITIONS,
];

export type ConditionInputs = Record<ConditionKey, string>;

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

/** A single condition with its unit and bounds, in the form's field shell. */
export function MeasurementField({
  spec,
  value,
  disabled,
  onChange,
  bakedNote,
}: MeasurementFieldProps) {
  const id = useId();
  const out =
    value.trim() !== "" && parseCondition(value, spec.min, spec.max) === null;
  return (
    <div className="fld">
      <label htmlFor={id}>
        {spec.label}
        <span className="hint">{spec.hint}</span>
      </label>
      <span className="ug" data-disabled={disabled || undefined}>
        <input
          id={id}
          type="number"
          inputMode="decimal"
          value={value}
          step={spec.step}
          min={spec.min}
          max={spec.max}
          placeholder={spec.placeholder}
          disabled={disabled}
          aria-invalid={out || undefined}
          aria-describedby={`${id}-why`}
          onChange={(e) => onChange(e.target.value)}
        />
        <span className="sfx">{spec.unit}</span>
      </span>
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
        <div className="ug">
          <select
            id={selectId}
            value={fluidId}
            disabled={disabled || fluids.status !== "ready"}
            onChange={(e) => onChange(e.target.value)}
          >
            {fluids.status === "ready" ? (
              list.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.name}
                </option>
              ))
            ) : (
              <option>
                {fluids.status === "loading"
                  ? "Loading fluids…"
                  : "Fluids unavailable"}
              </option>
            )}
          </select>
        </div>
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
