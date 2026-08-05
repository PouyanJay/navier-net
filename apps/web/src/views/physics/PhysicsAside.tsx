import { EquationBlock } from "../../components";
import {
  CapacityIcon,
  HugeiconsIcon,
  LockedIcon,
  OptionalPhysicsIcon,
  WarningIcon,
} from "../../components/icons";
import { ALL_FIELDS, FIELDS, type FieldName, PRESET_META } from "./model";
import type { EquationDisplay, PhysicsModel, Warning } from "./usePhysicsModel";

/** The equation that unlocks a Stage-B field, for the locked rows' reason. */
const UNLOCKED_BY: Partial<Record<FieldName, string>> = {
  p: "Momentum",
  T: "Energy",
};

/** Plain-language notes shown in each equation's hover popover. */
const DESC: Record<string, string> = {
  vof: "Interface advected by the local velocity, supervised by the segmented frames.",
  div: "Continuity, with the phase-change dilatation entering as an inferred source.",
  src: "The inferred source is held to the interface, where phase change occurs.",
  bc: "Inlet plug velocity and no-slip side walls.",
  mom: "Depth-averaged (Hele-Shaw) momentum with wall drag and surface tension; unlocks pressure. Mixture ρ̃, μ̃ and Re, We come from the selected fluid.",
  darcy:
    "Depth-averaged momentum reduced to Darcy drag, imposed on the explicit front.",
  kinematic:
    "The front moves with the normal component of the velocity it is immersed in.",
  laplace:
    "The pressure jump across the front is its curvature over the Weber number.",
  energy:
    "Wall heating with an interfacial mass–energy closure; unlocks temperature. Pe and the saturation properties come from the selected fluid.",
  evap: "One evaporation flux closes the continuity source and removes latent heat.",
};

const fieldLabel = (f: string) => FIELDS[f as FieldName]?.label ?? f;
const fmtGroup = (v: number) =>
  Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3);

function EquationInfo({
  eq,
  datasetName,
  groups,
}: {
  eq: EquationDisplay;
  datasetName: string;
  groups: Record<string, number>;
}) {
  const shown = eq.groups.filter((g) => groups[g] !== undefined);
  return (
    <div className="infopop" role="tooltip">
      <div className="eqmath">
        <EquationBlock tex={eq.tex} />
      </div>
      {DESC[eq.id] && <p className="eqdesc">{DESC[eq.id]}</p>}
      <div className="eqmeta">
        {eq.fields_required.map((f) => (
          <span className="fchip" key={f}>
            {fieldLabel(f)}
          </span>
        ))}
        {eq.fields_added.map((f) => (
          <span className="fchip add" key={f}>
            + field {fieldLabel(f)}
          </span>
        ))}
      </div>
      {shown.length > 0 && (
        <div className="groupsline">
          <span>{datasetName}:</span>
          {shown.map((g) => (
            <span key={g}>
              <b>{g}</b> = {fmtGroup(groups[g])}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * A core equation: a fact, not a control.
 *
 * These always train, and their weights are owned by the run-launch form, so
 * the row carried a switch that could never move and a field that could never
 * be typed in. It states the physics instead, and the math is one hover away.
 */
function CoreRow({
  eq,
  datasetName,
  groups,
}: {
  eq: EquationDisplay;
  datasetName: string;
  groups: Record<string, number>;
}) {
  return (
    <div className="pmrow core hasinfo" tabIndex={0}>
      <span className="pmrow-nm">
        <b>{eq.name}</b>
      </span>
      <span className="pmrow-w mono">w {eq.liveWeight}</span>
      <span className="infob" aria-hidden="true">
        i
      </span>
      <EquationInfo eq={eq} datasetName={datasetName} groups={groups} />
    </div>
  );
}

/** A Stage-B equation: a toggle that states its price, and its weight. */
function OptionalRow({
  eq,
  model,
  datasetName,
}: {
  eq: EquationDisplay;
  model: PhysicsModel;
  datasetName: string;
}) {
  const locked = !eq.toggleable;
  // Toggleable but not admitted: this equation adds the field, and a different
  // set of equations carries it under the current formulation.
  const replaced = eq.toggleable && !eq.admitted;
  return (
    <div className={eq.on || replaced ? "pmrow" : "pmrow off"}>
      <button
        type="button"
        role="switch"
        className="sw"
        aria-checked={replaced ? fieldsOn(model, eq) : eq.on}
        aria-label={eq.name}
        disabled={locked}
        title={
          locked ? "Follows the equation that unlocks its field" : undefined
        }
        onClick={() => model.toggleEquation(eq.id)}
      >
        <span className="knob" aria-hidden="true" />
      </button>
      <span className="pmrow-nm hasinfo" tabIndex={0}>
        <b>{eq.name}</b>
        {/* The price of the physics, in the currency that matters: the field
            it adds, and therefore a whole network to train. */}
        {eq.fields_added.map((f) => (
          <span className={`unlock ${f}`} key={f}>
            +{fieldLabel(f)}
          </span>
        ))}
        {replaced && (
          <span className="replaced">carried by the front conditions</span>
        )}
        <EquationInfo
          eq={eq}
          datasetName={datasetName}
          groups={groups(model)}
        />
      </span>
      <span className="pmrow-w">
        <label className="sr-only" htmlFor={`w-${eq.id}`}>
          {eq.name} loss weight
        </label>
        <input
          id={`w-${eq.id}`}
          type="number"
          step="0.1"
          min="0"
          value={eq.liveWeight}
          disabled={!eq.on}
          onChange={(e) =>
            model.setWeight(eq.weight_key, Number(e.target.value) || 0)
          }
        />
      </span>
    </div>
  );
}

const groups = (model: PhysicsModel) => model.groups;

/** Whether the field a switch-row adds is currently trained. */
const fieldsOn = (model: PhysicsModel, eq: EquationDisplay) =>
  eq.fields_added.every((f) => model.fieldOn(f as FieldName));

/** A caution rendered at the control that caused it. */
function Caution({ warning }: { warning: Warning }) {
  return (
    <p className="pmwarn" role="alert">
      <HugeiconsIcon icon={WarningIcon} size={13} aria-hidden="true" />
      <span>{warning.message}</span>
    </p>
  );
}

/**
 * Everything this stage SETS, as labelled bands: the physics that defines the
 * objective, and the capacity that has to carry it.
 *
 * The derived half — what it costs, the loss it builds, the ensemble it
 * describes — reads out on the canvas. Interface formulation is absent on
 * purpose: it is resolved at run launch (see `Formulation`), so this page
 * reports it rather than owning a second copy that would outrank the Solver.
 */
export function PhysicsAside({
  model,
  datasetName,
}: {
  model: PhysicsModel;
  datasetName: string;
}) {
  const core = model.equations.filter((e) => e.core);
  // A toggleable equation is the switch for the FIELD it adds, so it stays on
  // screen whichever formulation is active — hiding Momentum under the sharp
  // treatment would remove the only way to train pressure at all. What changes
  // is which equations then carry that field, which the row says for itself.
  const optional = model.equations.filter(
    (e) => !e.core && (e.toggleable || e.admitted),
  );
  const at = (key: Warning["at"]) => model.warnings.filter((w) => w.at === key);

  return (
    <>
      <section className="pmband">
        <p className="pmband-hd">
          <HugeiconsIcon icon={LockedIcon} size={13} aria-hidden="true" />
          Core physics
          <span className="hint">always trains · weights set at launch</span>
        </p>
        {core.map((eq) => (
          <CoreRow
            key={eq.id}
            eq={eq}
            datasetName={datasetName}
            groups={model.groups}
          />
        ))}
      </section>

      <section className="pmband">
        <p className="pmband-hd">
          <HugeiconsIcon
            icon={OptionalPhysicsIcon}
            size={13}
            aria-hidden="true"
          />
          Optional physics
          <span className="hint">
            {model.formulation.sharp ? "sharp front" : "diffuse"}
          </span>
        </p>
        {optional.map((eq) => (
          <OptionalRow
            key={eq.id}
            eq={eq}
            model={model}
            datasetName={datasetName}
          />
        ))}
      </section>

      <section className="pmband">
        <p className="pmband-hd">
          <HugeiconsIcon icon={CapacityIcon} size={13} aria-hidden="true" />
          Capacity
          {model.overrideCount > 0 && (
            <span className="hint over">
              {model.overrideCount} overridden
              <button type="button" onClick={model.resetToPreset}>
                reset
              </button>
            </span>
          )}
        </p>

        <div className="seg" role="radiogroup" aria-label="Capacity preset">
          {PRESET_META.map((preset) => {
            const on = model.preset === preset.name;
            return (
              <button
                key={preset.name}
                type="button"
                role="radio"
                aria-checked={on}
                className={on ? "segb on" : "segb"}
                onClick={() => model.applyPreset(preset.name)}
              >
                {preset.label}
                <span>{preset.note}</span>
              </button>
            );
          })}
        </div>

        <dl className="pmfields">
          <Global
            label="Fourier features"
            unit="pairs"
            range="4 – 512"
            value={model.globals.ff}
            step={16}
            min={4}
            onChange={(v) => model.setGlobal("ff", v)}
          />
          <Global
            label="Fourier scale σ_B"
            range="0.5 – 10"
            value={model.globals.ffScale}
            step={0.5}
            min={0.5}
            warn={at("ffScale").length > 0}
            onChange={(v) => model.setGlobal("ffScale", v)}
          />
          {at("ffScale").map((w, i) => (
            <Caution key={i} warning={w} />
          ))}
          <Global
            label="Interface ε"
            unit="L*"
            range="≥ 0.005 resolvable"
            value={model.globals.alphaEps}
            step={0.01}
            min={0.001}
            warn={at("alphaEps").length > 0}
            onChange={(v) => model.setGlobal("alphaEps", v)}
          />
          {at("alphaEps").map((w, i) => (
            <Caution key={i} warning={w} />
          ))}
        </dl>

        {/* Per-field width and depth: the shapes the preset derives, editable
            where they are set. What they COST reads out on the canvas, in the
            ensemble that draws them. */}
        <p className="pmsub">Per field · width × depth</p>
        <dl className="pmfields">
          {ALL_FIELDS.map((f) => {
            const on = model.fieldOn(f);
            const arch = model.perField[f];
            return (
              <div
                key={f}
                className={
                  "pmarch" +
                  (on ? "" : " off") +
                  (model.fieldOverridden(f) ? " over" : "")
                }
              >
                <dt>
                  <span
                    className="hue"
                    style={{ background: FIELDS[f].hue }}
                    aria-hidden="true"
                  />
                  {FIELDS[f].label}
                  {!on && (
                    <span className="locknote">
                      enable {UNLOCKED_BY[f]} to unlock
                    </span>
                  )}
                </dt>
                <dd>
                  <input
                    type="number"
                    min={8}
                    step={8}
                    value={arch.width}
                    disabled={!on}
                    aria-label={`${FIELDS[f].label} width`}
                    onChange={(e) =>
                      model.setFieldArch(
                        f,
                        "width",
                        Number(e.target.value) || arch.width,
                      )
                    }
                  />
                  <span className="times" aria-hidden="true">
                    ×
                  </span>
                  <input
                    type="number"
                    min={1}
                    step={1}
                    value={arch.depth}
                    disabled={!on}
                    aria-label={`${FIELDS[f].label} depth`}
                    onChange={(e) =>
                      model.setFieldArch(
                        f,
                        "depth",
                        Number(e.target.value) || arch.depth,
                      )
                    }
                  />
                </dd>
              </div>
            );
          })}
        </dl>
        {at("width").map((w, i) => (
          <Caution key={i} warning={w} />
        ))}
      </section>
    </>
  );
}

/** One global: label, unit and valid range on the label, value on the right. */
function Global({
  label,
  unit,
  range,
  value,
  step,
  min,
  warn,
  onChange,
}: {
  label: string;
  unit?: string;
  range: string;
  value: number;
  step: number;
  min: number;
  warn?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <div className={warn ? "pmglobal warn" : "pmglobal"}>
      <dt>
        <label htmlFor={`g-${label}`}>
          {label}
          {unit && <small> ({unit})</small>}
        </label>
        <span className="range mono">{range}</span>
      </dt>
      <dd>
        <input
          id={`g-${label}`}
          type="number"
          step={step}
          min={min}
          value={value}
          onChange={(e) => onChange(Number(e.target.value) || value)}
        />
      </dd>
    </div>
  );
}
