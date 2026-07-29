import { EquationBlock, Panel } from "../../components";
import { FIELDS, type FieldName } from "./model";
import type { EquationDisplay, PhysicsModel } from "./usePhysicsModel";

/** Plain-language notes shown in each equation's hover popover. */
const DESC: Record<string, string> = {
  vof: "Interface advected by the local velocity, supervised by the segmented frames.",
  div: "Continuity, with the phase-change dilatation entering as an inferred source.",
  src: "The inferred source is held to the interface, where phase change occurs.",
  bc: "Inlet plug velocity and no-slip side walls.",
  mom: "Depth-averaged (Hele-Shaw) momentum with wall drag and surface tension; unlocks pressure. Mixture ρ̃, μ̃ and Re, We come from the selected fluid.",
  energy:
    "Wall heating with an interfacial mass–energy closure; unlocks temperature. Pe and the saturation properties come from the selected fluid.",
  evap: "One evaporation flux closes the continuity source and removes latent heat.",
};

const fieldLabel = (f: string) => FIELDS[f as FieldName]?.label ?? f;
const fmtGroup = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3));

function EquationInfoPopover({
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

function EquationRow({
  eq,
  model,
  datasetName,
}: {
  eq: EquationDisplay;
  model: PhysicsModel;
  datasetName: string;
}) {
  return (
    <div className={eq.on ? "eqrow" : "eqrow off"}>
      <button
        type="button"
        role="switch"
        className="sw"
        aria-checked={eq.on}
        aria-label={eq.name}
        disabled={!eq.toggleable}
        title={
          eq.core
            ? "Core equation — always on"
            : !eq.toggleable
              ? "Unlocked with Energy + evaporation"
              : undefined
        }
        onClick={() => model.toggleEquation(eq.id)}
      >
        <span className="knob" aria-hidden="true" />
      </button>
      <div className="eqmain hasinfo" tabIndex={0}>
        <b>{eq.name}</b>
        <span className={eq.stage === "A" ? "tag a" : "tag b"}>{eq.stage}</span>
        {eq.core && <span className="tag core">core</span>}
        {eq.fields_added.length > 0 && (
          <span className="eqadds">+{eq.fields_added.map(fieldLabel).join(" +")}</span>
        )}
        <span className="wlab">
          <label htmlFor={`w-${eq.id}`}>w</label>
          <input
            id={`w-${eq.id}`}
            type="number"
            step="0.1"
            min="0"
            value={eq.liveWeight}
            disabled={!eq.on || eq.core}
            title={eq.core ? "Stage-A weights are set in the Solver run configuration" : undefined}
            onChange={(e) => model.setWeight(eq.weight_key, Number(e.target.value) || 0)}
          />
        </span>
        <span className="infob" aria-hidden="true">
          i
        </span>
        <EquationInfoPopover
          eq={eq}
          datasetName={datasetName}
          groups={model.groups}
        />
      </div>
    </div>
  );
}

/** The governing equations: compact toggle rows, math + detail in a popover.
 * ``datasetName`` is the selected series' display label (the popover shows it
 * next to the groups); ``model.dataset`` stays the id for the API. */
export function EquationsPanel({
  model,
  datasetName,
}: {
  model: PhysicsModel;
  datasetName: string;
}) {
  const active = model.equations.filter((e) => e.on).length;
  return (
    <Panel
      title="Governing equations"
      subtitle="nondimensional · properties from the selected fluid"
    >
      <div role="group" aria-label="Governing equations">
        {model.equations.map((eq) => (
          <EquationRow key={eq.id} eq={eq} model={model} datasetName={datasetName} />
        ))}
      </div>
      <p className="reco-note" style={{ marginTop: "12px" }}>
        <b>{active}</b> of {model.equations.length} equations active. Core equations are locked
        on; enabling Momentum or Energy unlocks pressure or temperature.
      </p>
    </Panel>
  );
}
