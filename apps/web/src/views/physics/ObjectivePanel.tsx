import { Panel } from "../../components";
import { FIELDS, type FieldName } from "./model";
import type { EquationDisplay, PhysicsModel } from "./usePhysicsModel";

/**
 * The colour a loss term takes: the field it principally trains.
 *
 * Reading the objective in the ensemble's own hues is what ties the two
 * panels together — a term and the network it pulls on are the same colour.
 */
const TERM_HUE: Record<string, FieldName> = {
  data: "phi",
  vof: "phi",
  div: "v",
  src: "s",
  bc: "u",
  mom: "p",
  darcy: "p",
  kinematic: "p",
  laplace: "p",
  energy: "T",
  evap: "T",
};

/** Short symbol for a term, as it is written in the loss. */
const TERM_SYMBOL: Record<string, string> = {
  data: "L_data",
  vof: "r_vof",
  div: "r_div",
  src: "r_src",
  bc: "L_bc",
  mom: "r_mom",
  darcy: "r_darcy",
  kinematic: "r_kin",
  laplace: "r_YL",
  energy: "r_energy",
  evap: "r_evap",
};

/**
 * The supervised data term. It is not a registry equation — nothing toggles it
 * and it has no residual — but it is the largest term in the objective, so an
 * objective that omitted it would be a lie of omission.
 */
const DATA_WEIGHT = 10;

function Term({ id, weight, on }: { id: string; weight: number; on: boolean }) {
  const hue = FIELDS[TERM_HUE[id] ?? "phi"].hue;
  return (
    <span
      className={on ? "obj-term" : "obj-term off"}
      style={{
        color: hue,
        background: `color-mix(in srgb, ${hue} 12%, transparent)`,
      }}
    >
      {weight !== 1 && <b>{weight}·</b>}
      {/* The symbol is its own node so the coefficient can change without
          re-flowing the name a reader is scanning for. */}
      <span className="obj-sym">{TERM_SYMBOL[id] ?? id}</span>
    </span>
  );
}

/**
 * The loss this configuration trains, derived live from the aside.
 *
 * The one line that proves the configuration and the run agree: every term is
 * an equation the page has admitted, every coefficient the weight it will
 * actually carry. Terms the formulation or the field set excludes are shown
 * ghosted rather than dropped — what is NOT being trained is as much a part of
 * the objective as what is.
 */
export function ObjectivePanel({ model }: { model: PhysicsModel }) {
  // Registry order, but supervision first: the objective reads the way it is
  // written in the paper, data and boundary conditions ahead of the residuals.
  const shown = model.equations.filter((e) => e.admitted || e.on);
  const lead: EquationDisplay[] = shown.filter((e) => e.id === "bc");
  const rest = shown.filter((e) => e.id !== "bc");
  const active = shown.filter((e) => e.on).length + 1; // + the data term

  return (
    <Panel
      title="The objective"
      subtitle={`${active} terms · weights as they will be carried`}
    >
      <p className="objective">
        <span className="obj-lhs">L</span> ={" "}
        <Term id="data" weight={DATA_WEIGHT} on />
        {[...lead, ...rest].map((eq) => (
          <span key={eq.id}>
            <span className={eq.on ? "obj-plus" : "obj-plus off"}>+</span>
            <Term id={eq.id} weight={eq.liveWeight} on={eq.on} />
          </span>
        ))}
      </p>
      <p className="obj-note">
        {/* Naming where each half of the weights lives is the whole point: a
            reader who edits a Stage-B weight here and a Stage-A weight at
            launch should never have to guess which one a run used. */}
        Stage-A weights are the launch defaults and final in the Solver; Stage-B
        weights are set beside their equations. Ghosted terms are not trained by
        this configuration.
      </p>
    </Panel>
  );
}

/**
 * Which interface treatment the composed run gets, and why.
 *
 * A readout, not a control: `model.sharp_interface` is resolved at run launch,
 * and the launcher appends the series' own overrides after its list, so a
 * per-series copy here would silently outrank an explicit Solver choice. The
 * page states what will happen and names where to change it.
 */
export function FormulationNote({ model }: { model: PhysicsModel }) {
  const f = model.formulation;
  return (
    <section
      className="pmform"
      role="status"
      aria-label="Interface formulation"
    >
      <span className={f.sharp ? "pmform-tag sharp" : "pmform-tag"}>
        {f.sharp ? "sharp front" : "diffuse"}
      </span>
      <p>
        {f.reason} <span className="dim">Chosen at run launch, in Solver.</span>
      </p>
    </section>
  );
}
