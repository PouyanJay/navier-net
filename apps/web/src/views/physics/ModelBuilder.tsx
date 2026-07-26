import { Panel } from "../../components";
import {
  derivePerField,
  estMemoryMB,
  estMinutesPer1kSteps,
  fmtCount,
  PRESETS,
  type PresetName,
} from "./model";
import type { PhysicsModel } from "./usePhysicsModel";

const PRESET_META: { name: PresetName; label: string; note: string }[] = [
  { name: "small", label: "Small", note: "fast sanity" },
  { name: "medium", label: "Medium", note: "balanced" },
  { name: "large", label: "Large", note: "paper-class" },
];

interface Reco {
  value: string;
  why: string;
  modified: boolean;
}

function recommendations(model: PhysicsModel): Reco[] {
  const preset = PRESETS[model.preset];
  const derived = derivePerField(preset);
  const g = model.globals;
  const pf = model.perField;
  const pOn = model.fieldOn("p") || model.fieldOn("T");
  return [
    {
      value: `σ_B = ${g.ffScale} · ε = ${g.alphaEps}`,
      why: `Spectral scale sized so ${g.ff} Fourier pairs resolve the interface half-width.`,
      modified: g.ff !== preset.ff || g.ffScale !== preset.ffScale,
    },
    {
      value: `α network ${pf.phi.width} × ${pf.phi.depth}`,
      why: "The steepest field — it carries the sigmoid(φ/ε) interface.",
      modified: pf.phi.width !== derived.phi.width || pf.phi.depth !== derived.phi.depth,
    },
    {
      value: `u, v, s networks ${pf.u.width} × ${pf.u.depth}`,
      why: "Smooth hidden fields; capacity follows α at ratio 1.0.",
      modified: pf.u.width !== derived.u.width || pf.u.depth !== derived.u.depth,
    },
    {
      value: pOn ? `p, T networks ${pf.p.width} × ${pf.p.depth}` : "p, T — locked",
      why: pOn
        ? "Stage-B fields sized +33% width, +2 depth for stiffer residuals."
        : "Enable Momentum or Energy to unlock pressure and temperature.",
      modified: pOn && (pf.p.width !== derived.p.width || pf.p.depth !== derived.p.depth),
    },
  ];
}

/** The model builder: capacity preset → live budget → derived recommendations. */
export function ModelBuilder({ model }: { model: PhysicsModel }) {
  const params = model.totalParams;
  const minutes = estMinutesPer1kSteps(params);
  const memory = estMemoryMB(params);
  const recos = recommendations(model);

  return (
    <Panel title="Model builder" subtitle="preset → derived → override" className="builder">
      <div className="seg" role="radiogroup" aria-label="Capacity preset">
        {PRESET_META.map((p) => {
          const on = model.preset === p.name;
          return (
            <button
              key={p.name}
              type="button"
              role="radio"
              aria-checked={on}
              className={on ? "segb on" : "segb"}
              onClick={() => model.applyPreset(p.name)}
            >
              {p.label}
              <span>{p.note}</span>
            </button>
          );
        })}
      </div>

      <div className="budget" role="status" aria-label="Model budget">
        <div className="bud">
          <div className="k">parameters</div>
          <div className="v">{fmtCount(params)}</div>
        </div>
        <div className="bud">
          <div className="k">est · 1k steps</div>
          <div className="v">
            {minutes.toFixed(1)} <small>min CPU</small>
          </div>
        </div>
        <div className="bud">
          <div className="k">est · memory</div>
          <div className="v">
            {memory.toFixed(0)} <small>MB</small>
          </div>
        </div>
      </div>

      <p className="reco-note">
        <b>Derived from your physics.</b> Parameter count is exact; time and memory are
        estimates. Hover a row for the reasoning; edit anything in the table below to override.
      </p>

      <div>
        {recos.map((r, i) => (
          <div key={i} className={r.modified ? "rrow mod" : "rrow"}>
            <span className="rv">{r.value}</span>
            <span className="hasinfo" tabIndex={0}>
              <span className="infob" aria-hidden="true">
                i
              </span>
              <div className="infopop rwhy" role="tooltip">
                {r.why}
              </div>
            </span>
          </div>
        ))}
      </div>

      {model.overrideCount > 0 && (
        <div className="builderfoot">
          <span className="ovrchip">
            {model.overrideCount} value{model.overrideCount === 1 ? "" : "s"} overridden
            <button type="button" onClick={model.resetToPreset}>
              reset all
            </button>
          </span>
        </div>
      )}
    </Panel>
  );
}
