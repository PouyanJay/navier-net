import { NumberField, Panel } from "../../components";
import { ALL_FIELDS, type FieldName, FIELDS, fmtCount } from "./model";
import type { PhysicsModel } from "./usePhysicsModel";

function validate(model: PhysicsModel): string[] {
  const warns: string[] = [];
  const g = model.globals;
  if (g.alphaEps < 0.005) {
    warns.push(
      `Interface ε = ${g.alphaEps} is very small; the image supervision may not resolve it. Raise ε or bin the frames.`,
    );
  }
  if (g.ffScale < 2.5 && g.alphaEps <= 0.03) {
    warns.push(
      `σ_B = ${g.ffScale} is likely too low to resolve ε = ${g.alphaEps}; the α network will smooth the interface.`,
    );
  }
  if (model.activeFields.some((f) => model.perField[f].width > 256)) {
    warns.push("Width > 256 — CPU step cost grows quadratically; plan for a GPU device in Solver.");
  }
  return warns;
}

/** Shared globals plus the per-field width/depth/transform table and validation. */
export function PerFieldTable({ model }: { model: PhysicsModel }) {
  const g = model.globals;
  const warns = validate(model);

  return (
    <Panel
      title="Per-field architecture"
      subtitle="deltas from the shared defaults"
      className="advanced"
    >
      <div className="cfg">
        <NumberField
          label="Fourier features"
          suffix="pairs"
          value={g.ff}
          min={4}
          step={16}
          onChange={(v) => model.setGlobal("ff", v)}
        />
        <NumberField
          label="FF scale σ_B"
          value={g.ffScale}
          min={0.5}
          step={0.5}
          onChange={(v) => model.setGlobal("ffScale", v)}
        />
        <NumberField
          label="Interface ε"
          suffix="L*"
          value={g.alphaEps}
          min={0.001}
          step={0.01}
          onChange={(v) => model.setGlobal("alphaEps", v)}
        />
      </div>

      <div className="tblwrap">
        <table className="tblc">
          <thead>
            <tr>
              <th scope="col">Field</th>
              <th scope="col">Width</th>
              <th scope="col">Depth</th>
              <th scope="col">Output transform</th>
              <th scope="col">Params</th>
              <th scope="col">Stage</th>
            </tr>
          </thead>
          <tbody>
            {ALL_FIELDS.map((f) => {
              const meta = FIELDS[f];
              const on = model.fieldOn(f);
              const arch = model.perField[f];
              const unlock = meta.needs === "mom" ? "Momentum" : "Energy";
              return (
                <tr key={f} className={on ? "" : "dis"}>
                  <td>
                    <span className="fieldtag">
                      <span className="hue" style={{ background: meta.hue }} />
                      {meta.label}
                    </span>
                  </td>
                  <td>
                    <input
                      type="number"
                      min={8}
                      step={8}
                      value={arch.width}
                      disabled={!on}
                      aria-label={`${meta.label} width`}
                      onChange={(e) =>
                        model.setFieldArch(f, "width", Number(e.target.value) || arch.width)
                      }
                    />
                  </td>
                  <td>
                    <input
                      type="number"
                      min={1}
                      step={1}
                      value={arch.depth}
                      disabled={!on}
                      aria-label={`${meta.label} depth`}
                      onChange={(e) =>
                        model.setFieldArch(f, "depth", Number(e.target.value) || arch.depth)
                      }
                    />
                  </td>
                  <td>
                    {on ? meta.transform : <span className="locknote">enable {unlock} to unlock</span>}
                  </td>
                  <td>{on ? fmtCount(model.fieldParamCount(f)) : "—"}</td>
                  <td>
                    <span className={meta.stage === "A" ? "tag a" : "tag b"}>{meta.stage}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {warns.length > 0 ? (
        warns.map((w, i) => (
          <div className="valwarn" role="alert" key={i}>
            <span aria-hidden="true">⚠</span>
            <span>{w}</span>
          </div>
        ))
      ) : (
        <p className="valok">✓ configuration valid — no warnings</p>
      )}
    </Panel>
  );
}

export type { FieldName };
