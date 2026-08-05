import { NumberField, Panel } from "../../components";
import { ALL_FIELDS, FIELDS, fmtCount } from "./model";
import type { PhysicsModel } from "./usePhysicsModel";

// Validation thresholds, named rather than inlined in the checks below.
const MIN_ALPHA_EPS = 0.005; // interface half-width the image supervision can resolve
const SMALL_EPS = 0.03; // below this, σ_B must be large enough to resolve the interface
const MIN_FF_SCALE_FOR_SMALL_EPS = 2.5;
const MAX_STABLE_WIDTH = 256; // beyond this, CPU step cost grows quadratically

function validate(model: PhysicsModel): string[] {
  const warns: string[] = [];
  const g = model.globals;
  if (g.alphaEps < MIN_ALPHA_EPS) {
    warns.push(
      `Interface ε = ${g.alphaEps} is very small; the image supervision may not resolve it. Raise ε or bin the frames.`,
    );
  }
  if (g.ffScale < MIN_FF_SCALE_FOR_SMALL_EPS && g.alphaEps <= SMALL_EPS) {
    warns.push(
      `σ_B = ${g.ffScale} is likely too low to resolve ε = ${g.alphaEps}; the α network will smooth the interface.`,
    );
  }
  if (
    model.activeFields.some((f) => model.perField[f].width > MAX_STABLE_WIDTH)
  ) {
    warns.push(
      `Width > ${MAX_STABLE_WIDTH}: CPU step cost grows quadratically; plan for a GPU device in Solver.`,
    );
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
                        model.setFieldArch(
                          f,
                          "width",
                          Number(e.target.value) || arch.width,
                        )
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
                        model.setFieldArch(
                          f,
                          "depth",
                          Number(e.target.value) || arch.depth,
                        )
                      }
                    />
                  </td>
                  <td>
                    {on ? (
                      meta.transform
                    ) : (
                      <span className="locknote">
                        enable {unlock} to unlock
                      </span>
                    )}
                  </td>
                  <td>{on ? fmtCount(model.fieldParamCount(f)) : "—"}</td>
                  <td>
                    <span className={meta.stage === "A" ? "tag a" : "tag b"}>
                      {meta.stage}
                    </span>
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
        <p className="valok">✓ configuration valid · no warnings</p>
      )}
    </Panel>
  );
}
