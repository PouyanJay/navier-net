import { estMemoryMB, estMinutesPer1kSteps, FIELDS, fmtCount } from "./model";
import type { PhysicsModel } from "./usePhysicsModel";

/**
 * What the configured physics costs: the stage's headline numbers.
 *
 * Derived, so it reads out on the canvas rather than sitting in the aside with
 * the knobs that produce it — and it leads the canvas because these three are
 * what a researcher checks before committing to a run.
 */
export function ModelBudget({ model }: { model: PhysicsModel }) {
  const params = model.totalParams;
  return (
    <div className="budget" role="status" aria-label="Model budget">
      <div className="bud">
        <div className="k">parameters</div>
        <div className="v">{fmtCount(params)}</div>
      </div>
      <div className="bud">
        <div className="k">est · 1k steps</div>
        <div className="v">
          {estMinutesPer1kSteps(params).toFixed(1)} <small>min CPU</small>
        </div>
      </div>
      <div className="bud">
        <div className="k">est · memory</div>
        <div className="v">
          {estMemoryMB(params).toFixed(0)} <small>MB</small>
        </div>
      </div>
      {/* How many networks the physics bought — the count the ensemble draws
          and the reason the other three numbers are what they are. */}
      <div className="bud">
        <div className="k">networks</div>
        <div className="v">
          {model.activeFields.length}{" "}
          <small>
            {model.activeFields.map((f) => FIELDS[f].label).join(" ")}
          </small>
        </div>
      </div>
    </div>
  );
}
