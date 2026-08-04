import { PRESET_META } from "./model";
import type { PhysicsModel } from "./usePhysicsModel";

/**
 * The capacity preset, and whatever has been overridden away from it.
 *
 * The config half of what used to be the Model builder: this is what a
 * researcher SETS, so it lives in the stage's aside beside the equations. What
 * the preset then DERIVES -- the budget and the network shapes -- reads out on
 * the canvas, next to the ensemble those shapes describe.
 */
export function CapacityPreset({ model }: { model: PhysicsModel }) {
  return (
    <section className="pm-capacity" aria-labelledby="pm-capacity-h">
      <h3 id="pm-capacity-h">Capacity</h3>
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
      {model.overrideCount > 0 && (
        <div className="builderfoot">
          <span className="ovrchip">
            {model.overrideCount} value{model.overrideCount === 1 ? "" : "s"}{" "}
            overridden
            <button type="button" onClick={model.resetToPreset}>
              reset all
            </button>
          </span>
        </div>
      )}
    </section>
  );
}
