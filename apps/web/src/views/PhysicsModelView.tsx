import { useState } from "react";

import { StageAside } from "../app/StageAside";
import { Button, Callout } from "../components";
import type { DatasetSummary } from "../lib/api";
import { seriesName, seriesNameOf } from "../lib/series";
import { CapacityPreset } from "./physics/CapacityPreset";
import { EnsembleCanvas } from "./physics/EnsembleCanvas";
import { EquationsPanel } from "./physics/EquationsPanel";
import { ModelBudget } from "./physics/ModelBudget";
import { PerFieldTable } from "./physics/PerFieldTable";
import { HugeiconsIcon, SaveIcon } from "../components/icons";
import { RunBar } from "./physics/RunBar";
import { usePhysicsModel } from "./physics/usePhysicsModel";
import "./physics/physics.css";

/** The equation rows carry a toggle, a name, badges and a weight field, so this
 * stage's aside needs more room than the default. */
const ASIDE = {
  title: "Physics & model",
  subtitle: "equations · architecture",
  width: 461,
};

interface PhysicsModelViewProps {
  datasets: DatasetSummary[];
}

export function PhysicsModelView({ datasets }: PhysicsModelViewProps) {
  const [selected, setSelected] = useState<string | null>(null);
  const dataset = selected ?? datasets[0]?.id ?? null;
  const load = usePhysicsModel(dataset);

  if (datasets.length === 0) {
    return (
      <Callout tone="info">
        No datasets yet. Upload a sequence in Datasets to configure its physics
        and model.
      </Callout>
    );
  }

  const model = load.status === "ready" ? load.model : null;

  return (
    <div className="stack">
      <div className="pm-head">
        <p>
          Set the equations and capacity on the left; everything here is derived
          from them. Hover any equation for its math and detail, or any derived
          row for its reasoning.
        </p>
      </div>

      {load.status === "loading" && (
        <p className="state-note" role="status">
          Loading physics &amp; model…
        </p>
      )}
      {load.status === "error" && (
        <Callout tone="error">{load.message}</Callout>
      )}

      {model && (
        <>
          {model.saveError && <Callout tone="error">{model.saveError}</Callout>}
          <StageAside {...ASIDE}>
            {datasets.length > 1 && (
              <label className="pm-dataset">
                <span className="runlead">Dataset</span>
                <select
                  value={dataset ?? ""}
                  onChange={(e) => setSelected(e.target.value)}
                  aria-label="Dataset"
                >
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {seriesName(d)}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <EquationsPanel
              model={model}
              datasetName={seriesNameOf(datasets, model.dataset)}
            />
            <CapacityPreset model={model} />
            {/* Saving belongs with the configuration it commits, and stays
                reachable without scrolling back up the equation list. */}
            <div className="aside-actions pm-actions">
              <span
                className={model.dirty ? "pm-statechip dirty" : "pm-statechip"}
              >
                <span className="dot" aria-hidden="true" />
                {model.dirty ? "unsaved changes" : "saved"}
              </span>
              <Button
                variant="primary"
                onClick={() => model.save()}
                disabled={!model.dirty || model.saving}
              >
                <HugeiconsIcon icon={SaveIcon} size={15} />
                {model.saving ? "Saving…" : "Save"}
              </Button>
            </div>
          </StageAside>

          {/* Canvas: what the configuration above derives, in the order a
              researcher reads it — the cost, then the architecture it buys. */}
          <ModelBudget model={model} />
          <EnsembleCanvas model={model} />
          <PerFieldTable model={model} />
          <RunBar model={model} />
        </>
      )}
    </div>
  );
}
