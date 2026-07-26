import { useState } from "react";

import { Button, Callout } from "../components";
import type { DatasetSummary } from "../lib/api";
import { EnsembleCanvas } from "./physics/EnsembleCanvas";
import { EquationsPanel } from "./physics/EquationsPanel";
import { ModelBuilder } from "./physics/ModelBuilder";
import { PerFieldTable } from "./physics/PerFieldTable";
import { RunBar } from "./physics/RunBar";
import { usePhysicsModel } from "./physics/usePhysicsModel";
import "./physics/physics.css";

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
        No datasets yet — upload a sequence in Datasets to configure its physics and model.
      </Callout>
    );
  }

  const model = load.status === "ready" ? load.model : null;

  return (
    <div className="stack">
      <div className="pm-head">
        <div>
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
                    {d.id}
                  </option>
                ))}
              </select>
            </label>
          )}
          <p>
            Toggle the governing equations; the builder derives the network ensemble. Hover any
            equation for its math and detail.
          </p>
        </div>
        <div className="pm-headacts">
          {model && (
            <span className={model.dirty ? "pm-statechip dirty" : "pm-statechip"}>
              <span className="dot" aria-hidden="true" />
              {model.dirty ? "unsaved changes" : "saved"}
            </span>
          )}
          <Button
            variant="primary"
            disabled={!model || !model.dirty || model.saving}
            onClick={() => model?.save()}
          >
            {model?.saving ? "Saving…" : "Save configuration"}
          </Button>
        </div>
      </div>

      {load.status === "loading" && (
        <p className="state-note" role="status">
          Loading physics &amp; model…
        </p>
      )}
      {load.status === "error" && <Callout tone="error">{load.message}</Callout>}

      {model && (
        <>
          {model.saveError && <Callout tone="error">{model.saveError}</Callout>}
          <div className="pm-grid">
            <EquationsPanel model={model} />
            <ModelBuilder model={model} />
          </div>
          <EnsembleCanvas model={model} />
          <PerFieldTable model={model} />
          <RunBar model={model} />
        </>
      )}
    </div>
  );
}
