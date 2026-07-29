import { useState } from "react";

import { Callout } from "../components";
import type { DatasetSummary } from "../lib/api";
import { seriesName } from "../lib/series";
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

/** Floppy-disk save glyph, drawn in the house SVG style (16-grid, 1.4 stroke). */
function SaveIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      aria-hidden="true"
      focusable="false"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M4 2.7h5.3L13.3 6.7v5.3a1.3 1.3 0 0 1-1.3 1.3H4a1.3 1.3 0 0 1-1.3-1.3V4A1.3 1.3 0 0 1 4 2.7Z" />
      <path d="M5.6 2.7v3h3.2v-3" />
      <path d="M5 13.3V9.6h6v3.7" />
    </svg>
  );
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
                    {seriesName(d)}
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
          <button
            type="button"
            className="btn ghost icon-only pm-save"
            data-tip={model?.saving ? "Saving…" : "Save configuration"}
            aria-label={model?.saving ? "Saving configuration" : "Save configuration"}
            disabled={!model || !model.dirty || model.saving}
            onClick={() => model?.save()}
          >
            <SaveIcon />
          </button>
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
