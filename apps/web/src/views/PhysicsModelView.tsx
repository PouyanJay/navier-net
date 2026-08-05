import { useState } from "react";

import { StageAside } from "../app/StageAside";
import { Button, Callout, Select } from "../components";
import { HugeiconsIcon, SaveIcon } from "../components/icons";
import type { DatasetSummary } from "../lib/api";
import { seriesName, seriesNameOf } from "../lib/series";
import { EnsembleCanvas } from "./physics/EnsembleCanvas";
import { ModelBudget } from "./physics/ModelBudget";
import { FormulationNote, ObjectivePanel } from "./physics/ObjectivePanel";
import { PhysicsAside } from "./physics/PhysicsAside";
import { RunBar } from "./physics/RunBar";
import { usePhysicsModel } from "./physics/usePhysicsModel";
import "./physics/physics.css";

/** The equation rows carry a toggle, a name, a badge and a weight field, so
 * this stage's aside needs more room than the default. */
const ASIDE = {
  title: "Physics & model",
  subtitle: "equations · architecture",
  width: 461,
};

interface PhysicsModelViewProps {
  datasets: DatasetSummary[];
}

/**
 * The physics stage: compose the objective on the left, read what it costs and
 * trains on the right.
 *
 * A PINN's physics configuration IS its loss function — every equation a term,
 * every term a weight, every field a network that has to carry it. So the
 * aside holds what is SET (the physics, the capacity) and the canvas holds
 * everything DERIVED from it, in the order a researcher checks it: the price,
 * the objective, the ensemble that pays it, the command that runs it.
 */
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
              <div className="pm-dataset">
                <Select
                  label="Series"
                  value={dataset ?? ""}
                  options={datasets.map((d) => ({
                    value: d.id,
                    label: seriesName(d),
                  }))}
                  onChange={setSelected}
                />
              </div>
            )}

            <PhysicsAside
              model={model}
              datasetName={seriesNameOf(datasets, model.dataset)}
            />

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

          {/* The canvas, in the order a researcher reads it: what it costs,
              the objective that cost buys, the ensemble that carries it, and
              the command that reproduces the lot. */}
          <ModelBudget model={model} />
          <FormulationNote model={model} />
          <ObjectivePanel model={model} />
          <EnsembleCanvas model={model} />
          <RunBar model={model} />
        </>
      )}
    </div>
  );
}
