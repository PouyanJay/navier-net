import { ArchitecturePanel } from "./physics/ArchitecturePanel";
import { GoverningEquations } from "./physics/GoverningEquations";
import { ModelTopology } from "./physics/ModelTopology";
import { useModel } from "./physics/useModel";
import "./runs.css";

export function PhysicsModelView() {
  const { model, error } = useModel();

  return (
    <div className="stack">
      <GoverningEquations />

      {error && (
        <p className="state-note error" role="alert">{error}</p>
      )}

      {model && (
        <>
          <ModelTopology model={model} />
          <ArchitecturePanel model={model} />
        </>
      )}
    </div>
  );
}
