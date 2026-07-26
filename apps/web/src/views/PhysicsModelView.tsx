import { Callout } from "../components";
import { ArchitecturePanel } from "./physics/ArchitecturePanel";
import { GoverningEquations } from "./physics/GoverningEquations";
import { ModelTopology } from "./physics/ModelTopology";
import { useModel } from "./physics/useModel";
import "./runs.css";

export function PhysicsModelView() {
  const load = useModel();

  return (
    <div className="stack">
      <GoverningEquations />

      {load.status === "loading" && (
        <p className="state-note" role="status">
          Loading model…
        </p>
      )}
      {load.status === "empty" && (
        <p className="state-note">
          No datasets yet; upload a sequence to see the model topology.
        </p>
      )}
      {load.status === "error" && (
        <Callout tone="error">{load.message}</Callout>
      )}
      {load.status === "ready" && (
        <>
          <ModelTopology model={load.model} />
          <ArchitecturePanel model={load.model} />
        </>
      )}
    </div>
  );
}
