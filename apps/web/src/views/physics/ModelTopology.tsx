import { Panel, TopologyChart, ViewCanvas } from "../../components";
import type { ModelArchitecture } from "../../lib/api";

/** Live schematic of the field-ensemble network. */
export function ModelTopology({ model }: { model: ModelArchitecture }) {
  return (
    <Panel
      title="Model topology — live"
      subtitle="One Fourier-feature MLP per field, from the model config"
    >
      <ViewCanvas>
        <TopologyChart model={model} />
      </ViewCanvas>
    </Panel>
  );
}
