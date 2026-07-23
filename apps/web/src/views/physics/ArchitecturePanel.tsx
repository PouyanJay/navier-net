import { DL, type KV, Panel } from "../../components";
import type { ModelArchitecture } from "../../lib/api";

function rows(net: ModelArchitecture): KV[] {
  return [
    { label: "Fields", value: net.fields.join(", ") },
    { label: "Fourier features", value: net.fourier_feats, hint: `scale ${net.fourier_scale}` },
    { label: "Hidden width", value: net.hidden },
    { label: "Hidden layers", value: net.layers },
    { label: "Activation", value: net.nodewise_activation ? "adaptive tanh (per-neuron)" : "adaptive tanh" },
    { label: "Interface half-width ε", value: net.alpha_eps },
  ];
}

/** Per-field architecture parameters, read live from the model config. */
export function ArchitecturePanel({ model }: { model: ModelArchitecture }) {
  return (
    <Panel title="Advanced — per-field architecture" subtitle="Fourier-feature MLP ensemble">
      <DL items={rows(model)} />
    </Panel>
  );
}
