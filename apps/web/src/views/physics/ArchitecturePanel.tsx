import { DL, type KV, Panel } from "../../components";
import type { ModelArchitecture } from "../../lib/api";

function rows(model: ModelArchitecture): KV[] {
  return [
    { label: "Fields", value: model.fields.join(", ") },
    {
      label: "Fourier features",
      value: model.fourier_feats,
      hint: `scale ${model.fourier_scale}`,
    },
    { label: "Hidden width", value: model.hidden },
    { label: "Hidden layers", value: model.layers },
    {
      label: "Activation",
      value: model.nodewise_activation
        ? "adaptive tanh (per-neuron)"
        : "adaptive tanh",
    },
    {
      label: "Interface half-width ε",
      value: model.alpha_eps,
      hint: "non-dim",
    },
  ];
}

/** Per-field architecture parameters, read live from the model config. */
export function ArchitecturePanel({ model }: { model: ModelArchitecture }) {
  return (
    <Panel
      title="Advanced; per-field architecture"
      subtitle="Fourier-feature MLP ensemble"
    >
      <DL items={rows(model)} />
    </Panel>
  );
}
