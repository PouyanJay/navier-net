import { useState } from "react";

import { Panel } from "../../components";
import type { DimensionlessGroups } from "../../lib/api";

interface Tile {
  key: string;
  label: string;
  digits: number;
  unit?: string;
  /** Transform from the API's value to the displayed one (e.g. m/s → mm/s). */
  scale?: number;
  /** Full name and what the group is (its ratio), for the definition panel. */
  name: string;
  about: string;
  /** What the current value means physically — the regime it puts the run in. */
  reading: (value: number) => string;
}

// Order and vocabulary follow the mockup; every value is the pipeline's own,
// and the definitions match naviernet.physics.groups.
const TILES: Tile[] = [
  {
    key: "Re",
    label: "RE",
    digits: 1,
    name: "Reynolds number",
    about: "Inertial forces ÷ viscous forces, ρ·U_ref·L_ref / µ.",
    reading: (v) =>
      v < 2300
        ? "Laminar: viscosity keeps the flow smooth and orderly."
        : "Turbulent: inertia overwhelms viscous damping.",
  },
  {
    key: "We",
    label: "WE",
    digits: 2,
    name: "Weber number",
    about: "Inertia ÷ surface tension, ρ·U_ref²·L_ref / σ.",
    reading: (v) =>
      v < 10
        ? "Surface tension dominates, so the vapour caps stay rounded."
        : "Inertia deforms the interface out of round.",
  },
  {
    key: "Ca",
    label: "CA",
    digits: 4,
    name: "Capillary number",
    about: "Viscous drag ÷ surface tension, µ·U_ref / σ.",
    reading: (v) =>
      v < 1
        ? "Surface tension wins: the bubble leaves a thin Bretherton film (δ ∝ Ca^⅔)."
        : "Viscous drag stretches the interface into long thin films.",
  },
  {
    key: "Pr",
    label: "PR",
    digits: 2,
    name: "Prandtl number",
    about: "Momentum diffusivity ÷ thermal diffusivity, cp·µ / k.",
    reading: (v) =>
      `Momentum spreads ~${v.toFixed(0)}× faster than heat, so the thermal boundary layer is thin.`,
  },
  {
    key: "Pe",
    label: "PE",
    digits: 0,
    name: "Péclet number",
    about: "Advection ÷ thermal diffusion, Re · Pr.",
    reading: (v) =>
      `Advection outpaces conduction ~${v.toFixed(0)}×: heat is carried by the flow, not diffused.`,
  },
  {
    key: "Ja_per_5K",
    label: "JA",
    digits: 3,
    unit: "@5K",
    name: "Jakob number",
    about: "Sensible ÷ latent heat for a 5 K superheat, cp·ΔT / h_lv.",
    reading: () =>
      "Small: the superheat stores little energy against the latent heat, so growth is latent-heat-limited.",
  },
  {
    key: "Bond",
    label: "BOND",
    digits: 3,
    name: "Bond number",
    about: "Gravity ÷ surface tension, (ρ_l − ρ_v)·g·D_h² / σ.",
    reading: (v) =>
      v < 0.1
        ? "Gravity is negligible: surface tension holds the interface, independent of orientation."
        : "Gravity is comparable to surface tension and shapes the interface.",
  },
  {
    key: "hele_shaw",
    label: "HELE-SHAW",
    digits: 3,
    unit: "drag",
    name: "Hele-Shaw drag",
    about:
      "Depth-averaged wall-drag coefficient for the thin channel, 12·(L_ref/H)² / Re.",
    reading: () =>
      "How much the close top and bottom walls retard the confined flow — the gap makes it act like a Hele-Shaw cell.",
  },
  {
    key: "U_in_m_s",
    label: "U_IN",
    digits: 1,
    unit: "mm/s",
    scale: 1000,
    name: "Inlet velocity",
    about:
      "Mean liquid speed from the flow rate over the channel cross-section, Q / (w·H).",
    reading: () => "The bulk speed the liquid carries the growing bubble at.",
  },
  {
    key: "Dh_um",
    label: "D_H",
    digits: 0,
    unit: "µm",
    name: "Hydraulic diameter",
    about:
      "4·area ÷ wetted perimeter of the rectangular channel — the length scale for Re and the film.",
    reading: () => "The channel's effective diameter.",
  },
  {
    key: "bretherton_film_um",
    label: "Δ FILM",
    digits: 1,
    unit: "µm",
    name: "Bretherton film",
    about:
      "Lubricating liquid film the advancing meniscus leaves on the wall, δ ≈ 1.34·(H/2)·Ca^⅔.",
    reading: () =>
      "How much liquid stays between the bubble and the heated wall — a thicker film means weaker wall contact.",
  },
  {
    key: "t_ref_ms",
    label: "T_REF",
    digits: 2,
    unit: "ms",
    name: "Reference time",
    about:
      "Convective timescale L_ref / U_ref that non-dimensionalises the tensors' time axis.",
    reading: () => "One non-dimensional time unit; the frame interval is measured against it.",
  },
];

const DEFAULT_KEY = "Ca";

function shownValue(tile: Tile, groups: DimensionlessGroups): string {
  return (groups[tile.key] * (tile.scale ?? 1)).toFixed(tile.digits);
}

/** Dimensionless groups as mono stat tiles. Selecting a tile explains that group
 * and reads back what its value means for this run. */
export function GroupsPanel({
  datasetId,
  groups,
}: {
  datasetId: string;
  groups: DimensionlessGroups;
}) {
  const present = TILES.filter((tile) => groups[tile.key] != null);
  const [selectedKey, setSelectedKey] = useState(DEFAULT_KEY);
  const selected =
    present.find((tile) => tile.key === selectedKey) ?? present[0] ?? null;

  return (
    <Panel
      title="Derived dimensionless groups"
      subtitle={`dataset: ${datasetId}`}
    >
      <div
        className="groups"
        role="group"
        aria-label="Dimensionless groups — select one for its definition"
      >
        {present.map((tile) => {
          const isSel = selected?.key === tile.key;
          return (
            <button
              type="button"
              key={tile.key}
              className={isSel ? "gtile sel" : "gtile"}
              aria-pressed={isSel}
              onClick={() => setSelectedKey(tile.key)}
            >
              <span className="k mono">{tile.label}</span>
              <span className="v mono">
                {shownValue(tile, groups)}
                {tile.unit && <em>{tile.unit}</em>}
              </span>
            </button>
          );
        })}
      </div>
      {selected && (
        <div
          className="gdetail"
          role="region"
          aria-live="polite"
          aria-label={`${selected.name} definition`}
        >
          <div className="gdetail-hd">
            <b>{selected.name}</b>
            <span className="gdetail-val mono">
              {shownValue(selected, groups)}
              {selected.unit && <em>{selected.unit}</em>}
            </span>
          </div>
          <p>{selected.about}</p>
          <p className="gdetail-read">{selected.reading(groups[selected.key])}</p>
        </div>
      )}
    </Panel>
  );
}
