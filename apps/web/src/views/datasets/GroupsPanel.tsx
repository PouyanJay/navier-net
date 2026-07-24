import { Panel } from "../../components";
import type { DimensionlessGroups } from "../../lib/api";

interface Tile {
  key: string;
  label: string;
  digits: number;
  unit?: string;
  /** Regime-defining tiles get the mockup's accent highlight. */
  hot?: boolean;
  /** Optional transform from the API's value to the displayed one. */
  scale?: number;
}

// Order and vocabulary follow the mockup; every value is the pipeline's own.
const TILES: Tile[] = [
  { key: "Re", label: "RE", digits: 1 },
  { key: "We", label: "WE", digits: 2 },
  { key: "Ca", label: "CA", digits: 4, hot: true },
  { key: "Pr", label: "PR", digits: 2 },
  { key: "Pe", label: "PE", digits: 0 },
  { key: "Ja_per_5K", label: "JA", digits: 3, unit: "@5K" },
  { key: "Bond", label: "BOND", digits: 3 },
  { key: "hele_shaw", label: "HELE-SHAW", digits: 3, unit: "drag" },
  { key: "U_in_m_s", label: "U_IN", digits: 1, unit: "mm/s", scale: 1000 },
  { key: "Dh_um", label: "D_H", digits: 0, unit: "µm" },
  { key: "bretherton_film_um", label: "Δ FILM", digits: 1, unit: "µm", hot: true },
  { key: "t_ref_ms", label: "T_REF", digits: 2, unit: "ms" },
];

function regimeReadback(groups: DimensionlessGroups): string | null {
  const { Re, We, Ca, Bond, bretherton_film_um: film } = groups;
  if ([Re, We, Ca, Bond, film].some((value) => value == null)) return null;
  const flow = Re < 2300 ? "laminar" : "turbulent";
  const interfaceRegime =
    We < 10 ? "surface-tension–dominated caps" : "inertia-dominated interface";
  const gravity = Bond < 0.1 ? "negligible" : "relevant";
  return (
    `${flow} (Re ${Re.toFixed(0)}), ${interfaceRegime} (We ${We.toFixed(1)}), ` +
    `Bretherton thin-film regime (Ca ${Ca.toFixed(3)} → δ = ${film.toFixed(1)} µm), ` +
    `gravity ${gravity} (Bo ${Bond.toFixed(3)}).`
  );
}

/** Dimensionless groups as mono stat tiles, recomputed with the conditions. */
export function GroupsPanel({
  datasetId,
  groups,
}: {
  datasetId: string;
  groups: DimensionlessGroups;
}) {
  const readback = regimeReadback(groups);
  return (
    <Panel title="Derived dimensionless groups" subtitle={`dataset: ${datasetId}`}>
      <div className="groups">
        {TILES.filter((tile) => groups[tile.key] != null).map((tile) => (
          <div className={tile.hot ? "gtile hot" : "gtile"} key={tile.key}>
            <div className="k mono">{tile.label}</div>
            <div className="v mono">
              {(groups[tile.key] * (tile.scale ?? 1)).toFixed(tile.digits)}
              {tile.unit && <em>{tile.unit}</em>}
            </div>
          </div>
        ))}
      </div>
      {readback && (
        <p className="note">
          <b>Regime read-back:</b> {readback}
        </p>
      )}
    </Panel>
  );
}
