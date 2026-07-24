import { Panel } from "../../components";
import type { DimensionlessGroups } from "../../lib/api";

interface Tile {
  key: string;
  label: string;
  digits: number;
  unit?: string;
}

const TILES: Tile[] = [
  { key: "Re", label: "RE", digits: 1 },
  { key: "We", label: "WE", digits: 2 },
  { key: "Ca", label: "CA", digits: 4 },
  { key: "Pr", label: "PR", digits: 2 },
  { key: "Bond", label: "BOND", digits: 3 },
  { key: "hele_shaw", label: "HELE-SHAW", digits: 3 },
  { key: "bretherton_film_um", label: "Δ FILM", digits: 1, unit: "µm" },
  { key: "Dh_um", label: "D_H", digits: 0, unit: "µm" },
];

/** Dimensionless groups as mono stat tiles, recomputed with the conditions. */
export function GroupsPanel({
  datasetId,
  groups,
}: {
  datasetId: string;
  groups: DimensionlessGroups;
}) {
  return (
    <Panel title="Derived dimensionless groups" subtitle={`dataset: ${datasetId}`}>
      <div className="groups">
        {TILES.filter((tile) => groups[tile.key] != null).map((tile) => (
          <div className="gtile" key={tile.key}>
            <div className="k mono">{tile.label}</div>
            <div className="v mono">
              {groups[tile.key].toFixed(tile.digits)}
              {tile.unit && <em>{tile.unit}</em>}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
