import { DL, type KV, Panel } from "../../components";
import type { DimensionlessGroups } from "../../lib/api";

const fmt = (v: number | undefined, digits = 3) => (v != null ? v.toFixed(digits) : "—");

function rows(g: DimensionlessGroups): KV[] {
  return [
    { label: "Reynolds (Re)", value: fmt(g.Re, 1) },
    { label: "Weber (We)", value: fmt(g.We) },
    { label: "Capillary (Ca)", value: fmt(g.Ca, 4) },
    { label: "Prandtl (Pr)", value: fmt(g.Pr, 2) },
    { label: "Bond", value: fmt(g.Bond, 4) },
    { label: "Hele-Shaw drag", value: fmt(g.hele_shaw) },
    { label: "Bretherton film", value: fmt(g.bretherton_film_um, 2), hint: "µm" },
    { label: "Hydraulic diameter", value: fmt(g.Dh_um, 0), hint: "µm" },
  ];
}

/** Dimensionless groups, computed live from the dataset's config. */
export function GroupsPanel({ groups }: { groups: DimensionlessGroups }) {
  return (
    <Panel title="Derived dimensionless groups" subtitle="Computed from the operating conditions">
      <DL items={rows(groups)} />
    </Panel>
  );
}
