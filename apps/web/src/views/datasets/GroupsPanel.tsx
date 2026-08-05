import { Panel } from "../../components";
import type { DimensionlessGroups } from "../../lib/api";

/**
 * A dimensionless group, or a derived scale, as the panel reads it.
 *
 * The two are presented differently because they answer different questions. A
 * group is a ratio: it says which of two effects owns the flow, so it gets a
 * dial and the reading its position produces. A scale is a magnitude with a
 * unit — micrometres, milliseconds — so it has no parity to point a needle at
 * and stays a value.
 */
interface Group {
  key: string;
  /** The group's symbol, set in brackets after its name. Null where physics
   * has no accepted one — inventing notation is worse than omitting it. */
  label: string | null;
  name: string;
  digits: number;
  /** The ratio itself, set in mono under the name. */
  ratio: string;
  /** Prose form of the ratio, for the bar cards' definition line. */
  about: string;
  /** What the current value means physically. */
  reading: (value: number) => string;
  /**
   * The decades this group is plotted over, and where its verdict flips if it
   * has such a point.
   *
   * Four of these have a published threshold, and for those the bar's job is
   * the distance to it. The rest have none, so the bar reads as a magnitude
   * and labels the span instead of drawing a tick against nothing. The spans
   * are editorial: the decades each group is normally quoted across.
   */
  regime: { min: number; max: number; threshold?: number; at?: string };
}

interface Scale {
  key: string;
  label: string;
  name: string;
  digits: number;
  unit: string;
  /** Transform from the API's value to the displayed one (e.g. m/s → mm/s). */
  scale?: number;
  ratio: string;
}

/* Every value is the pipeline's own and the definitions match
   naviernet.physics.groups. */
const GROUPS: Group[] = [
  {
    key: "Re",
    label: "Re",
    name: "Reynolds number",
    digits: 1,
    ratio: "ρ·U·L / µ",
    about: "Inertial forces ÷ viscous forces, ρ·U_ref·L_ref / µ.",
    regime: { threshold: 2300, min: 1, max: 1e5, at: "Re ≈ 2300" },
    reading: (v) =>
      v < 2300
        ? "Laminar: viscosity keeps the flow orderly."
        : "Turbulent: inertia overwhelms viscous damping.",
  },
  {
    key: "We",
    label: "We",
    name: "Weber number",
    digits: 2,
    ratio: "ρ·U²·L / σ",
    about: "Inertia ÷ surface tension, ρ·U_ref²·L_ref / σ.",
    regime: { threshold: 10, min: 1e-2, max: 1e3, at: "We ≈ 10" },
    reading: (v) =>
      v < 10
        ? "Surface tension dominates; the vapour caps stay rounded."
        : "Inertia deforms the interface out of round.",
  },
  {
    key: "Ca",
    label: "Ca",
    name: "Capillary number",
    digits: 4,
    ratio: "µ·U / σ",
    about: "Viscous drag ÷ surface tension, µ·U_ref / σ.",
    regime: { threshold: 1, min: 1e-4, max: 1e2, at: "Ca ≈ 1" },
    reading: (v) =>
      v < 1
        ? "Surface tension wins; a thin Bretherton film is left behind."
        : "Viscous drag stretches the interface into long thin films.",
  },
  {
    key: "Bond",
    label: "Bo",
    name: "Bond number",
    digits: 3,
    ratio: "Δρ·g·D² / σ",
    about: "Gravity ÷ surface tension, (ρ_l − ρ_v)·g·D_h² / σ.",
    regime: { threshold: 0.1, min: 1e-3, max: 1e2, at: "Bo ≈ 0.1" },
    reading: (v) =>
      v < 0.1
        ? "Gravity negligible; orientation does not matter."
        : "Gravity is comparable to surface tension and shapes the interface.",
  },
  {
    key: "Pr",
    label: "Pr",
    name: "Prandtl number",
    digits: 2,
    ratio: "cp·µ / k",
    about: "Momentum diffusivity ÷ thermal diffusivity, cp·µ / k.",
    regime: { min: 1e-2, max: 1e3 },
    reading: (v) => `Momentum spreads ~${v.toFixed(0)}× faster than heat.`,
  },
  {
    key: "Pe",
    label: "Pe",
    name: "Péclet number",
    digits: 0,
    ratio: "Re · Pr",
    about: "Advection ÷ thermal diffusion, Re · Pr.",
    regime: { min: 1e-2, max: 1e5 },
    reading: () => "Heat is carried by the flow, not diffused.",
  },
  {
    key: "Ja_per_5K",
    label: "Ja",
    name: "Jakob number",
    digits: 3,
    ratio: "cp·ΔT / h_lv",
    about: "Sensible ÷ latent heat for a 5 K superheat, cp·ΔT / h_lv.",
    regime: { min: 1e-3, max: 1e1 },
    reading: () => "Growth is latent-heat-limited (5 K superheat).",
  },
  {
    key: "hele_shaw",
    label: null,
    name: "Hele-Shaw drag",
    digits: 3,
    ratio: "12·(L/H)² / Re",
    about: "Depth-averaged wall drag for the thin channel, 12·(L_ref/H)² / Re.",
    regime: { min: 1e-2, max: 1e2 },
    reading: () =>
      "How much the close top and bottom walls retard the confined flow.",
  },
];

const SCALES: Scale[] = [
  {
    key: "U_in_m_s",
    label: "U_IN",
    name: "Inlet velocity",
    digits: 1,
    unit: "mm/s",
    scale: 1000,
    ratio: "Q / (w·H)",
  },
  {
    key: "Dh_um",
    label: "D_H",
    name: "Hydraulic diameter",
    digits: 0,
    unit: "µm",
    ratio: "4A / P",
  },
  {
    key: "bretherton_film_um",
    label: "Δ FILM",
    name: "Bretherton film",
    digits: 1,
    unit: "µm",
    ratio: "1.34·(H/2)·Ca^⅔",
  },
  {
    key: "t_ref_ms",
    label: "T_REF",
    name: "Reference time",
    digits: 2,
    unit: "ms",
    ratio: "L / U",
  },
];

/** Position on a log axis, 0 to 1, clamped so an extreme value stays on the bar. */
function logPosition(value: number, min: number, max: number): number {
  if (!(value > 0)) return 0;
  const t =
    (Math.log10(value) - Math.log10(min)) / (Math.log10(max) - Math.log10(min));
  return Math.min(1, Math.max(0, t));
}

/**
 * One group as a card: its value, what the ratio is, and what that value means
 * for this run.
 *
 * The four groups with a threshold also draw where the value sits against it,
 * because their number only means something relative to that flip. The rest
 * are magnitudes, so they carry the same card without the scale.
 *
 * Every group used to be a bare tile you had to select to learn anything about,
 * which meant eleven of the twelve readings were hidden at any moment.
 */
function RegimeCard({ group, value }: { group: Group; value: number }) {
  return (
    <div
      className="regime"
      role="group"
      aria-label={`${group.name} definition`}
    >
      <div className="regime-hd">
        <span className="regime-name">
          {group.name}{" "}
          {group.label && (
            <span className="regime-sym mono">({group.label})</span>
          )}
        </span>
        <span className="regime-val mono">{value.toFixed(group.digits)}</span>
      </div>
      <p className="regime-about">{group.about}</p>
      <RegimeScale regime={group.regime} value={value} />
      <p className="regime-verdict">{group.reading(value)}</p>
    </div>
  );
}

/** Where the value falls across the decades this group is quoted over. */
function RegimeScale({
  regime,
  value,
}: {
  regime: Group["regime"];
  value: number;
}) {
  const { min, max, threshold, at } = regime;
  return (
    <div className="regime-scale" aria-hidden="true">
      <span className="regime-rail" />
      {threshold !== undefined ? (
        <span
          className="regime-th"
          style={{ left: `${logPosition(threshold, min, max) * 100}%` }}
        >
          <em className="mono">{at}</em>
        </span>
      ) : (
        // No published flip, so the span itself is the reference: without it
        // the marker would sit against nothing at all.
        <>
          <em className="regime-end mono">{decade(min)}</em>
          <em className="regime-end end mono">{decade(max)}</em>
        </>
      )}
      <span
        className="regime-mk"
        style={{ left: `${logPosition(value, min, max) * 100}%` }}
      />
    </div>
  );
}

/** A power of ten as "10³", for the ends of an unthresholded scale. */
function decade(value: number): string {
  const power = Math.round(Math.log10(value));
  const digits = String(Math.abs(power))
    .split("")
    .map((d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(d)])
    .join("");
  return `10${power < 0 ? "⁻" : ""}${digits}`;
}

/** Every dimensionless group as a card, and the dimensional scales as values. */
export function GroupsPanel({
  datasetName,
  groups,
}: {
  datasetName: string;
  groups: DimensionlessGroups;
}) {
  const present = GROUPS.filter((group) => groups[group.key] != null);
  const scales = SCALES.filter((scale) => groups[scale.key] != null);

  return (
    <Panel title="Regime" subtitle={`derived from ${datasetName}`}>
      <div className="regimes">
        {present.map((group) => (
          <RegimeCard key={group.key} group={group} value={groups[group.key]} />
        ))}
      </div>

      {scales.length > 0 && (
        <>
          <p className="groups-lbl">Derived scales</p>
          {/* These carry units, so they are magnitudes rather than a contest
              between two effects and there is no ratio to plot. */}
          <div className="scales">
            {scales.map((item) => (
              <div key={item.key} className="scale" title={item.name}>
                <span className="k mono">{item.label}</span>
                <span className="v mono">
                  {(groups[item.key] * (item.scale ?? 1)).toFixed(item.digits)}
                  <em>{item.unit}</em>
                </span>
                <span className="f mono">{item.ratio}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </Panel>
  );
}
