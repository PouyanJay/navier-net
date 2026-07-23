import { Chip, DL, type KV, Panel, Stat } from "../../components";
import type { PhysicsValidation as Validation } from "../../lib/api";

const fmt = (v: number | null, digits = 3) => (v != null ? v.toFixed(digits) : "—");

/** Physics validation: inferred vs measured kinematics + key dimensionless groups. */
export function PhysicsValidation({ validation }: { validation: Validation }) {
  const err = validation.nose_speed_error_pct;
  const errTone = err == null ? "default" : err < 10 ? "green" : "amber";

  const groups: KV[] = [
    { label: "Bretherton film", value: fmt(validation.bretherton_film_um, 2), hint: "µm" },
    { label: "Reynolds (Re)", value: fmt(validation.reynolds, 1) },
    { label: "Weber (We)", value: fmt(validation.weber, 3) },
    { label: "Capillary (Ca)", value: fmt(validation.capillary, 4) },
    { label: "Prandtl (Pr)", value: fmt(validation.prandtl, 2) },
    { label: "Hele-Shaw drag", value: fmt(validation.hele_shaw, 3) },
  ];

  return (
    <Panel title="Physics validation" subtitle="Inferred kinematics vs measurement">
      <div className="nose-compare">
        <Stat
          label="Nose speed · inferred"
          value={fmt(validation.nose_speed_inferred_mm_s, 0)}
          unit="mm/s"
          tone="green"
          hint="no velocity data was supervised"
        />
        <Stat
          label="Nose speed · measured"
          value={fmt(validation.nose_speed_measured_mm_s, 0)}
          unit="mm/s"
        />
        {err != null && (
          <div style={{ paddingBottom: "var(--s1)" }}>
            <Chip tone={errTone}>{err.toFixed(1)}% error</Chip>
          </div>
        )}
      </div>
      <DL items={groups} />
    </Panel>
  );
}
