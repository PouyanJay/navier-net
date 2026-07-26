import { Chip, DL, type KV, Panel, Stat } from "../../components";
import type { PhysicsValidation as Validation } from "../../lib/api";

// Inferred nose speed within this % of the measured value counts as good
// agreement (green); beyond it, flag for attention (amber).
const NOSE_SPEED_ERROR_TOLERANCE_PCT = 10;

const fmt = (v: number | null, digits = 3) =>
  v != null ? v.toFixed(digits) : "n/a";

function groupRows(v: Validation): KV[] {
  return [
    {
      label: "Bretherton film",
      value: fmt(v.bretherton_film_um, 2),
      hint: "µm",
    },
    { label: "Reynolds (Re)", value: fmt(v.reynolds, 1) },
    { label: "Weber (We)", value: fmt(v.weber, 3) },
    { label: "Capillary (Ca)", value: fmt(v.capillary, 4) },
    { label: "Prandtl (Pr)", value: fmt(v.prandtl, 2) },
    { label: "Hele-Shaw drag", value: fmt(v.hele_shaw, 3) },
  ];
}

function NoseSpeedCompare({ v }: { v: Validation }) {
  const err = v.nose_speed_error_pct;
  const errTone =
    err == null
      ? "default"
      : err < NOSE_SPEED_ERROR_TOLERANCE_PCT
        ? "green"
        : "amber";
  return (
    <div className="nose-compare">
      <Stat
        label="Nose speed · inferred"
        value={fmt(v.nose_speed_inferred_mm_s, 0)}
        unit="mm/s"
        tone="green"
        hint="no velocity data was supervised"
      />
      <Stat
        label="Nose speed · measured"
        value={fmt(v.nose_speed_measured_mm_s, 0)}
        unit="mm/s"
      />
      {err != null && (
        <div className="nose-error">
          <Chip tone={errTone}>{err.toFixed(1)}% error</Chip>
        </div>
      )}
    </div>
  );
}

interface Check {
  tone: "green" | "amber";
  title: string;
  detail: string;
}

/** Physics checks that hold (or stay open) independently of training. */
function checkRows(v: Validation): Check[] {
  const checks: Check[] = [];
  if (v.nose_speed_error_pct != null) {
    const good = v.nose_speed_error_pct < NOSE_SPEED_ERROR_TOLERANCE_PCT;
    checks.push({
      tone: good ? "green" : "amber",
      title: good
        ? "Nose speed agrees with measurement"
        : "Nose speed deviates",
      detail: `inferred ${fmt(v.nose_speed_inferred_mm_s, 0)} vs measured ${fmt(
        v.nose_speed_measured_mm_s,
        0,
      )} mm/s; neither was supervised`,
    });
  }
  if (v.bretherton_film_um != null && v.capillary != null) {
    checks.push({
      tone: "green",
      title: "Bretherton thin-film regime",
      detail: `Ca = ${fmt(v.capillary, 4)} → predicted side film δ ≈ ${fmt(
        v.bretherton_film_um,
        1,
      )} µm`,
    });
  }
  checks.push({
    tone: "amber",
    title: "Global mass closure: open",
    detail:
      "free dilatation source not yet quantitative · resolved by the Stage-B evaporation coupling",
  });
  return checks;
}

function CheckRow({ check }: { check: Check }) {
  return (
    <li className="check" data-tone={check.tone}>
      <span className="check-ic" aria-hidden="true">
        {check.tone === "green" ? "✓" : "!"}
      </span>
      <span>
        <b>{check.title}</b>
        <span className="check-sub">{check.detail}</span>
      </span>
    </li>
  );
}

/** Physics validation: inferred vs measured kinematics + key dimensionless groups. */
export function PhysicsValidationPanel({
  validation,
}: {
  validation: Validation;
}) {
  return (
    <Panel title="Physics validation" subtitle="independent of training">
      <NoseSpeedCompare v={validation} />
      <ul className="checks">
        {checkRows(validation).map((check) => (
          <CheckRow key={check.title} check={check} />
        ))}
      </ul>
      <DL items={groupRows(validation)} />
    </Panel>
  );
}
