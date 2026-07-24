export type Tone = "default" | "green" | "amber" | "red";

interface StatProps {
  label: string;
  value: string | number;
  unit?: string;
  tone?: Tone;
  hint?: string;
}

/** A single metric: label, a monospace value, optional unit and hint. */
export function Stat({
  label,
  value,
  unit,
  tone = "default",
  hint,
}: StatProps) {
  return (
    <div className="stat" data-tone={tone}>
      <span className="lbl">{label}</span>
      <span className="val">
        {value}
        {unit && <span className="unit">{unit}</span>}
      </span>
      {hint && <span className="hint">{hint}</span>}
    </div>
  );
}
