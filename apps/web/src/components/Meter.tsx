interface MeterProps {
  value: number;
  max: number;
  label: string;
}

/** A thin determinate progress bar (the run's step progress). */
export function Meter({ value, max, label }: MeterProps) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div
      className="meter"
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={max}
      aria-valuenow={Math.min(value, max)}
    >
      <i style={{ width: `${pct}%` }} />
    </div>
  );
}
