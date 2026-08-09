import { useId, type ReactNode } from "react";

interface SwitchProps {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  /** Rendered beside the label, outside it: a badge is context (a verdict, a
   * count), not part of the switch's accessible name. */
  badge?: ReactNode;
}

/** An accessible on/off switch with its visible label and an optional mono hint. */
export function Switch({
  label,
  hint,
  checked,
  onChange,
  disabled,
  badge,
}: SwitchProps) {
  const labelId = useId();
  return (
    <div className="wsw">
      <button
        type="button"
        role="switch"
        className="sw"
        aria-checked={checked}
        aria-labelledby={labelId}
        disabled={disabled}
        onClick={() => onChange(!checked)}
      >
        <span className="knob" aria-hidden="true" />
      </button>
      <span className="sw-label" id={labelId}>
        {label}
      </span>
      {badge}
      {hint && <span className="sw-hint">{hint}</span>}
    </div>
  );
}
