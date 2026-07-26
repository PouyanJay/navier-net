import { useId } from "react";

interface SwitchProps {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

/** An accessible on/off switch with its visible label and an optional mono hint. */
export function Switch({
  label,
  hint,
  checked,
  onChange,
  disabled,
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
      {hint && <span className="sw-hint">{hint}</span>}
    </div>
  );
}
