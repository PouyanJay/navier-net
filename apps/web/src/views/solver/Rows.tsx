import { useId, type ReactNode } from "react";

import { Switch } from "../../components";

interface ValueRowProps {
  label: ReactNode;
  /** The unit, the valid range, or what the number means. */
  hint?: ReactNode;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  suffix?: string;
  disabled?: boolean;
}

/**
 * One numeric setting as a row: name on the left, value on the right.
 *
 * The rail is 384px wide, so a two-up grid of stacked label-over-input fields
 * leaves every label wrapping onto three lines. A row gives the label the whole
 * width it needs and the number a fixed, scannable column — the same idiom the
 * Physics aside uses for its globals.
 */
export function ValueRow({
  label,
  hint,
  value,
  onChange,
  min,
  max,
  step,
  suffix,
  disabled,
}: ValueRowProps) {
  const id = useId();
  return (
    <div className="vrow">
      <label htmlFor={id}>
        {label}
        {hint && <span className="vrow-hint">{hint}</span>}
      </label>
      <span className="ug" data-disabled={disabled || undefined}>
        <input
          id={id}
          type="number"
          value={value}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          onChange={(event) => {
            const next = event.target.valueAsNumber;
            if (!Number.isNaN(next)) onChange(next);
          }}
        />
        {suffix && <span className="sfx">{suffix}</span>}
      </span>
    </div>
  );
}

interface SwitchRowProps {
  label: string;
  /** What the option does, in the interface's own voice. */
  hint?: string;
  checked: boolean;
  onChange: (on: boolean) => void;
  disabled?: boolean;
  /** Why it is unavailable, and what to do about it. Shown at the control. */
  reason?: ReactNode;
  /** The settings this option owns, revealed only while it is on. */
  children?: ReactNode;
}

/**
 * One switch and everything it governs: its own tuning fields are nested under
 * it, not collected in a grid at the foot of the rail where nothing says which
 * switch they belong to.
 */
export function SwitchRow({
  label,
  hint,
  checked,
  onChange,
  disabled,
  reason,
  children,
}: SwitchRowProps) {
  return (
    <div className="srow">
      <Switch
        label={label}
        hint={hint}
        checked={checked}
        onChange={onChange}
        disabled={disabled}
      />
      {reason && <p className="srow-why">{reason}</p>}
      {checked && children && <div className="srow-sub">{children}</div>}
    </div>
  );
}

/** A quiet sub-heading inside a band, for a group too small to be its own band. */
export function SubLabel({
  children,
  note,
}: {
  children: ReactNode;
  note?: string;
}) {
  return (
    <p className="sublabel">
      {children}
      {note && <span>{note}</span>}
    </p>
  );
}
