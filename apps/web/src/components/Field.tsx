import { useId, type ReactNode } from "react";

interface FieldShellProps {
  id: string;
  label: ReactNode;
  hint?: string;
  children: ReactNode;
}

/** Shared label + control shell for the config fields below. */
function FieldShell({ id, label, hint, children }: FieldShellProps) {
  return (
    <div className="fld">
      <label htmlFor={id}>
        {label}
        {hint && <span className="fld-hint"> {hint}</span>}
      </label>
      {children}
    </div>
  );
}

interface NumberFieldProps {
  label: ReactNode;
  hint?: string;
  value: number;
  onChange: (value: number) => void;
  suffix?: string;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}

/**
 * A labeled numeric input for one run-config value, with an optional unit
 * suffix. Every instance maps 1:1 onto a Hydra config field (enterprise-ui §7).
 */
export function NumberField({
  label,
  hint,
  value,
  onChange,
  suffix,
  min,
  max,
  step,
  disabled,
}: NumberFieldProps) {
  const id = useId();
  return (
    <FieldShell id={id} label={label} hint={hint}>
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
    </FieldShell>
  );
}

interface TextFieldProps {
  label: ReactNode;
  hint?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  suffix?: string;
  disabled?: boolean;
  invalid?: boolean;
}

/** A labeled free-text input (e.g. the sweep's comma-separated seed list). */
export function TextField({
  label,
  hint,
  value,
  onChange,
  placeholder,
  suffix,
  disabled,
  invalid,
}: TextFieldProps) {
  const id = useId();
  return (
    <FieldShell id={id} label={label} hint={hint}>
      <span className="ug" data-disabled={disabled || undefined}>
        <input
          id={id}
          type="text"
          value={value}
          placeholder={placeholder}
          disabled={disabled}
          aria-invalid={invalid || undefined}
          onChange={(event) => onChange(event.target.value)}
        />
        {suffix && <span className="sfx">{suffix}</span>}
      </span>
    </FieldShell>
  );
}

export interface SelectOption {
  value: string;
  label: string;
}

interface SelectFieldProps {
  label: ReactNode;
  hint?: string;
  value: string;
  onChange: (value: string) => void;
  options: SelectOption[];
  disabled?: boolean;
}

/** A labeled select for one run-config value. */
export function SelectField({
  label,
  hint,
  value,
  onChange,
  options,
  disabled,
}: SelectFieldProps) {
  const id = useId();
  return (
    <FieldShell id={id} label={label} hint={hint}>
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </FieldShell>
  );
}
