import { useId, type ReactNode } from "react";

import { Switch } from "../../components";
import { HugeiconsIcon, WarningIcon } from "../../components/icons";

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

/**
 * What the bench said about an option, carried ON the option.
 *
 * The platform's discipline is default-off-until-benched, so the verdict is
 * exactly the guidance a user needs at the switch: "positive" earned a measured
 * gain, "negative" degraded every arm tried, "inert" was benched and moved
 * nothing, "unbenched" has no numbers yet, and "recipe" ships on by default.
 */
export type VerdictTone =
  "positive" | "negative" | "inert" | "unbenched" | "recipe";

export interface Verdict {
  tone: VerdictTone;
  /** The measured number, when one exists (e.g. "+0.039 val"). */
  note?: string;
}

const VERDICT_TEXT: Record<VerdictTone, string> = {
  positive: "benched +",
  negative: "benched −",
  inert: "benched ∅",
  unbenched: "unbenched",
  recipe: "default on",
};

const VERDICT_CLASS: Record<VerdictTone, string> = {
  positive: "vchip pos",
  negative: "vchip neg",
  inert: "vchip inert",
  unbenched: "vchip open",
  recipe: "vchip recipe",
};

/** The verdict as a small chip beside a switch's label. */
export function VerdictChip({ verdict }: { verdict: Verdict }) {
  return (
    <span className={VERDICT_CLASS[verdict.tone]}>
      {VERDICT_TEXT[verdict.tone]}
      {verdict.note ? ` · ${verdict.note}` : ""}
    </span>
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
  /** What the bench said about this option, shown beside its label. */
  verdict?: Verdict;
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
  verdict,
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
        badge={verdict ? <VerdictChip verdict={verdict} /> : undefined}
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

/** A caution rendered at the control that caused it: why a choice is withheld
 * or inconsistent, and what to do about it. */
export function Caution({ children }: { children: ReactNode }) {
  return (
    <p className="rail-caution" role="status">
      <HugeiconsIcon icon={WarningIcon} size={13} aria-hidden="true" />
      <span>{children}</span>
    </p>
  );
}
