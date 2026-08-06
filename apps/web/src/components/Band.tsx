import { useId, useState, type ComponentProps, type ReactNode } from "react";

import { HugeiconsIcon, MenuOpenIcon } from "./icons";

type IconGlyph = ComponentProps<typeof HugeiconsIcon>["icon"];

interface BandProps {
  /** The glyph naming what this band holds (icons.ts vocabulary). */
  icon: IconGlyph;
  label: string;
  /** Right-hand context: what the band's controls currently amount to. */
  hint?: ReactNode;
  /** Give the band a disclosure control. Collapsed, its hint is all that shows. */
  collapsible?: boolean;
  /** Initial state of a collapsible band. Open unless a stage says otherwise. */
  defaultOpen?: boolean;
  children: ReactNode;
}

/**
 * One labelled section of a stage's aside: a bare icon-and-label separator over
 * its rows, no plate and no border, so the sections read apart without a box
 * around each one.
 *
 * Collapsible bands keep the hint visible while folded — a folded band still
 * says what it is set to, so folding never hides state, only its controls.
 */
export function Band({
  icon,
  label,
  hint,
  collapsible,
  defaultOpen = true,
  children,
}: BandProps) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = useId();

  const heading = (
    <>
      <HugeiconsIcon icon={icon} size={13} aria-hidden="true" />
      {label}
      {hint && <span className="band-hint">{hint}</span>}
    </>
  );

  if (!collapsible) {
    return (
      <section className="band">
        <p className="band-hd">{heading}</p>
        {children}
      </section>
    );
  }

  return (
    <section className={open ? "band" : "band folded"}>
      <h3 className="band-hd">
        <button
          type="button"
          className="band-toggle"
          aria-expanded={open}
          aria-controls={bodyId}
          onClick={() => setOpen((current) => !current)}
        >
          {heading}
          <HugeiconsIcon
            className="band-chev"
            icon={MenuOpenIcon}
            size={13}
            aria-hidden="true"
          />
        </button>
      </h3>
      <div id={bodyId} hidden={!open}>
        {children}
      </div>
    </section>
  );
}
