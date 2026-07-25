import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  /** Extra class on the card, e.g. for a layout variant. */
  className?: string;
}

/** A card grouping one concern: header (title left, mono subtitle right,
 * optional actions) and body, matching the mockup's card/hd/sub vocabulary. */
export function Panel({
  title,
  subtitle,
  actions,
  children,
  className,
}: PanelProps) {
  return (
    <section className={className ? `card ${className}` : "card"}>
      <div className="hd">
        <h2>{title}</h2>
        {subtitle && <span className="sub">{subtitle}</span>}
        {actions}
      </div>
      <div className="body">{children}</div>
    </section>
  );
}
