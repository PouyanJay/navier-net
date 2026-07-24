import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
}

/** A card grouping one concern: header (title left, mono subtitle right,
 * optional actions) and body — the mockup's card/hd/sub vocabulary. */
export function Panel({ title, subtitle, actions, children }: PanelProps) {
  return (
    <section className="card">
      <div className="hd">
        <h2>{title}</h2>
        {subtitle && <span className="sub">{subtitle}</span>}
        {actions}
      </div>
      <div className="body">{children}</div>
    </section>
  );
}
