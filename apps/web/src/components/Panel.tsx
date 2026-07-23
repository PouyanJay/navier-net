import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
}

/** A card grouping one concern: header (title + optional actions) and body. */
export function Panel({ title, subtitle, actions, children }: PanelProps) {
  return (
    <section className="card">
      <div className="hd">
        <div>
          <h2>{title}</h2>
          {subtitle && <p className="sub">{subtitle}</p>}
        </div>
        {actions}
      </div>
      <div className="body">{children}</div>
    </section>
  );
}
