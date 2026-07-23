import type { ReactNode } from "react";

export interface KV {
  label: string;
  value: ReactNode;
  hint?: string;
}

/** A definition list of label -> (monospace) value rows. */
export function DL({ items }: { items: KV[] }) {
  return (
    <dl className="dl">
      {items.map((item) => (
        <div key={item.label} className="dl-row">
          <dt>{item.label}</dt>
          <dd className="mono">
            {item.value}
            {item.hint && <span className="dl-hint">{item.hint}</span>}
          </dd>
        </div>
      ))}
    </dl>
  );
}
