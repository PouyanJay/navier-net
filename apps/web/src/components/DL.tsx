import type { ReactNode } from "react";

export interface KV {
  label: string;
  value: ReactNode;
  hint?: string;
  /**
   * Optional heading this row sits under. Consecutive rows sharing a group are
   * banded together; rows without one are listed plainly, so the flat callers
   * are unaffected.
   */
  group?: string;
}

/** A definition list of label -> (monospace) value rows, optionally banded. */
export function DL({ items }: { items: KV[] }) {
  return (
    <dl className="dl">
      {bands(items).map((band, index) => (
        <div key={band.group ?? index} className="dl-band">
          {band.group && <p className="dl-bandlbl">{band.group}</p>}
          {band.items.map((item) => (
            <div key={item.label} className="dl-row">
              <dt>{item.label}</dt>
              <dd className="mono">
                {item.value}
                {item.hint && <span className="dl-hint">{item.hint}</span>}
              </dd>
            </div>
          ))}
        </div>
      ))}
    </dl>
  );
}

/** Runs of consecutive rows sharing a group, in the order they were given. */
function bands(items: KV[]): { group?: string; items: KV[] }[] {
  const out: { group?: string; items: KV[] }[] = [];
  for (const item of items) {
    const last = out[out.length - 1];
    if (last && last.group === item.group) last.items.push(item);
    else out.push({ group: item.group, items: [item] });
  }
  return out;
}
