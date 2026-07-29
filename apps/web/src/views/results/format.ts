import type { RunSummary } from "../../lib/api";

/** "2026-07-26T09:14:02+00:00" → "26 Jul · 09:14" (run-browser density). */
export function formatRunDate(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return null;
  const day = date.toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
  });
  const time = date.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
  });
  return `${day} · ${time}`;
}

/** The one number a run's list row leads with, and what to call it. */
export function runHeadline(
  run: RunSummary,
): { value: string; label: string } | null {
  if (run.status === "running") {
    const total = run.steps_total ?? 0;
    const pct =
      total > 0 ? Math.round(((run.steps_done ?? 0) / total) * 100) : 0;
    return { value: `${pct}%`, label: "progress" };
  }
  if (run.val_iou_mean != null)
    return { value: run.val_iou_mean.toFixed(3), label: "val IoU" };
  if (run.iou_holdout != null)
    return { value: run.iou_holdout.toFixed(3), label: "holdout" };
  return null;
}

/** "26 Jul · 09:14 · 3 cond · 1 held out" — the row's one-line pedigree. */
export function runRowMeta(run: RunSummary): string {
  const conditions = run.datasets?.length
    ? run.datasets.length
    : run.dataset
      ? 1
      : 0;
  const parts = [
    formatRunDate(run.date)?.split(" · ")[0],
    conditions > 0 ? `${conditions} cond` : null,
    run.heldout_datasets?.length
      ? `${run.heldout_datasets.length} held out`
      : null,
    run.status === "failed" ? "failed" : null,
  ];
  return parts.filter(Boolean).join(" · ") || run.status;
}

/** The datasets a run trained on and the ones held out (axis B). */
export function runConditions(run: RunSummary): {
  all: string[];
  heldout: Set<string>;
} {
  const all = run.datasets?.length
    ? run.datasets
    : run.dataset
      ? [run.dataset]
      : [];
  return { all, heldout: new Set(run.heldout_datasets ?? []) };
}

/** Render a config object as indented YAML-ish text for the snapshot viewer.
 * Read-only display; the .hydra snapshot on disk stays the source of truth. */
export function toYamlish(value: unknown, indent = 0): string {
  if (value === null || value === undefined) return "null";
  if (Array.isArray(value))
    return `[${value.map((v) => String(v)).join(", ")}]`;
  if (typeof value !== "object") return String(value);
  const pad = "  ".repeat(indent);
  return Object.entries(value as Record<string, unknown>)
    .map(([key, entry]) =>
      entry !== null && typeof entry === "object" && !Array.isArray(entry)
        ? `${pad}${key}:\n${toYamlish(entry, indent + 1)}`
        : `${pad}${key}: ${toYamlish(entry)}`,
    )
    .join("\n");
}
