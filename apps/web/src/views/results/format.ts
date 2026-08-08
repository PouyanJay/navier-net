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

/**
 * The metric a row leads with, in the order of what it actually says about the
 * run: a score on frames the model never saw beats one on frames it did.
 *
 * The rungs matter as much as the order. `iou_holdout` belongs to the retired
 * single-frame holdout and `val_iou_mean` only to joint runs, so leading with
 * those two alone left every ordinary run showing no number at all — while the
 * `iou_val` and `iou_mean` sitting in the same metrics.json went unread.
 */
const METRIC_LADDER: {
  key: "val_iou_mean" | "iou_val" | "iou_mean" | "iou_holdout";
  label: string;
}[] = [
  { key: "val_iou_mean", label: "val IoU" },
  { key: "iou_val", label: "val IoU" },
  { key: "iou_mean", label: "mean IoU" },
  { key: "iou_holdout", label: "holdout" },
];

/** The one number a run's list row leads with, and what to call it. A number
 * with the wrong label is worse than no number, so the rung is always named. */
export function runHeadline(
  run: RunSummary,
): { value: string; label: string } | null {
  if (run.status === "running") {
    const total = run.steps_total ?? 0;
    const pct =
      total > 0 ? Math.round(((run.steps_done ?? 0) / total) * 100) : 0;
    return { value: `${pct}%`, label: "progress" };
  }
  for (const rung of METRIC_LADDER) {
    const value = run[rung.key];
    if (value != null) return { value: value.toFixed(3), label: rung.label };
  }
  return null;
}

/** The metrics a run may be ranked by, and how to read one off a summary. */
export const RANK_METRICS = [
  {
    id: "val",
    label: "val IoU",
    of: (run: RunSummary) => run.val_iou_mean ?? run.iou_val ?? null,
  },
  {
    id: "mean",
    label: "mean IoU",
    of: (run: RunSummary) => run.iou_mean ?? null,
  },
  // Ordering by date changes the ORDER, not the number: a row still leads with
  // the best score that run recorded, so a chronological list is not a list of
  // dashes. The fallback runs the SAME ladder the row's label does
  // (runHeadline) -- with only the val rungs here, a mean-only run showed a
  // dash labelled "mean IoU": a value and a label from two different ladders.
  {
    id: "date",
    label: "newest",
    of: (run: RunSummary) => run.val_iou_mean ?? run.iou_val ?? run.iou_mean ?? null,
  },
] as const;

export type RankMetricId = (typeof RANK_METRICS)[number]["id"];

export const rankMetric = (id: RankMetricId) =>
  RANK_METRICS.find((metric) => metric.id === id) ?? RANK_METRICS[0];

/**
 * The provenance chips a row carries beyond its recipe: how many conditions the
 * run spans, and how many of them it was never shown.
 *
 * Only the distinguishing ones. Every run in a single-series project spans one
 * condition, so saying "1 cond" fourteen times told nobody anything; a joint run
 * and a held-out condition are worth the space precisely because they are rare.
 */
export function runProvenance(run: RunSummary): string[] {
  const conditions = run.datasets?.length ?? (run.dataset ? 1 : 0);
  const heldout = run.heldout_datasets?.length ?? 0;
  return [
    conditions > 1 ? `${conditions} cond` : null,
    heldout > 0 ? `${heldout} held out` : null,
  ].filter((chip): chip is string => chip !== null);
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
 * Read-only display; the .hydra snapshot on disk stays the source of truth.
 * Arrays print their items via String() — exact for the config's scalar lists,
 * deliberately not a general YAML serializer. */
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

/** Nose-speed agreement within this band reads as a pass (README's 177 vs 180
 * mm/s datum sits comfortably inside it). One constant for every tab. */
export const NOSE_SPEED_TOLERANCE_PCT = 10;

export function noseSpeedTone(
  errorPct: number | null | undefined,
): "default" | "green" | "amber" {
  if (errorPct == null) return "default";
  return errorPct < NOSE_SPEED_TOLERANCE_PCT ? "green" : "amber";
}

/** IoU values print with three decimals everywhere. */
export const fmtIou = (value: number | null | undefined) =>
  value != null ? value.toFixed(3) : "—";

/** What a run is called in the UI, most specific first: a name the user typed,
 * then the series' label for legacy CLI runs auto-named after their dataset
 * (`run_name` defaulted to the dataset id), then the id itself. The id stays
 * the immutable key — provenance surfaces show it raw. */
export function runDisplayName(
  run: RunSummary,
  datasetLabels: Map<string, string>,
): string {
  if (run.label) return run.label;
  if (run.dataset && run.id === run.dataset)
    return datasetLabels.get(run.dataset) ?? run.id;
  return run.id;
}
