import type { DatasetSummary } from "./api";

/** The human name for a series: its editable display label, or the id when
 * unset. The id stays the immutable key; this is only what the UI shows. */
export function seriesName(series: Pick<DatasetSummary, "id" | "label">): string {
  return series.label ?? series.id;
}

/** Resolve a bare series id (e.g. a run's `dataset`) to its *current* display
 * name via a datasets list. Falls back to the id when the series isn't in the
 * list (renamed-away resolves live; a deleted series shows its id). */
export function seriesNameOf(datasets: DatasetSummary[], id: string): string {
  return datasets.find((d) => d.id === id)?.label ?? id;
}
