import { useCallback, useEffect, useState } from "react";

import { api, type LossRecord, type RunDetail, type RunSummary } from "../../lib/api";
import { errorMessage } from "../../lib/errors";

/** Comparison is capped at the number of distinct series colors. */
export const MAX_COMPARED = 4;

export interface ComparedRun {
  detail: RunDetail;
  history: LossRecord[];
}

export interface Comparison {
  candidates: RunSummary[];
  selected: string[];
  toggle: (id: string) => void;
  runs: ComparedRun[] | null; // null while loading
  error: string | null;
}

/**
 * Owns the comparison selection and its data: for each selected run, the
 * detail (metrics) plus loss history. Sweep children sort adjacent naturally
 * (they share the sweep-id prefix).
 */
export function useComparison(candidates: RunSummary[]): Comparison {
  const [selected, setSelected] = useState<string[]>([]);
  const [runs, setRuns] = useState<ComparedRun[] | null>([]);
  const [error, setError] = useState<string | null>(null);

  const toggle = useCallback((id: string) => {
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((entry) => entry !== id);
      if (prev.length >= MAX_COMPARED) return prev;
      return [...prev, id];
    });
  }, []);

  useEffect(() => {
    if (selected.length === 0) {
      setRuns([]);
      setError(null);
      return;
    }
    let alive = true;
    setRuns(null);
    Promise.all(
      selected.map(async (id) => {
        const [detail, history] = await Promise.all([
          api.getRun(id),
          // Loss history may legitimately be absent (e.g. an imported run).
          api.getLossHistory(id).catch(() => [] as LossRecord[]),
        ]);
        return { detail, history };
      }),
    )
      .then((loaded) => {
        if (!alive) return;
        setRuns(loaded);
        setError(null);
      })
      .catch((err) => {
        if (!alive) return;
        setRuns([]);
        setError(errorMessage(err));
      });
    return () => {
      alive = false;
    };
  }, [selected]);

  return { candidates, selected, toggle, runs, error };
}
