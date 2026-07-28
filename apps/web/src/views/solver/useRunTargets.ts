import { useCallback, useEffect, useState } from "react";

import { api, type DatasetSummary, type RunSummary } from "../../lib/api";

export interface RunTargets {
  /** Processed datasets a new run can train on, scoped to the open project. */
  available: DatasetSummary[] | null;
  /** The subset selected to train; more than one means a joint run. */
  selected: string[];
  toggleDataset: (id: string) => void;
  selectAll: (on: boolean) => void;
  /** The one selected series held out of training (axis B): loaded, never
   * supervised, scored as a transfer test. "" = none. Always one of `selected`. */
  heldout: string;
  setHeldout: (id: string) => void;
  resume: boolean;
  setResume: (on: boolean) => void;
  resumableRuns: RunSummary[];
  resumeRunId: string;
  setResumeRunId: (id: string) => void;
  refreshRuns: () => void;
  loadError: string | null;
}

/**
 * Owns what a launch can target: the project's processed datasets (multi-select
 * for a joint run) and the trained runs (for a resume), each with its current
 * selection. The run lifecycle itself lives in `useSolverRun`.
 *
 * `projectDatasets` scopes the dataset list to the open project; `null` (no
 * project) leaves it workspace-wide.
 */
export function useRunTargets(projectDatasets: string[] | null): RunTargets {
  const [available, setAvailable] = useState<DatasetSummary[] | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [heldout, setHeldout] = useState<string>("");
  const [resume, setResume] = useState(false);
  const [resumableRuns, setResumableRuns] = useState<RunSummary[]>([]);
  const [resumeRunId, setResumeRunId] = useState("");
  const [loadError, setLoadError] = useState<string | null>(null);

  const refreshRuns = useCallback(() => {
    api
      .listRuns()
      .then((list) => {
        const trained = list.filter((run) => run.status === "trained");
        setResumableRuns(trained);
        setResumeRunId((cur) => cur || trained[0]?.id || "");
      })
      // Launching still works without the list, but say why resume is empty.
      .catch(() =>
        setLoadError("Could not load existing runs; resume is unavailable."),
      );
  }, []);

  // A stable dependency for the effect: the project's dataset set as a string.
  const scopeKey = projectDatasets ? projectDatasets.join(",") : "";

  useEffect(() => {
    api
      .listDatasets()
      .then((list) => {
        const scope = projectDatasets;
        const processed = list.filter(
          (entry) =>
            entry.processed && (scope === null || scope.includes(entry.id)),
        );
        setAvailable(processed);
        // Default to training the whole project; the user can narrow it.
        setSelected(processed.map((entry) => entry.id));
        setHeldout("");
      })
      .catch(() =>
        setLoadError("Could not load datasets; is the API running?"),
      );
    refreshRuns();
    // scopeKey stands in for projectDatasets' identity so the effect re-runs
    // only when the project's dataset set actually changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshRuns, scopeKey]);

  const toggleDataset = useCallback((id: string) => {
    setSelected((cur) => {
      const next = cur.includes(id)
        ? cur.filter((x) => x !== id)
        : [...cur, id];
      // Hold-out must stay one of the selected series, and needs at least two
      // selected (something left to train on) -- otherwise clear it, so a
      // deselection can never strand the run in an all-held-out state.
      setHeldout((held) =>
        held && next.includes(held) && next.length > 1 ? held : "",
      );
      return next;
    });
  }, []);

  const selectAll = useCallback(
    (on: boolean) => {
      setSelected(on ? (available ?? []).map((entry) => entry.id) : []);
      if (!on) setHeldout("");
    },
    [available],
  );

  return {
    available,
    selected,
    toggleDataset,
    selectAll,
    heldout,
    setHeldout,
    resume,
    setResume,
    resumableRuns,
    resumeRunId,
    setResumeRunId,
    refreshRuns,
    loadError,
  };
}
