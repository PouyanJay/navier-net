import { useCallback, useEffect, useState } from "react";

import { api, type DatasetSummary, type RunSummary } from "../../lib/api";

export interface RunTargets {
  datasets: DatasetSummary[] | null;
  dataset: string;
  setDataset: (id: string) => void;
  resume: boolean;
  setResume: (on: boolean) => void;
  resumableRuns: RunSummary[];
  resumeRunId: string;
  setResumeRunId: (id: string) => void;
  refreshRuns: () => void;
  loadError: string | null;
}

/**
 * Owns what a launch can target: the processed datasets (for a new run) and
 * the trained runs (for a resume), each with its current selection. The run
 * lifecycle itself lives in `useSolverRun`.
 */
export function useRunTargets(): RunTargets {
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [dataset, setDataset] = useState("");
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
      .catch(() => setLoadError("Could not load existing runs — resume is unavailable."));
  }, []);

  useEffect(() => {
    api
      .listDatasets()
      .then((list) => {
        const processed = list.filter((entry) => entry.processed);
        setDatasets(processed);
        setDataset((cur) => cur || processed[0]?.id || "");
      })
      .catch(() => setLoadError("Could not load datasets — is the API running?"));
    refreshRuns();
  }, [refreshRuns]);

  return {
    datasets,
    dataset,
    setDataset,
    resume,
    setResume,
    resumableRuns,
    resumeRunId,
    setResumeRunId,
    refreshRuns,
    loadError,
  };
}
