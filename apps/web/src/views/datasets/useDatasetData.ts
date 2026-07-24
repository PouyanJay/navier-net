import { useCallback, useEffect, useRef, useState } from "react";

import {
  api,
  ApiError,
  type ConditionsResponse,
  type DatasetDetail,
  type DatasetSummary,
  type DimensionlessGroups,
  type PreprocessStatus,
  type QcData,
} from "../../lib/api";
import { errorMessage } from "../../lib/errors";
import { isTrainedRun } from "../../lib/runs";

const POLL_INTERVAL_MS = 1000;

export interface DatasetData {
  datasets: DatasetSummary[] | null;
  selected: string | null;
  setSelected: (id: string) => void;
  detail: DatasetDetail | null;
  groups: DimensionlessGroups | null;
  preprocess: PreprocessStatus | null;
  error: string | null;
  /** Add or remove a 1-based camera frame from the series' exclusion set. */
  toggleExcludedFrame: (frame: number) => Promise<void>;
  /** A rejected or failed exclusion edit, for the sequence panel to surface. */
  exclusionError: string | null;
  runPreprocess: () => Promise<void>;
  /** Fold a saved conditions round-trip into the detail + groups state. */
  applyConditions: (response: ConditionsResponse) => void;
  /** Re-fetch the dataset list (e.g. after a new series is uploaded). */
  refresh: () => Promise<void>;
}

/** The dataset list plus the current selection (defaulting to the first). */
function useDatasetList(focusId?: string | null) {
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [selected, setSelected] = useState<string | null>(focusId ?? null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const list = await api.listDatasets();
    setDatasets(list);
    setSelected((cur) => cur ?? list[0]?.id ?? null);
  }, []);

  useEffect(() => {
    refresh().catch((err) => setError(errorMessage(err)));
  }, [refresh]);
  useEffect(() => {
    if (focusId) setSelected(focusId); // follow an externally-focused dataset
  }, [focusId]);

  return { datasets, selected, setSelected, refresh, error, setError };
}

/** Poll a running preprocess job to completion, then run onSettled once. */
function usePreprocessPolling(
  selected: string | null,
  preprocess: PreprocessStatus | null,
  setPreprocess: (s: PreprocessStatus) => void,
  onSettled: () => Promise<void>,
  onError: (msg: string) => void,
) {
  useEffect(() => {
    if (preprocess?.state !== "running" || !selected) return;
    const id = window.setInterval(async () => {
      try {
        const status = await api.getPreprocessStatus(selected);
        setPreprocess(status);
        if (status.state !== "running") {
          window.clearInterval(id);
          await onSettled();
        }
      } catch (err) {
        window.clearInterval(id);
        onError(errorMessage(err));
      }
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [preprocess?.state, selected, setPreprocess, onSettled, onError]);
}

/** Toggle a frame in a sorted exclusion set, leaving it sorted. */
function withFrameToggled(excluded: number[], frame: number): number[] {
  return excluded.includes(frame)
    ? excluded.filter((n) => n !== frame)
    : [...excluded, frame].sort((a, b) => a - b);
}

/**
 * Owns the Datasets view's data: the list + selection (via useDatasetList), the
 * selected dataset's detail/groups/status, preprocess polling, and the upload /
 * run-preprocess actions that refresh the affected state.
 */
export function useDatasetData(focusId?: string | null): DatasetData {
  const { datasets, selected, setSelected, refresh, error, setError } =
    useDatasetList(focusId);
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [groups, setGroups] = useState<DimensionlessGroups | null>(null);
  const [preprocess, setPreprocess] = useState<PreprocessStatus | null>(null);
  const [exclusionError, setExclusionError] = useState<string | null>(null);
  // The set the user has asked for, updated synchronously on click so a burst of
  // toggles composes instead of each one racing off the same stale render.
  const wanted = useRef<number[]>([]);
  useEffect(() => {
    wanted.current = detail?.excluded_frames ?? [];
  }, [detail]);

  const loadSelected = useCallback(
    async (id: string) => {
      setError(null);
      const [d, g, p] = await Promise.all([
        api.getDataset(id),
        api.getDatasetGroups(id),
        api.getPreprocessStatus(id),
      ]);
      setDetail(d);
      setGroups(g);
      setPreprocess(p);
    },
    [setError],
  );

  useEffect(() => {
    if (selected)
      loadSelected(selected).catch((err) => setError(errorMessage(err)));
  }, [selected, loadSelected, setError]);

  const onSettled = useCallback(async () => {
    if (!selected) return;
    await loadSelected(selected);
    await refresh();
  }, [selected, loadSelected, refresh]);

  usePreprocessPolling(
    selected,
    preprocess,
    setPreprocess,
    onSettled,
    setError,
  );

  const runPreprocess = useCallback(async () => {
    if (!selected) return;
    setError(null);
    try {
      setPreprocess(await api.startPreprocess(selected));
    } catch (err) {
      setError(errorMessage(err));
    }
  }, [selected, setError]);

  const toggleExcludedFrame = useCallback(
    async (frame: number) => {
      if (!selected) return;
      const next = withFrameToggled(wanted.current, frame);
      wanted.current = next;
      setExclusionError(null);
      // Optimistic: the border follows the click, then the server confirms it.
      setDetail((current) =>
        current ? { ...current, excluded_frames: next } : current,
      );
      try {
        const saved = await api.setExcludedFrames(selected, next);
        if (wanted.current === next) setDetail(saved); // a later toggle wins
      } catch (err) {
        setExclusionError(errorMessage(err));
        // The strip must never show an exclusion the server did not accept.
        await loadSelected(selected).catch((reloadErr) =>
          setError(errorMessage(reloadErr)),
        );
      }
    },
    [selected, loadSelected, setError],
  );

  const applyConditions = useCallback(
    (response: ConditionsResponse) => {
      setDetail((current) =>
        current
          ? {
              ...current,
              conditions: response.conditions,
              conditions_set: true,
            }
          : current,
      );
      setGroups(response.groups);
      refresh().catch(() => {}); // library chips ("needs conditions") follow
    },
    [refresh],
  );

  return {
    datasets,
    selected,
    setSelected,
    detail,
    groups,
    preprocess,
    error,
    toggleExcludedFrame,
    exclusionError,
    runPreprocess,
    applyConditions,
    refresh,
  };
}

/** Dataset ids with a trained run; drives the library's status chips. */
export function useTrainedIds(): Set<string> {
  const [ids, setIds] = useState<Set<string>>(new Set());
  useEffect(() => {
    api
      .listRuns()
      .then((runs) =>
        setIds(
          new Set(
            runs
              .filter(isTrainedRun)
              .flatMap((run) => (run.dataset ? [run.dataset] : [])),
          ),
        ),
      )
      .catch(() => setIds(new Set())); // chips only; panels surface real errors
  }, []);
  return ids;
}

/** The selected series' QC chart data, refreshed after preprocessing.
 * A 404 means "not preprocessed yet" (expected); anything else is surfaced. */
export function useQcData(
  selected: string | null,
  processed: boolean,
): { qc: QcData | null; qcError: string | null } {
  const [qc, setQc] = useState<QcData | null>(null);
  const [qcError, setQcError] = useState<string | null>(null);

  useEffect(() => {
    let stale = false;
    setQc(null);
    setQcError(null);
    if (!selected || !processed) return;
    api
      .getQcData(selected)
      .then((payload) => {
        if (!stale) setQc(payload);
      })
      .catch((err) => {
        if (stale || (err instanceof ApiError && err.status === 404)) return;
        setQcError(errorMessage(err));
      });
    return () => {
      stale = true;
    };
  }, [selected, processed]);

  return { qc, qcError };
}
