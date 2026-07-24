import { useCallback, useEffect, useState } from "react";

import {
  api,
  type ConditionsResponse,
  type DatasetDetail,
  type DatasetSummary,
  type DimensionlessGroups,
  type PreprocessStatus,
} from "../../lib/api";
import { errorMessage } from "../../lib/errors";

const POLL_INTERVAL_MS = 1000;

export interface DatasetData {
  datasets: DatasetSummary[] | null;
  selected: string | null;
  setSelected: (id: string) => void;
  detail: DatasetDetail | null;
  groups: DimensionlessGroups | null;
  preprocess: PreprocessStatus | null;
  error: string | null;
  busy: boolean;
  upload: (files: FileList | File[]) => Promise<void>;
  runPreprocess: () => Promise<void>;
  /** Fold a saved conditions round-trip into the detail + groups state. */
  applyConditions: (response: ConditionsResponse) => void;
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

/**
 * Owns the Datasets view's data: the list + selection (via useDatasetList), the
 * selected dataset's detail/groups/status, preprocess polling, and the upload /
 * run-preprocess actions that refresh the affected state.
 */
export function useDatasetData(focusId?: string | null): DatasetData {
  const { datasets, selected, setSelected, refresh, error, setError } = useDatasetList(focusId);
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [groups, setGroups] = useState<DimensionlessGroups | null>(null);
  const [preprocess, setPreprocess] = useState<PreprocessStatus | null>(null);
  const [busy, setBusy] = useState(false);

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
    if (selected) loadSelected(selected).catch((err) => setError(errorMessage(err)));
  }, [selected, loadSelected, setError]);

  const onSettled = useCallback(async () => {
    if (!selected) return;
    await loadSelected(selected);
    await refresh();
  }, [selected, loadSelected, refresh]);

  usePreprocessPolling(selected, preprocess, setPreprocess, onSettled, setError);

  const upload = useCallback(
    async (files: FileList | File[]) => {
      if (!selected) return;
      setBusy(true);
      setError(null);
      try {
        await api.uploadFrames(selected, files);
        await onSettled();
      } catch (err) {
        setError(errorMessage(err));
      } finally {
        setBusy(false);
      }
    },
    [selected, onSettled, setError],
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

  const applyConditions = useCallback(
    (response: ConditionsResponse) => {
      setDetail((current) =>
        current ? { ...current, conditions: response.conditions, conditions_set: true } : current,
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
    busy,
    upload,
    runPreprocess,
    applyConditions,
  };
}
