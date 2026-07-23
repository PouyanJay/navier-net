import { useCallback, useEffect, useRef, useState } from "react";

import {
  api,
  type DatasetDetail,
  type DatasetSummary,
  type DimensionlessGroups,
  type PreprocessStatus,
} from "../../lib/api";

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

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
}

/**
 * Owns the Datasets view's data: the dataset list, the selected dataset's detail
 * + live groups, and the preprocess job (polled while running). Exposes upload
 * and run-preprocess actions that refresh the affected state.
 */
export function useDatasetData(focusId?: string | null): DatasetData {
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [selected, setSelected] = useState<string | null>(focusId ?? null);
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [groups, setGroups] = useState<DimensionlessGroups | null>(null);
  const [preprocess, setPreprocess] = useState<PreprocessStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const pollRef = useRef<number | null>(null);

  const refreshList = useCallback(async () => {
    const list = await api.listDatasets();
    setDatasets(list);
    setSelected((cur) => cur ?? list[0]?.id ?? null);
  }, []);

  useEffect(() => {
    refreshList().catch((err) => setError(message(err)));
  }, [refreshList]);

  // Follow an externally-focused dataset (e.g. opened from the Projects grid).
  useEffect(() => {
    if (focusId) setSelected(focusId);
  }, [focusId]);

  const loadSelected = useCallback(async (id: string) => {
    setError(null);
    const [d, g, p] = await Promise.all([
      api.getDataset(id),
      api.getDatasetGroups(id),
      api.getPreprocessStatus(id),
    ]);
    setDetail(d);
    setGroups(g);
    setPreprocess(p);
  }, []);

  useEffect(() => {
    if (selected) loadSelected(selected).catch((err) => setError(message(err)));
  }, [selected, loadSelected]);

  // Poll while a preprocess job is running; refresh detail when it finishes.
  useEffect(() => {
    if (preprocess?.state !== "running" || !selected) return;
    pollRef.current = window.setInterval(async () => {
      try {
        const status = await api.getPreprocessStatus(selected);
        setPreprocess(status);
        if (status.state !== "running") {
          window.clearInterval(pollRef.current!);
          await loadSelected(selected);
          await refreshList();
        }
      } catch (err) {
        setError(message(err));
      }
    }, 1000);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, [preprocess?.state, selected, loadSelected, refreshList]);

  const upload = useCallback(
    async (files: FileList | File[]) => {
      if (!selected) return;
      setBusy(true);
      setError(null);
      try {
        await api.uploadFrames(selected, files);
        await loadSelected(selected);
        await refreshList();
      } catch (err) {
        setError(message(err));
      } finally {
        setBusy(false);
      }
    },
    [selected, loadSelected, refreshList],
  );

  const runPreprocess = useCallback(async () => {
    if (!selected) return;
    setError(null);
    try {
      setPreprocess(await api.startPreprocess(selected));
    } catch (err) {
      setError(message(err));
    }
  }, [selected]);

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
  };
}
