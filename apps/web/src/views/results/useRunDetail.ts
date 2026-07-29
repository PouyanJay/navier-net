import { useEffect, useState } from "react";

import { api, type RunDetail } from "../../lib/api";

interface RunDetailState {
  detail: RunDetail | null;
  error: string | null;
  loading: boolean;
}

/** The selected run's full detail (metrics, config snapshot, artifacts). */
export function useRunDetail(runId: string | null): RunDetailState {
  const [state, setState] = useState<RunDetailState>({
    detail: null,
    error: null,
    loading: false,
  });

  useEffect(() => {
    if (!runId) {
      setState({ detail: null, error: null, loading: false });
      return;
    }
    let mounted = true;
    setState({ detail: null, error: null, loading: true });
    api
      .getRun(runId)
      .then((detail) => {
        if (mounted) setState({ detail, error: null, loading: false });
      })
      .catch((exc: unknown) => {
        if (mounted)
          setState({
            detail: null,
            error: exc instanceof Error ? exc.message : String(exc),
            loading: false,
          });
      });
    return () => {
      mounted = false;
    };
  }, [runId]);

  return state;
}
