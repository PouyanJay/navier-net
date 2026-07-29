import { useEffect, useState } from "react";

import { ApiError } from "../../lib/api";

export interface ApiResource<T> {
  data: T | null;
  /** Real failures only when `nullOn404`; a 404 then means "not there yet". */
  error: string | null;
  loading: boolean;
}

/**
 * The one fetch-into-state pattern every per-run resource shares: reset on key
 * change, guard against late responses from a stale key, and keep loading /
 * error / data as one value. `key = null` means "nothing selected".
 */
export function useApiResource<T>(
  key: string | null,
  fetcher: (key: string) => Promise<T>,
  options: { nullOn404?: boolean } = {},
): ApiResource<T> {
  const { nullOn404 = false } = options;
  const [state, setState] = useState<ApiResource<T>>({
    data: null,
    error: null,
    loading: false,
  });

  useEffect(() => {
    if (key == null) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    let mounted = true;
    setState({ data: null, error: null, loading: true });
    fetcher(key)
      .then((data) => {
        if (mounted) setState({ data, error: null, loading: false });
      })
      .catch((exc: unknown) => {
        if (!mounted) return;
        if (nullOn404 && exc instanceof ApiError && exc.status === 404) {
          setState({ data: null, error: null, loading: false });
        } else {
          setState({
            data: null,
            error: exc instanceof Error ? exc.message : String(exc),
            loading: false,
          });
        }
      });
    return () => {
      mounted = false;
    };
    // `fetcher` is intentionally not a dependency: callers pass inline
    // functions whose identity changes every render but whose behavior is
    // fully determined by `key`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, nullOn404]);

  return state;
}
