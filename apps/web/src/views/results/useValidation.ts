import { useEffect, useState } from "react";

import { api, ApiError, type PhysicsValidation } from "../../lib/api";

interface ValidationState {
  validation: PhysicsValidation | null;
  /** Real failures only; a 404 means "not evaluated yet", which is a state. */
  error: string | null;
  loading: boolean;
}

/** The selected run's physics-validation summary (two-axis + checks). */
export function useValidation(runId: string | null): ValidationState {
  const [state, setState] = useState<ValidationState>({
    validation: null,
    error: null,
    loading: false,
  });

  useEffect(() => {
    if (!runId) {
      setState({ validation: null, error: null, loading: false });
      return;
    }
    let mounted = true;
    setState({ validation: null, error: null, loading: true });
    api
      .getValidation(runId)
      .then((validation) => {
        if (mounted) setState({ validation, error: null, loading: false });
      })
      .catch((exc: unknown) => {
        if (!mounted) return;
        if (exc instanceof ApiError && exc.status === 404) {
          setState({ validation: null, error: null, loading: false });
        } else {
          setState({
            validation: null,
            error: exc instanceof Error ? exc.message : String(exc),
            loading: false,
          });
        }
      });
    return () => {
      mounted = false;
    };
  }, [runId]);

  return state;
}
