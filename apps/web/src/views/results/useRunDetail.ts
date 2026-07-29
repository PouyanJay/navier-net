import { api, type RunDetail } from "../../lib/api";
import { useApiResource } from "./useApiResource";

/** The selected run's full detail (metrics, config snapshot, artifacts). */
export function useRunDetail(runId: string | null) {
  const { data, error, loading } = useApiResource<RunDetail>(runId, (id) =>
    api.getRun(id),
  );
  return { detail: data, error, loading };
}
