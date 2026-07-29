import { api, type PhysicsValidation } from "../../lib/api";
import { useApiResource } from "./useApiResource";

/** The selected run's physics-validation summary (two-axis + checks).
 * A 404 means "not evaluated yet" — a state, not an error. */
export function useValidation(runId: string | null) {
  const { data, error, loading } = useApiResource<PhysicsValidation>(
    runId,
    (id) => api.getValidation(id),
    { nullOn404: true },
  );
  return { validation: data, error, loading };
}
