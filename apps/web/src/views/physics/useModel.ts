import { useEffect, useState } from "react";

import { api, type ModelArchitecture } from "../../lib/api";
import { errorMessage } from "../../lib/errors";

/** The model architecture for the first available dataset (they share it). */
export function useModel() {
  const [model, setModel] = useState<ModelArchitecture | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .listDatasets()
      .then((datasets) => {
        if (!alive || datasets.length === 0) return;
        return api.getModel(datasets[0].id).then((m) => alive && setModel(m));
      })
      .catch((err) => alive && setError(errorMessage(err)));
    return () => {
      alive = false;
    };
  }, []);

  return { model, error };
}
