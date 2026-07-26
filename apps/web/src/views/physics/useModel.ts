import { useEffect, useState } from "react";

import { api, type ModelArchitecture } from "../../lib/api";
import { errorMessage } from "../../lib/errors";

export type ModelLoad =
  | { status: "loading" }
  | { status: "empty" } // no datasets to read a model from
  | { status: "error"; message: string }
  | { status: "ready"; model: ModelArchitecture };

/** The model architecture for the first available dataset (they share it). */
export function useModel(): ModelLoad {
  const [load, setLoad] = useState<ModelLoad>({ status: "loading" });

  useEffect(() => {
    let alive = true;
    api
      .listDatasets()
      .then(async (datasets) => {
        if (!alive) return;
        if (datasets.length === 0) {
          setLoad({ status: "empty" });
          return;
        }
        const model = await api.getModel(datasets[0].id);
        if (alive) setLoad({ status: "ready", model });
      })
      .catch(
        (err) =>
          alive && setLoad({ status: "error", message: errorMessage(err) }),
      );
    return () => {
      alive = false;
    };
  }, []);

  return load;
}
