import { useEffect, useState } from "react";

import { api, type PhysicsValidation, type RunDetail, type RunSummary } from "../../lib/api";
import { errorMessage as message } from "../../lib/errors";

export type RunsLoad =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; runs: RunSummary[] };

export type DetailLoad =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; detail: RunDetail; validation: PhysicsValidation };

/**
 * Owns the Results view's data: the run list (defaulting the selection to the
 * first trained run) and the selected run's detail + validation.
 */
export function useResultsData() {
  const [runs, setRuns] = useState<RunsLoad>({ status: "loading" });
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<DetailLoad>({ status: "loading" });

  useEffect(() => {
    let alive = true;
    api
      .listRuns()
      .then((list) => {
        if (!alive) return;
        setRuns({ status: "ready", runs: list });
        const trained = list.find((r) => r.status === "trained") ?? list[0];
        setSelected(trained?.id ?? null);
      })
      .catch((err) => alive && setRuns({ status: "error", message: message(err) }));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!selected) return;
    let alive = true;
    setDetail({ status: "loading" });
    Promise.all([api.getRun(selected), api.getValidation(selected)])
      .then(([d, v]) => alive && setDetail({ status: "ready", detail: d, validation: v }))
      .catch((err) => alive && setDetail({ status: "error", message: message(err) }));
    return () => {
      alive = false;
    };
  }, [selected]);

  return { runs, selected, setSelected, detail };
}
