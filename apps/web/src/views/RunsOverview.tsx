import { useEffect, useState } from "react";

import { Chip, Panel, Stat, StatusDot } from "../components";
import { api, type RunSummary } from "../lib/api";
import "./runs.css";

type Load =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; runs: RunSummary[] };

/**
 * Phase-0 walking skeleton: fetch the real runs from the API and render them.
 * Proves the whole path UI -> API -> outputs/ end to end.
 */
export function RunsOverview() {
  const [load, setLoad] = useState<Load>({ status: "loading" });

  useEffect(() => {
    let alive = true;
    api
      .listRuns()
      .then((runs) => alive && setLoad({ status: "ready", runs }))
      .catch((err: unknown) => {
        if (alive) {
          setLoad({ status: "error", message: err instanceof Error ? err.message : String(err) });
        }
      });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <Panel title="Runs" subtitle="Trained solver runs found under outputs/">
      {load.status === "loading" && (
        <p className="state-note" role="status">
          Loading runs…
        </p>
      )}

      {load.status === "error" && (
        <p className="state-note error" role="alert">
          Could not load runs: {load.message}. Is the API running on :8000?
        </p>
      )}

      {load.status === "ready" && load.runs.length === 0 && (
        <p className="state-note">No runs yet. Train a model to see it here.</p>
      )}

      {load.status === "ready" && load.runs.length > 0 && (
        <div className="runlist">
          {load.runs.map((run) => (
            <RunRow key={run.id} run={run} />
          ))}
        </div>
      )}
    </Panel>
  );
}

function RunRow({ run }: { run: RunSummary }) {
  const trained = run.status === "trained";
  const holdout = run.iou_holdout;
  return (
    <article className="runrow">
      <div>
        <span className="id">{run.id}</span>
        <div className="meta">
          {run.dataset && <Chip tone="accent">{run.dataset}</Chip>}
          <StatusDot tone={trained ? "green" : "default"} label={run.status} />
        </div>
      </div>
      {holdout != null ? (
        <Stat
          label="Holdout IoU"
          value={holdout.toFixed(3)}
          tone={holdout > 0.95 ? "green" : "amber"}
          hint="frame 6 — never supervised"
        />
      ) : (
        <Stat label="Holdout IoU" value="—" hint="not evaluated" />
      )}
    </article>
  );
}
