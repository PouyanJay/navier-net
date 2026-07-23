import { useEffect, useState } from "react";

import { Chip, StatusDot } from "../components";
import { api, type PhysicsValidation, type RunDetail, type RunSummary } from "../lib/api";
import { AgreementPerFrame } from "./results/AgreementPerFrame";
import { Deliverables } from "./results/Deliverables";
import { PhysicsValidation as PhysicsPanel } from "./results/PhysicsValidation";
import "./results/results.css";
import "./runs.css";

type RunsLoad =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; runs: RunSummary[] };

type DetailLoad =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; detail: RunDetail; validation: PhysicsValidation };

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function ResultsView() {
  const [runs, setRuns] = useState<RunsLoad>({ status: "loading" });
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<DetailLoad>({ status: "loading" });

  // Load the run list, then default to the first trained run.
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

  // Load the selected run's detail + validation.
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

  if (runs.status === "loading") {
    return <p className="state-note" role="status">Loading runs…</p>;
  }
  if (runs.status === "error") {
    return (
      <p className="state-note error" role="alert">
        Could not load runs: {runs.message}. Is the API running on :8000?
      </p>
    );
  }
  if (runs.runs.length === 0 || !selected) {
    return <p className="state-note">No runs yet. Train a model to see results here.</p>;
  }

  return (
    <div className="stack">
      <RunHeader
        runs={runs.runs}
        selected={selected}
        onSelect={setSelected}
        detail={detail.status === "ready" ? detail.detail : null}
      />

      {detail.status === "loading" && (
        <p className="state-note" role="status">Loading results…</p>
      )}
      {detail.status === "error" && (
        <p className="state-note error" role="alert">
          Could not load results: {detail.message}
        </p>
      )}
      {detail.status === "ready" && (
        <>
          <AgreementPerFrame detail={detail.detail} />
          <PhysicsPanel validation={detail.validation} />
          <Deliverables runId={detail.detail.id} artifacts={detail.detail.artifacts} />
        </>
      )}
    </div>
  );
}

interface RunHeaderProps {
  runs: RunSummary[];
  selected: string;
  onSelect: (id: string) => void;
  detail: RunDetail | null;
}

function RunHeader({ runs, selected, onSelect, detail }: RunHeaderProps) {
  const current = runs.find((r) => r.id === selected);
  return (
    <div className="results-head">
      <span className="id">{selected}</span>
      {current?.dataset && <Chip tone="accent">{current.dataset}</Chip>}
      {current && (
        <StatusDot tone={current.status === "trained" ? "green" : "default"} label={current.status} />
      )}
      {detail?.steps != null && <span className="mono steps">{detail.steps} steps</span>}
      {runs.length > 1 && (
        <select aria-label="Select run" value={selected} onChange={(e) => onSelect(e.target.value)}>
          {runs.map((r) => (
            <option key={r.id} value={r.id}>
              {r.id}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
