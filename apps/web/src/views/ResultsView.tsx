import { Chip, StatusDot } from "../components";
import type { RunDetail, RunSummary } from "../lib/api";
import { AgreementPerFrame } from "./results/AgreementPerFrame";
import { Deliverables } from "./results/Deliverables";
import { PhysicsValidationPanel } from "./results/PhysicsValidation";
import { useResultsData } from "./results/useResultsData";
import "./results/results.css";
import "./runs.css";

export function ResultsView() {
  const { runs, selected, setSelected, detail } = useResultsData();

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
          <PhysicsValidationPanel validation={detail.validation} />
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
  const statusTone = current?.status === "trained" ? "green" : "default";
  return (
    <div className="results-head">
      <span className="id">{selected}</span>
      {current?.dataset && <Chip tone="accent">{current.dataset}</Chip>}
      {current && <StatusDot tone={statusTone} label={current.status} />}
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
