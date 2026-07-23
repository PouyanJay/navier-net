import { useCallback, useEffect, useMemo, useState } from "react";

import { Button, Chip, Console, LossChart, Panel, StatusDot, ViewCanvas } from "../components";
import { api, type DatasetSummary, type RunJobStatus, type RunSummary } from "../lib/api";
import { LossWeightsPanel, RunConfigPanel } from "./solver/ConfigPanels";
import { FORM_DEFAULTS, toLaunchRequest, type SolverFormState } from "./solver/form";
import { MonitorPanel } from "./solver/MonitorPanel";
import { useSolverRun } from "./solver/useSolverRun";
import "./solver/solver.css";

interface SolverViewProps {
  /** Reports run-state changes so the app shell can show the training pill. */
  onRunState?: (status: RunJobStatus | null) => void;
}

function statusDot(status: RunJobStatus | null): { tone: "default" | "accent" | "green" | "red"; label: string } {
  if (!status) return { tone: "default", label: "idle" };
  if (status.state === "running")
    return { tone: "accent", label: `running · ${status.stage ?? "train"}` };
  if (status.state === "done") return { tone: "green", label: "done" };
  return { tone: "red", label: "error" };
}

/** The Solver: configure a run on the left, watch it live on the right. */
export function SolverView({ onRunState }: SolverViewProps) {
  const [form, setForm] = useState<SolverFormState>(FORM_DEFAULTS);
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [dataset, setDataset] = useState("");
  const [resume, setResume] = useState(false);
  const [resumeRunId, setResumeRunId] = useState("");
  const [loadError, setLoadError] = useState<string | null>(null);

  const refreshRuns = useCallback(() => {
    api
      .listRuns()
      .then((list) => {
        const trained = list.filter((run) => run.status === "trained");
        setRuns(trained);
        setResumeRunId((cur) => cur || trained[0]?.id || "");
      })
      .catch(() => {}); // resume stays unavailable; launching still works
  }, []);

  const run = useSolverRun(onRunState, refreshRuns);

  useEffect(() => {
    api
      .listDatasets()
      .then((list) => {
        const processed = list.filter((entry) => entry.processed);
        setDatasets(processed);
        setDataset((cur) => cur || processed[0]?.id || "");
      })
      .catch(() => setLoadError("Could not load datasets — is the API running?"));
    refreshRuns();
  }, [refreshRuns]);

  const patchForm = useCallback(
    (patch: Partial<SolverFormState>) => setForm((prev) => ({ ...prev, ...patch })),
    [],
  );

  const submit = useCallback(() => {
    const target = resume ? { resumeRunId } : { dataset };
    if ((resume && !resumeRunId) || (!resume && !dataset)) return;
    void run.start(toLaunchRequest(form, target));
  }, [run, form, resume, resumeRunId, dataset]);

  const rebalanceSteps = useMemo(() => {
    if (run.hist.length === 0) return [];
    const last = run.hist[run.hist.length - 1].step;
    const marks: number[] = [];
    for (let s = form.rebalance_every; s <= last; s += form.rebalance_every) marks.push(s);
    return marks;
  }, [run.hist, form.rebalance_every]);

  const dot = statusDot(run.status);
  const latest = run.hist.length > 0 ? run.hist[run.hist.length - 1] : null;
  const canRun = !run.running && (resume ? resumeRunId !== "" : dataset !== "");
  const noDatasets = datasets !== null && datasets.length === 0;

  return (
    <>
      <div className="solver-head">
        <StatusDot tone={dot.tone} label={dot.label} />
        {run.status && <span className="id">{run.status.run_id}</span>}
        {run.status?.state === "running" && run.status.stage && (
          <Chip tone="accent">{run.status.stage}</Chip>
        )}
        <div className="actions">
          <Button onClick={run.reset} disabled={run.running}>
            Reset
          </Button>
          <Button variant="primary" onClick={submit} disabled={!canRun}>
            Run
          </Button>
        </div>
      </div>
      {run.error && (
        <p className="state-note error" role="alert">
          {run.error}
        </p>
      )}
      {run.status?.state === "error" && run.status.message && (
        <p className="state-note error" role="alert">
          Run failed: {run.status.message}
        </p>
      )}
      {loadError && <p className="state-note error">{loadError}</p>}
      {noDatasets && (
        <p className="state-note">
          No preprocessed dataset yet — upload and preprocess one under Datasets &amp;
          conditions to enable the solver.
        </p>
      )}
      <div className="solx">
        <div className="solver-col">
          <RunConfigPanel
            form={form}
            onForm={patchForm}
            datasetOptions={(datasets ?? []).map((d) => ({ value: d.id, label: d.id }))}
            dataset={dataset}
            onDataset={setDataset}
            resume={resume}
            onResume={setResume}
            resumableRuns={runs}
            resumeRunId={resumeRunId}
            onResumeRunId={setResumeRunId}
            locked={run.running}
          />
          <LossWeightsPanel
            weights={form.weights}
            rebalanceEvery={form.rebalance_every}
            onForm={patchForm}
            locked={run.running || resume}
          />
        </div>
        <div className="solver-col">
          <MonitorPanel status={run.status} latest={latest} holdoutIou={run.holdoutIou} />
          <Panel title="Loss history" subtitle="log₁₀ · rebalance markers">
            <ViewCanvas>
              {run.hist.length >= 2 ? (
                <LossChart records={run.hist} rebalanceSteps={rebalanceSteps} />
              ) : (
                <p className="canvas-note">
                  Loss history appears once the run logs its first records.
                </p>
              )}
            </ViewCanvas>
          </Panel>
          <Panel title="Solver console" subtitle="pipeline log · live">
            <Console lines={run.lines} label="Solver console" />
          </Panel>
        </div>
      </div>
    </>
  );
}
