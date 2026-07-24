import { useCallback, useMemo, useState } from "react";

import { Button, Chip, Console, LossChart, Panel, StatusDot, ViewCanvas } from "../components";
import type { RunJobStatus } from "../lib/api";
import { LossWeightsPanel, RunConfigPanel } from "./solver/ConfigPanels";
import { FORM_DEFAULTS, parseSeeds, toLaunchRequest, type SolverFormState } from "./solver/form";
import { MonitorPanel } from "./solver/MonitorPanel";
import { SweepPanel } from "./solver/SweepPanel";
import { useRunTargets } from "./solver/useRunTargets";
import { useSolverRun } from "./solver/useSolverRun";
import "./solver/solver.css";

interface SolverViewProps {
  /** Reports run-state changes so the app shell can show the training pill. */
  onRunState?: (status: RunJobStatus | null) => void;
}

interface DotState {
  tone: "default" | "accent" | "green" | "red";
  label: string;
}

function statusDot(status: RunJobStatus | null): DotState {
  if (!status) return { tone: "default", label: "idle" };
  if (status.state === "queued") return { tone: "default", label: "queued" };
  if (status.state === "running")
    return { tone: "accent", label: `running · ${status.stage ?? "train"}` };
  if (status.state === "done") return { tone: "green", label: "done" };
  return { tone: "red", label: "error" };
}

/** The Solver: configure a run on the left, watch it live on the right. */
export function SolverView({ onRunState }: SolverViewProps) {
  const [form, setForm] = useState<SolverFormState>(FORM_DEFAULTS);
  const [sweepMode, setSweepMode] = useState(false);
  const [seedsText, setSeedsText] = useState("0, 1, 2");
  const targets = useRunTargets();
  const run = useSolverRun(onRunState, targets.refreshRuns);

  const patchForm = useCallback(
    (patch: Partial<SolverFormState>) => setForm((prev) => ({ ...prev, ...patch })),
    [],
  );

  const seeds = useMemo(() => parseSeeds(seedsText), [seedsText]);

  const submit = useCallback(() => {
    const { resume, resumeRunId, dataset } = targets;
    if (sweepMode) {
      if (!dataset || !seeds) return;
      void run.startSweep({ ...toLaunchRequest(form, { dataset }), seeds });
      return;
    }
    if ((resume && !resumeRunId) || (!resume && !dataset)) return;
    const target = resume ? { resumeRunId } : { dataset };
    void run.start(toLaunchRequest(form, target));
  }, [run, form, targets, sweepMode, seeds]);

  const rebalanceSteps = useMemo(() => {
    if (run.hist.length === 0) return [];
    const last = run.hist[run.hist.length - 1].step;
    const marks: number[] = [];
    for (let s = form.rebalance_every; s <= last; s += form.rebalance_every) marks.push(s);
    return marks;
  }, [run.hist, form.rebalance_every]);

  const dot = statusDot(run.status);
  const latest = run.hist.length > 0 ? run.hist[run.hist.length - 1] : null;
  const targetReady = targets.resume ? targets.resumeRunId !== "" : targets.dataset !== "";
  const canRun = !run.running && targetReady && (!sweepMode || seeds !== null);
  const noDatasets = targets.datasets !== null && targets.datasets.length === 0;

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
      {targets.loadError && (
        <p className="state-note error" role="alert">
          {targets.loadError}
        </p>
      )}
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
            datasetOptions={(targets.datasets ?? []).map((d) => ({ value: d.id, label: d.id }))}
            dataset={targets.dataset}
            onDataset={targets.setDataset}
            resume={targets.resume}
            onResume={targets.setResume}
            resumableRuns={targets.resumableRuns}
            resumeRunId={targets.resumeRunId}
            onResumeRunId={targets.setResumeRunId}
            sweepMode={sweepMode}
            onSweepMode={(on) => {
              setSweepMode(on);
              if (on) targets.setResume(false);
            }}
            seedsText={seedsText}
            onSeedsText={setSeedsText}
            seedsValid={seeds !== null}
            locked={run.running}
          />
          <LossWeightsPanel
            weights={form.weights}
            rebalanceEvery={form.rebalance_every}
            onForm={patchForm}
            locked={run.running || targets.resume}
          />
        </div>
        <div className="solver-col">
          {run.sweep && <SweepPanel sweep={run.sweep} />}
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
