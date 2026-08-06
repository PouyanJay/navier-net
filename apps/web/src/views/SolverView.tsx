import { useCallback, useMemo, useState } from "react";

import { StageAside } from "../app/StageAside";
import { ChartFrame } from "../components/ChartFrame";
import {
  Callout,
  Chip,
  Console,
  LossChart,
  Panel,
  StatusDot,
  ViewCanvas,
} from "../components";
import type { ProjectSummary, RunJobStatus } from "../lib/api";
import { useCapabilityGuard } from "./solver/capabilityGuard";
import {
  FORM_DEFAULTS,
  parseSeeds,
  toLaunchRequest,
  type SolverFormState,
} from "./solver/form";
import { MonitorPanel } from "./solver/MonitorPanel";
import type { LaunchMode } from "./solver/RunBand";
import { SolverAside } from "./solver/SolverAside";
import { SweepPanel } from "./solver/SweepPanel";
import { useRunTargets } from "./solver/useRunTargets";
import { useSeriesCapability } from "./solver/useSeriesCapability";
import { useSolverRun } from "./solver/useSolverRun";
import "./solver/solver.css";

/** The rail is a stack of labelled rows, not a two-up field grid, so it needs
 * less room than the Physics aside — enough for a long label and its value. */
const ASIDE = {
  title: "Solver",
  subtitle: "configure & run",
  width: 400,
};

interface SolverViewProps {
  /** Reports run-state changes so the app shell can show the training pill. */
  onRunState?: (status: RunJobStatus | null) => void;
  /** Scopes the trainable datasets to the open project (null = workspace-wide). */
  project?: ProjectSummary | null;
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
export function SolverView({ onRunState, project }: SolverViewProps) {
  const [form, setForm] = useState<SolverFormState>(FORM_DEFAULTS);
  const [mode, setMode] = useState<LaunchMode>("new");
  const [seedsText, setSeedsText] = useState("0, 1, 2");
  const targets = useRunTargets(project?.datasets ?? null);
  const run = useSolverRun(onRunState, targets.refreshRuns);
  // The launcher resolves the interface family against the primary series, so
  // that is the one whose fields decide which treatments the rail may offer.
  const capability = useSeriesCapability(targets.selected[0] ?? null);

  const patchForm = useCallback(
    (patch: Partial<SolverFormState>) =>
      setForm((prev) => ({ ...prev, ...patch })),
    [],
  );

  useCapabilityGuard(form, patchForm, capability);

  const seeds = useMemo(() => parseSeeds(seedsText), [seedsText]);

  const selectMode = useCallback(
    (next: LaunchMode) => {
      setMode(next);
      targets.setResume(next === "resume");
    },
    [targets],
  );

  const submit = useCallback(() => {
    const { resumeRunId, selected, heldout } = targets;
    if (mode === "sweep") {
      // A sweep varies seeds on one dataset; use the first selected series.
      if (selected.length === 0 || !seeds) return;
      void run.startSweep({
        ...toLaunchRequest(form, { datasets: [selected[0]] }),
        seeds,
      });
      return;
    }
    if (mode === "resume") {
      if (!resumeRunId) return;
      void run.start(toLaunchRequest(form, { resumeRunId }));
      return;
    }
    if (selected.length === 0) return;
    // One series may be held out of training (axis B) as a transfer test.
    void run.start(
      toLaunchRequest(form, {
        datasets: selected,
        heldout: heldout ? [heldout] : [],
      }),
    );
  }, [run, form, targets, mode, seeds]);

  const rebalanceSteps = useMemo(() => {
    if (run.hist.length === 0) return [];
    const last = run.hist[run.hist.length - 1].step;
    const marks: number[] = [];
    for (let s = form.rebalance_every; s <= last; s += form.rebalance_every)
      marks.push(s);
    return marks;
  }, [run.hist, form.rebalance_every]);

  const dot = statusDot(run.status);
  const latest = run.hist.length > 0 ? run.hist[run.hist.length - 1] : null;
  const noDatasets =
    targets.available !== null && targets.available.length === 0;

  return (
    <>
      <StageAside {...ASIDE}>
        <SolverAside
          form={form}
          onForm={patchForm}
          mode={mode}
          onMode={selectMode}
          targets={targets}
          capability={capability}
          seedsText={seedsText}
          onSeedsText={setSeedsText}
          seeds={seeds}
          running={run.running}
          onReset={run.reset}
          onRun={submit}
        />
      </StageAside>

      <div className="solver-head">
        <StatusDot tone={dot.tone} label={dot.label} />
        {run.status && <span className="id">{run.status.run_id}</span>}
        {run.status?.state === "running" && run.status.stage && (
          <Chip tone="accent">{run.status.stage}</Chip>
        )}
      </div>
      {run.error && <Callout tone="error">{run.error}</Callout>}
      {run.status?.state === "error" && run.status.message && (
        <Callout tone="error" title="Run failed">
          {run.status.message}
        </Callout>
      )}
      {targets.loadError && <Callout tone="error">{targets.loadError}</Callout>}
      {noDatasets && (
        <p className="state-note">
          No preprocessed dataset yet; upload and preprocess one under Datasets
          &amp; conditions to enable the solver.
        </p>
      )}
      {/* Canvas: what the run produces, once the configuration in the rail has
          launched it. */}
      <div className="solver-col">
        {run.sweep && <SweepPanel sweep={run.sweep} />}
        <MonitorPanel
          status={run.status}
          latest={latest}
          // Show a stat once it's configured (so a running run is labelled
          // right) or once its value arrives from the finished run's metrics.
          showVal={form.val_fraction > 0 || run.valIou != null}
          showTransfer={targets.heldout !== "" || run.transferIou != null}
          holdoutIou={run.holdoutIou}
          valIou={run.valIou}
          transferIou={run.transferIou}
        />
        <Panel title="Loss history" subtitle="log₁₀ · rebalance markers">
          {run.hist.length >= 2 ? (
            <ChartFrame
              name={`${run.status?.run_id ?? "run"}-live-loss`}
              title="Live loss history"
              rows={run.hist as unknown as Record<string, unknown>[]}
              render={() => (
                <ViewCanvas>
                  <LossChart
                    records={run.hist}
                    rebalanceSteps={rebalanceSteps}
                  />
                </ViewCanvas>
              )}
            />
          ) : (
            <ViewCanvas>
              <p className="canvas-note">
                Loss history appears once the run logs its first records.
              </p>
            </ViewCanvas>
          )}
        </Panel>
        <Panel title="Solver console" subtitle="pipeline log · live">
          <Console lines={run.lines} label="Solver console" />
        </Panel>
      </div>
    </>
  );
}
