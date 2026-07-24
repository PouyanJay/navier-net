import { useCallback, useEffect, useRef, useState } from "react";

import { AppShell, NAV_ITEMS, type PlatformStatus } from "./app/AppShell";
import { useToast } from "./components/Toast";
import { api, type RunJobStatus } from "./lib/api";
import { DatasetsView } from "./views/DatasetsView";
import { PhysicsModelView } from "./views/PhysicsModelView";
import { ProjectsView } from "./views/ProjectsView";
import { ResultsView } from "./views/ResultsView";
import { SolverView } from "./views/SolverView";

const PAGE_TITLE: Record<string, string> = Object.fromEntries(
  NAV_ITEMS.map((item) => [item.id, item.label]),
);

const PAGE_INTRO: Record<string, string> = {
  projects:
    "Each imaged dataset is a project. Open one to inspect its conditions, upload frames, and preprocess.",
  datasets:
    "Operating conditions, derived dimensionless groups, the raw image sequence, and calibration/segmentation.",
  physics:
    "The governing equations the network is constrained by, and the live architecture of the field ensemble.",
  solver:
    "Configure the optimization — every value below is an input to the run. The holdout frame is never supervised; its IoU is the live generalization metric. Runs are resumable from the checkpoint.",
  results:
    "Solver runs and their validation against the measured bubble. Every number is read live from the pipeline's own artifacts.",
};

const IDLE_STATUS: PlatformStatus = { done: { physics: true }, latestRun: null };

export function App() {
  const [active, setActive] = useState("results");
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [activeRun, setActiveRun] = useState<RunJobStatus | null>(null);
  const [status, setStatus] = useState<PlatformStatus>(IDLE_STATUS);
  const previousRun = useRef<RunJobStatus | null>(null);
  const toast = useToast();

  const refreshStatus = useCallback(() => {
    Promise.all([api.listDatasets(), api.listRuns()])
      .then(([datasets, runs]) => {
        const trained = runs.filter((run) => run.status === "trained");
        const evaluated = runs.filter((run) => run.iou_holdout != null);
        const latest = trained[trained.length - 1] ?? null;
        setStatus({
          done: {
            datasets: datasets.some((dataset) => dataset.processed),
            physics: true, // the governing equations ship with the platform
            solver: trained.length > 0,
            results: evaluated.length > 0,
          },
          latestRun: latest ? { id: latest.id, steps: latest.steps } : null,
        });
      })
      .catch(() => setStatus(IDLE_STATUS)); // chrome only — views surface real errors
  }, []);

  // Pick up a run already in flight (e.g. after a page reload mid-training).
  useEffect(() => {
    refreshStatus();
    api
      .getActiveRun()
      .then(setActiveRun)
      .catch(() => {}); // the pill is best-effort
  }, [refreshStatus]);

  const handleRunState = useCallback(
    (run: RunJobStatus | null) => {
      const previous = previousRun.current;
      if (run && previous && previous.run_id === run.run_id && previous.state === "running") {
        if (run.state === "done") {
          toast("Training complete", run.run_id, "ok");
          refreshStatus();
        }
        if (run.state === "error") toast("Run failed", run.message ?? run.run_id, "err");
      }
      previousRun.current = run;
      setActiveRun(run);
    },
    [toast, refreshStatus],
  );

  function openDataset(id: string) {
    setDatasetId(id);
    setActive("datasets");
  }

  return (
    <AppShell active={active} onNavigate={setActive} activeRun={activeRun} status={status}>
      <header className="pagehead">
        <div>
          <h1>{PAGE_TITLE[active]}</h1>
          {PAGE_INTRO[active] && <p>{PAGE_INTRO[active]}</p>}
        </div>
      </header>
      <div className="stack">
        {active === "results" && <ResultsView />}
        {active === "projects" && <ProjectsView onOpen={openDataset} />}
        {active === "datasets" && <DatasetsView datasetId={datasetId} />}
        {active === "physics" && <PhysicsModelView />}
        {active === "solver" && <SolverView onRunState={handleRunState} />}
      </div>
    </AppShell>
  );
}
