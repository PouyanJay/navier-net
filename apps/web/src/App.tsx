import { useEffect, useState } from "react";

import { AppShell, NAV_ITEMS } from "./app/AppShell";
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

export function App() {
  const [active, setActive] = useState("results");
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [activeRun, setActiveRun] = useState<RunJobStatus | null>(null);

  // Pick up a run already in flight (e.g. after a page reload mid-training).
  useEffect(() => {
    api
      .getActiveRun()
      .then(setActiveRun)
      .catch(() => {}); // the pill is best-effort
  }, []);

  function openDataset(id: string) {
    setDatasetId(id);
    setActive("datasets");
  }

  return (
    <AppShell active={active} onNavigate={setActive}>
      <header className="pagehead">
        <div>
          <h1>{PAGE_TITLE[active]}</h1>
          {PAGE_INTRO[active] && <p>{PAGE_INTRO[active]}</p>}
        </div>
        {activeRun?.state === "running" && (
          <button type="button" className="runpill" onClick={() => setActive("solver")}>
            <span className="pdot" aria-hidden="true" />
            Training · <span className="mono">{activeRun.run_id}</span>
          </button>
        )}
      </header>
      <div className="stack">
        {active === "results" && <ResultsView />}
        {active === "projects" && <ProjectsView onOpen={openDataset} />}
        {active === "datasets" && <DatasetsView datasetId={datasetId} />}
        {active === "physics" && <PhysicsModelView />}
        {active === "solver" && <SolverView onRunState={setActiveRun} />}
      </div>
    </AppShell>
  );
}
