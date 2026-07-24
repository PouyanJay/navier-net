import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AppShell, NAV_ITEMS, type PlatformStatus } from "./app/AppShell";
import { Button } from "./components";
import { useToast } from "./components/Toast";
import {
  api,
  type DatasetSummary,
  type ProjectSummary,
  type RunJobStatus,
  type RunSummary,
} from "./lib/api";
import { DatasetsView } from "./views/DatasetsView";
import { EmptyProjectUpload } from "./views/datasets/EmptyProjectUpload";
import { PhysicsModelView } from "./views/PhysicsModelView";
import { ProjectsView } from "./views/ProjectsView";
import { ResultsView } from "./views/ResultsView";
import { SolverView } from "./views/SolverView";

const PAGE_TITLE: Record<string, string> = Object.fromEntries(
  NAV_ITEMS.map((item) => [item.id, item.label]),
);

const PAGE_INTRO: Record<string, string> = {
  projects:
    "Each project scopes its own datasets, physics configuration, runs, and results. Open a project to enter its reconstruction pipeline.",
  datasets:
    "Operating conditions, derived dimensionless groups, the raw image sequence, and calibration/segmentation.",
  physics:
    "The governing equations the network is constrained by, and the live architecture of the field ensemble.",
  solver:
    "Configure the optimization — every value below is an input to the run. The holdout frame is never supervised; its IoU is the live generalization metric. Runs are resumable from the checkpoint.",
  results:
    "Solver runs and their validation against the measured bubble. Every number is read live from the pipeline's own artifacts.",
};

const IDLE_STATUS: PlatformStatus = { done: { physics: true }, latestRun: null, projects: 0 };

interface RepoFacts {
  datasets: DatasetSummary[];
  runs: RunSummary[];
  projectCount: number;
}

export function App() {
  const [active, setActive] = useState("projects");
  const [project, setProject] = useState<ProjectSummary | null>(null);
  const [creatingProject, setCreatingProject] = useState(false);
  const [activeRun, setActiveRun] = useState<RunJobStatus | null>(null);
  const [repo, setRepo] = useState<RepoFacts | null>(null);
  const previousRun = useRef<RunJobStatus | null>(null);
  const toast = useToast();

  const refreshStatus = useCallback(() => {
    Promise.all([api.listDatasets(), api.listRuns(), api.listProjects()])
      .then(([datasets, runs, projects]) =>
        setRepo({ datasets, runs, projectCount: projects.length }),
      )
      .catch(() => setRepo(null)); // chrome only — views surface real errors
  }, []);

  // Stage flags are scoped to the open project: an empty project shows an
  // untouched pipeline even when other projects have trained runs.
  const status = useMemo<PlatformStatus>(() => {
    if (!repo) return IDLE_STATUS;
    const datasets = project
      ? repo.datasets.filter((dataset) => dataset.id === project.dataset)
      : repo.datasets;
    const runs = project ? repo.runs.filter((run) => run.dataset === project.dataset) : repo.runs;
    const trained = runs.filter((run) => run.status === "trained");
    const latest = trained[trained.length - 1] ?? null;
    return {
      done: {
        datasets: datasets.some((dataset) => dataset.processed),
        physics: true, // the governing equations ship with the platform
        solver: trained.length > 0,
        results: runs.some((run) => run.iou_holdout != null),
      },
      latestRun: latest ? { id: latest.id, steps: latest.steps } : null,
      projects: repo.projectCount,
    };
  }, [repo, project]);

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

  const openProject = useCallback((selected: ProjectSummary) => {
    setProject(selected);
    setActive("datasets");
  }, []);

  // Stable identity: AppShell memoizes its palette actions on this callback.
  const goHome = useCallback(() => {
    setProject(null);
    setActive("projects");
  }, []);

  // The first upload attaches the dataset; the pipeline unlocks from there.
  const attachDataset = useCallback(
    (updated: ProjectSummary) => {
      setProject(updated);
      refreshStatus();
    },
    [refreshStatus],
  );

  return (
    <AppShell
      active={active}
      onNavigate={setActive}
      activeRun={activeRun}
      status={status}
      project={project?.name ?? null}
      onHome={goHome}
    >
      <header className="pagehead">
        <div>
          <h1>{PAGE_TITLE[active]}</h1>
          {PAGE_INTRO[active] && <p>{PAGE_INTRO[active]}</p>}
        </div>
        {active === "projects" && (
          <Button variant="primary" onClick={() => setCreatingProject(true)}>
            ＋ New project
          </Button>
        )}
      </header>
      <div className="stack">
        {active === "results" && <ResultsView />}
        {active === "projects" && (
          <ProjectsView
            onOpen={openProject}
            creating={creatingProject}
            onCreatingChange={setCreatingProject}
            onChanged={refreshStatus}
          />
        )}
        {active === "datasets" &&
          (project && !project.dataset ? (
            <EmptyProjectUpload project={project} onAttached={attachDataset} />
          ) : (
            <DatasetsView datasetId={project?.dataset} />
          ))}
        {active === "physics" && <PhysicsModelView />}
        {active === "solver" && <SolverView onRunState={handleRunState} />}
      </div>
    </AppShell>
  );
}
