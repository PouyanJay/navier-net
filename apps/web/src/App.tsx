import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Navigate,
  Route,
  Routes,
  useNavigate,
  useParams,
} from "react-router-dom";

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
import { seriesNameOf } from "./lib/series";
import { DatasetsView } from "./views/DatasetsView";
import { PhysicsModelView } from "./views/PhysicsModelView";
import { ProjectsView } from "./views/ProjectsView";
import { ResultsPage } from "./views/results/ResultsPage";
import { SolverView } from "./views/SolverView";

const PAGE_TITLE: Record<string, string> = Object.fromEntries(
  NAV_ITEMS.map((item) => [item.id, item.label]),
);

const PAGE_INTRO: Record<string, string> = {
  projects:
    "Each project scopes its own datasets, physics configuration, runs, and results. Open a project to enter its reconstruction pipeline.",
  datasets:
    "Each uploaded image series carries its own operating conditions; the solver never shares conditions across datasets. Select a series to review its frames and edit its conditions; dimensionless groups recompute live for the selected dataset.",
  physics:
    "The governing equations the network is constrained by, and the live architecture of the field ensemble.",
  solver:
    "Configure the optimization: every value below is an input to the run. The holdout frame is never supervised; its IoU is the live generalization metric. Runs are resumable from the checkpoint.",
  results:
    "Solver runs and their validation against the measured bubble. Every number is read live from the pipeline's own artifacts.",
};

// Each stage's "continue" action, advancing along the pipeline (mockup flow).
const CONTINUE: Record<string, { label: string; next: string }> = {
  datasets: { label: "Continue to physics →", next: "physics" },
  physics: { label: "Continue to solver →", next: "solver" },
  solver: { label: "Continue to results →", next: "results" },
};

const IDLE_STATUS: PlatformStatus = { projects: 0 };

interface RepoFacts {
  datasets: DatasetSummary[];
  runs: RunSummary[];
  projectCount: number;
}

/** Valid stage segments for /projects/:pid/:stage — anything else goes home. */
const PROJECT_STAGES = new Set(["datasets", "physics", "solver", "results"]);

export function App() {
  return (
    <Routes>
      <Route path="/" element={<Workspace />} />
      <Route path="/projects/:pid/:stage" element={<Workspace />} />
      <Route path="/projects/:pid/results/:runId" element={<Workspace />} />
      <Route
        path="/projects/:pid/results/:runId/:tab"
        element={<Workspace />}
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function Workspace() {
  const params = useParams<{ pid?: string; stage?: string; runId?: string }>();
  const navigate = useNavigate();
  // The deep results routes have no :stage param; they are always "results".
  const stage = params.pid ? (params.stage ?? "results") : undefined;
  const active = stage && PROJECT_STAGES.has(stage) ? stage : "projects";
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
      .catch(() => setRepo(null)); // chrome only; views surface real errors
  }, []);

  const status = useMemo<PlatformStatus>(
    () => (repo ? { projects: repo.projectCount } : IDLE_STATUS),
    [repo],
  );

  /**
   * The run the breadcrumb names, read from the URL rather than from "whichever
   * run is newest". Only the deep results routes carry a :runId, so every other
   * stage gets no run segment at all -- which is right, because on Datasets or
   * Solver there is no run in view to name.
   */
  const runCrumb = useMemo(() => {
    if (!params.runId || !repo) return null;
    const run = repo.runs.find((candidate) => candidate.id === params.runId);
    if (!run) return null;
    // Dataset-named legacy runs follow their series' current display label.
    const name =
      run.dataset && run.id === run.dataset
        ? (seriesNameOf(repo.datasets, run.dataset) ?? run.id)
        : run.id;
    return { id: run.id, name };
  }, [params.runId, repo]);

  // Pick up a run already in flight (e.g. after a page reload mid-training).
  useEffect(() => {
    refreshStatus();
    api
      .getActiveRun()
      .then(setActiveRun)
      .catch(() => {}); // the pill is best-effort
  }, [refreshStatus]);

  // The URL is the source of truth for the open project: resolve :pid to a
  // project (deep links, reloads), and clear it when navigating home.
  useEffect(() => {
    const pid = params.pid;
    if (!pid) {
      setProject(null);
      return;
    }
    let mounted = true;
    setProject((current) => (current?.id === pid ? current : null));
    api
      .getProject(pid)
      .then((loaded) => {
        if (mounted) setProject(loaded);
      })
      .catch(() => {
        if (mounted) {
          toast("Project not found", pid, "err");
          navigate("/", { replace: true });
        }
      });
    return () => {
      mounted = false;
    };
  }, [params.pid, navigate, toast]);

  // Stage navigation writes the URL; the URL drives everything else.
  const goToStage = useCallback(
    (id: string) => {
      if (id === "projects" || !params.pid) navigate("/");
      else navigate(`/projects/${params.pid}/${id}`);
    },
    [navigate, params.pid],
  );

  const handleRunState = useCallback(
    (run: RunJobStatus | null) => {
      const previous = previousRun.current;
      if (
        run &&
        previous &&
        previous.run_id === run.run_id &&
        previous.state === "running"
      ) {
        if (run.state === "done") {
          toast("Training complete", run.run_id, "ok");
          refreshStatus();
        }
        if (run.state === "error")
          toast("Run failed", run.message ?? run.run_id, "err");
      }
      previousRun.current = run;
      setActiveRun(run);
    },
    [toast, refreshStatus],
  );

  const openProject = useCallback(
    (selected: ProjectSummary) => {
      setProject(selected); // seed before the :pid effect re-resolves it
      navigate(`/projects/${selected.id}/datasets`);
    },
    [navigate],
  );

  // Stable identity: AppShell memoizes its palette actions on this callback.
  const goHome = useCallback(() => {
    navigate("/");
  }, [navigate]);

  // Series uploads update the project's dataset list; stage flags follow.
  const handleProjectChanged = useCallback(
    (updated: ProjectSummary) => {
      setProject(updated);
      refreshStatus();
    },
    [refreshStatus],
  );

  // A project URL with an unknown stage segment is not a page.
  if (stage && !PROJECT_STAGES.has(stage)) return <Navigate to="/" replace />;

  return (
    <AppShell
      active={active}
      onNavigate={goToStage}
      activeRun={activeRun}
      status={status}
      runCrumb={runCrumb}
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
        {project && CONTINUE[active] && (
          <Button
            variant="primary"
            onClick={() => goToStage(CONTINUE[active].next)}
          >
            {CONTINUE[active].label}
          </Button>
        )}
      </header>
      <div className="stack">
        {active === "results" && project && <ResultsPage project={project} />}
        {active === "projects" && (
          <ProjectsView
            onOpen={openProject}
            creating={creatingProject}
            onCreatingChange={setCreatingProject}
            onChanged={refreshStatus}
          />
        )}
        {active === "datasets" && project && (
          <DatasetsView
            project={project}
            onProjectChanged={handleProjectChanged}
          />
        )}
        {active === "physics" && (
          <PhysicsModelView
            datasets={
              repo
                ? project
                  ? repo.datasets.filter((d) => project.datasets.includes(d.id))
                  : repo.datasets
                : []
            }
          />
        )}
        {active === "solver" && (
          <SolverView onRunState={handleRunState} project={project} />
        )}
      </div>
    </AppShell>
  );
}
