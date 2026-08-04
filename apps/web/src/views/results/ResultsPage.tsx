import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { Callout, ConfirmDeleteDialog, Panel } from "../../components";
import { ErrorBoundary } from "../../components/ErrorBoundary";
import { useToast } from "../../components/Toast";
import {
  api,
  type DatasetSummary,
  type ProjectSummary,
  type RunStatus,
  type RunSummary,
} from "../../lib/api";
import { isTrainedRun } from "../../lib/runs";
import {
  runConditions,
  runDisplayName,
  runHeadline,
  runRowMeta,
} from "./format";
import { AgreementTab } from "./AgreementTab";
import { CompareTab } from "./CompareTab";
import { ExportTab } from "./ExportTab";
import { FieldsTab } from "./FieldsTab";
import { FrontVelocityTab } from "./FrontVelocityTab";
import { OverviewTab } from "./OverviewTab";
import { PhysicsTab } from "./PhysicsTab";
import { TrainingTab } from "./TrainingTab";
import { ReconTab } from "./ReconTab";
import { RunHeader } from "./RunHeader";
import { useRunDetail } from "./useRunDetail";
import { useValidation } from "./useValidation";
import "./results.css";

/** How many further optimisation steps a header "Resume training" asks for. */
const RESUME_EXTRA_STEPS = 1500;
const MAX_STEPS = 20_000;

const DOT_TONE: Record<RunStatus, string> = {
  running: "live",
  trained: "ok",
  failed: "err",
  empty: "",
};

/** Arrow keys walk the run list, matching the command palette's listbox. */
function moveRunFocus(event: KeyboardEvent<HTMLDivElement>) {
  if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
  const options = [
    ...event.currentTarget.querySelectorAll<HTMLButtonElement>(
      "[role='option']",
    ),
  ];
  const index = options.indexOf(document.activeElement as HTMLButtonElement);
  if (index === -1) return;
  event.preventDefault();
  const next =
    options[
      (index + (event.key === "ArrowDown" ? 1 : options.length - 1)) %
        options.length
    ];
  next.focus();
}

/** The output tabs, in the mockup's reading order. */
export const RESULT_TABS = [
  { id: "overview", label: "Overview" },
  { id: "recon", label: "Reconstruction" },
  { id: "fields", label: "Fields" },
  { id: "agreement", label: "Agreement & transfer" },
  { id: "physics", label: "Physics" },
  { id: "velocity", label: "Front velocity" },
  { id: "training", label: "Training" },
  { id: "compare", label: "Compare" },
  { id: "export", label: "Artifacts & export" },
] as const;

export type ResultTabId = (typeof RESULT_TABS)[number]["id"];

const TAB_IDS = new Set<string>(RESULT_TABS.map((tab) => tab.id));

interface ResultsPageProps {
  project: ProjectSummary;
}

/**
 * Results & validation: the project's runs on the left, the selected run's
 * outputs in tabs on the right. Run and tab selection live in the URL
 * (/projects/:pid/results/:runId/:tab) so any result is deep-linkable.
 */
export function ResultsPage({ project }: ResultsPageProps) {
  const params = useParams<{ runId?: string; tab?: string }>();
  const navigate = useNavigate();
  const toast = useToast();
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [resuming, setResuming] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  useEffect(() => {
    let mounted = true;
    setRuns(null);
    setError(null);
    Promise.all([api.listRuns(project.id), api.listDatasets()])
      .then(([listed, allDatasets]) => {
        if (!mounted) return;
        setRuns(listed);
        setDatasets(allDatasets);
      })
      .catch((exc: unknown) => {
        if (mounted) setError(exc instanceof Error ? exc.message : String(exc));
      });
    return () => {
      mounted = false;
    };
  }, [project.id]);

  const datasetLabels = useMemo(
    () =>
      new Map(
        datasets.map((dataset) => [dataset.id, dataset.label ?? dataset.id]),
      ),
    [datasets],
  );

  const base = `/projects/${project.id}/results`;

  // URL selection, falling back to the first trained run (else first run).
  const selected = useMemo(() => {
    if (!runs || runs.length === 0) return null;
    const byUrl = params.runId
      ? runs.find((run) => run.id === params.runId)
      : undefined;
    return byUrl ?? runs.find(isTrainedRun) ?? runs[0];
  }, [runs, params.runId]);

  const activeTab: ResultTabId =
    params.tab && TAB_IDS.has(params.tab)
      ? (params.tab as ResultTabId)
      : "overview";

  const { detail } = useRunDetail(selected?.id ?? null);
  const { validation } = useValidation(selected?.id ?? null);

  // Per-condition panels (viewport, kinematics, fields…) view one condition of
  // a joint run at a time; default to the run's first dataset.
  const [viewDataset, setViewDataset] = useState<string | null>(null);
  const selectedId = selected?.id ?? null;
  useEffect(() => {
    // Keyed on the id, not the object: a background list refresh must not
    // reset the researcher's chosen condition.
    setViewDataset(selected ? (runConditions(selected).all[0] ?? null) : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId]);

  const openRun = (runId: string) =>
    navigate(`${base}/${encodeURIComponent(runId)}`);
  const openTab = (tab: ResultTabId) => {
    if (selected) navigate(`${base}/${encodeURIComponent(selected.id)}/${tab}`);
  };

  const resumeSelected = () => {
    if (!selected) return;
    setResuming(true);
    const target = Math.min(
      MAX_STEPS,
      (detail?.steps ?? selected.steps ?? 0) + RESUME_EXTRA_STEPS,
    );
    api
      .startRun({ resume: true, run_id: selected.id, steps: target })
      .then(() => {
        toast("Resume queued", `${selected.id} → ${target} steps`, "ok");
        return api.listRuns(project.id).then(setRuns);
      })
      .catch((exc: unknown) =>
        toast(
          "Resume rejected",
          exc instanceof Error ? exc.message : String(exc),
          "err",
        ),
      )
      .finally(() => setResuming(false));
  };

  if (error)
    return (
      <Callout tone="error" title="Runs unavailable">
        {error}
      </Callout>
    );

  return (
    <div className="resx">
      <div className="res-left">
        <Panel title="Runs" subtitle="this project · outputs/">
          {runs === null ? (
            <p className="res-quiet">Loading runs…</p>
          ) : runs.length === 0 ? (
            <div className="res-empty">
              <b>No runs yet</b>
              Launch the first training run from the Solver stage.
            </div>
          ) : (
            <div
              className="runlist"
              role="listbox"
              aria-label="Runs of this project"
              onKeyDown={moveRunFocus}
            >
              {runs.map((run) => {
                const headline = runHeadline(run);
                return (
                  <button
                    key={run.id}
                    type="button"
                    role="option"
                    aria-selected={run.id === selected?.id}
                    className={
                      "runrow" + (run.id === selected?.id ? " sel" : "")
                    }
                    onClick={() => openRun(run.id)}
                  >
                    <span
                      className={`runrow-dot ${DOT_TONE[run.status]}`}
                      aria-hidden="true"
                    />
                    <span className="runrow-main">
                      <b>{runDisplayName(run, datasetLabels)}</b>
                      <span>{runRowMeta(run)}</span>
                    </span>
                    {headline && (
                      <span className="runrow-headline">
                        {headline.value}
                        <small>{headline.label}</small>
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </Panel>
      </div>

      <div className="res-main">
        {selected && (
          <RunHeader
            run={selected}
            detail={detail}
            datasetLabels={datasetLabels}
            viewDataset={viewDataset}
            onViewDataset={setViewDataset}
            onResume={resumeSelected}
            resuming={resuming}
            onDelete={() => setConfirmingDelete(true)}
          />
        )}
        {confirmingDelete && selected && (
          <ConfirmDeleteDialog
            title="Delete run"
            confirmLabel="Delete run"
            onClose={() => setConfirmingDelete(false)}
            onConfirm={async () => {
              const name = runDisplayName(selected, datasetLabels);
              await api.deleteRun(selected.id);
              toast("Run deleted", name, "ok");
              setRuns(await api.listRuns(project.id));
              navigate(base); // drop the deleted run from the URL; selection falls back
              setConfirmingDelete(false);
            }}
          >
            Delete <b>{runDisplayName(selected, datasetLabels)}</b> and
            everything under <code>outputs/{selected.id}</code> — checkpoint,
            figures, video and metrics.
          </ConfirmDeleteDialog>
        )}
        <div className="tabbar" role="tablist" aria-label="Run outputs">
          {RESULT_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              id={`tab-${tab.id}`}
              aria-selected={tab.id === activeTab}
              aria-controls={`panel-${tab.id}`}
              tabIndex={tab.id === activeTab ? 0 : -1}
              className="tab"
              disabled={!selected}
              onClick={() => openTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <section
          className="stack"
          role="tabpanel"
          id={`panel-${activeTab}`}
          aria-labelledby={`tab-${activeTab}`}
        >
          <ErrorBoundary
            label={`The ${RESULT_TABS.find((tab) => tab.id === activeTab)!.label} tab`}
            resetKey={`${selected?.id ?? ""}:${activeTab}`}
          >
            {selected ? (
              activeTab === "overview" ? (
                <OverviewTab
                  run={selected}
                  detail={detail}
                  validation={validation}
                  datasetLabels={datasetLabels}
                  onOpenTab={openTab}
                />
              ) : activeTab === "recon" ? (
                <ReconTab
                  run={selected}
                  detail={detail}
                  validation={validation}
                  viewDataset={viewDataset}
                  datasetLabels={datasetLabels}
                />
              ) : activeTab === "fields" ? (
                <FieldsTab
                  runId={selected.id}
                  dataset={viewDataset}
                  datasetName={
                    viewDataset
                      ? (datasetLabels.get(viewDataset) ?? viewDataset)
                      : null
                  }
                  joint={runConditions(selected).all.length > 1}
                />
              ) : activeTab === "agreement" ? (
                <AgreementTab
                  run={selected}
                  metrics={detail?.metrics ?? null}
                  validation={validation}
                  datasetLabels={datasetLabels}
                />
              ) : activeTab === "physics" ? (
                <PhysicsTab
                  runId={selected.id}
                  frontGeometry={Boolean(
                    (detail?.config as { model?: { front_geometry?: boolean } })
                      ?.model?.front_geometry,
                  )}
                  dataset={viewDataset}
                  datasetName={
                    viewDataset
                      ? (datasetLabels.get(viewDataset) ?? viewDataset)
                      : null
                  }
                  validation={validation}
                />
              ) : activeTab === "velocity" ? (
                <FrontVelocityTab runId={selected.id} dataset={viewDataset} />
              ) : activeTab === "training" ? (
                <TrainingTab runId={selected.id} />
              ) : activeTab === "compare" ? (
                <CompareTab
                  runs={runs ?? []}
                  currentId={selected.id}
                  datasetLabels={datasetLabels}
                />
              ) : (
                <ExportTab
                  run={selected}
                  detail={detail}
                  viewDataset={viewDataset}
                  datasetLabels={datasetLabels}
                />
              )
            ) : runs !== null ? (
              <div className="res-empty">
                <b>Nothing to show</b>
                Results appear after the project's first run.
              </div>
            ) : null}
          </ErrorBoundary>
        </section>
      </div>
    </div>
  );
}
