import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";

import { StageAside } from "../../app/StageAside";
import { Band, Callout, ConfirmDeleteDialog } from "../../components";
import { StageResultsIcon } from "../../components/icons";
import { ErrorBoundary } from "../../components/ErrorBoundary";
import { useToast } from "../../components/Toast";
import {
  api,
  type DatasetSummary,
  type ProjectSummary,
  type RunSummary,
} from "../../lib/api";
import { isTrainedRun, MAX_COMPARED } from "../../lib/runs";
import { runConditions, runDisplayName } from "./format";
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
import { buildFamilies } from "./runFamilies";
import { runCapability } from "./runCapability";
import { tabBadges } from "./tabBadges";
import { RunRail } from "./RunRail";
import { useRunDetail } from "./useRunDetail";
import { useValidation } from "./useValidation";
import "./results.css";

/** The run rows carry a name, its recipe and a ranked metric, so this stage's
 * aside takes a little more than the shell's default. */
const ASIDE = {
  title: "Results & validation",
  subtitle: "runs · outputs",
  width: 403,
};

/** How many further optimisation steps a header "Resume training" asks for. */
const RESUME_EXTRA_STEPS = 1500;
const MAX_STEPS = 20_000;

/** The output tabs, in the mockup's reading order. */
export const RESULT_TABS = [
  { id: "overview", label: "Overview" },
  { id: "recon", label: "Reconstruction" },
  { id: "fields", label: "Fields" },
  { id: "agreement", label: "Agreement" },
  { id: "physics", label: "Physics" },
  { id: "velocity", label: "Front velocity" },
  { id: "training", label: "Training" },
  { id: "compare", label: "Compare" },
  { id: "export", label: "Artifacts" },
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
  const [search] = useSearchParams();
  // Runs the rail picked alongside the one in the path, for the Compare tab.
  const comparedWith = useMemo(
    () =>
      (search.get("with") ?? "")
        .split(",")
        .filter(Boolean)
        .slice(0, MAX_COMPARED - 1),
    [search],
  );

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
  // One answer to "what can this run still show", shared by every tab that
  // would otherwise ask the checkpoint a question it cannot answer.
  const capability = useMemo(
    () => (selected ? runCapability(selected, detail) : null),
    [selected, detail],
  );
  const { validation, loading: validationLoading } = useValidation(
    selected?.id ?? null,
  );
  // Where this run stands in the rail's ranking, carried into the header so the
  // standing you chose it by is still on screen while you read it.
  const standing = useMemo(() => {
    if (!runs || !selected) return null;
    const families = buildFamilies(runs, { metric: "val" });
    const family = families.find((entry) =>
      entry.runs.some((run) => run.id === selected.id),
    );
    if (!family?.rank) return null;
    const ranked = families.filter((entry) => entry.rank != null).length;
    return family.behind == null
      ? `rank ${family.rank} of ${ranked} · best val IoU`
      : `rank ${family.rank} of ${ranked} · −${family.behind.toFixed(3)}`;
  }, [runs, selected]);

  const badges = useMemo(
    () =>
      selected && capability
        ? tabBadges(
            selected,
            capability,
            validation,
            (runs ?? []).filter(isTrainedRun).length,
            datasetLabels.get(selected.dataset ?? "") ?? selected.dataset ?? "",
          )
        : {},
    [selected, capability, validation, runs, datasetLabels],
  );

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
  /** Open the Compare tab on an explicit set of runs. The picks travel in the
   * URL like every other selection on this stage, so a comparison is a link. */
  const openCompare = (ids: string[]) => {
    const [first, ...rest] = ids;
    const query = rest.length
      ? `?with=${rest.map(encodeURIComponent).join(",")}`
      : "";
    navigate(`${base}/${encodeURIComponent(first)}/compare${query}`);
  };
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

  /** Rename the selected run. The list is refetched rather than patched locally
   * so the rail, the compare picker and the header all read one answer — the
   * server's. Errors propagate to the header, which owns the editor's state. */
  const renameSelected = async (label: string) => {
    if (!selected) return;
    const renamed = await api.setRunLabel(selected.id, label);
    setRuns(await api.listRuns(project.id));
    toast("Run renamed", renamed.label ?? selected.id, "ok");
  };

  if (error)
    return (
      <Callout tone="error" title="Runs unavailable">
        {error}
      </Callout>
    );

  return (
    <>
      <StageAside {...ASIDE}>
        {runs === null ? (
          <Band icon={StageResultsIcon} label="Runs" hint="outputs/">
            <p className="res-quiet">Loading runs…</p>
          </Band>
        ) : runs.length === 0 ? (
          <Band icon={StageResultsIcon} label="Runs" hint="outputs/">
            <div className="res-empty">
              <b>No runs yet</b>
              Launch the first training run from the Solver stage.
            </div>
          </Band>
        ) : (
          <RunRail
            runs={runs}
            datasetLabels={datasetLabels}
            selectedId={selected?.id ?? null}
            onOpen={openRun}
            onCompare={openCompare}
          />
        )}
      </StageAside>

      <div className="res-main">
        {selected && (
          <RunHeader
            // Remounting on selection drops any half-finished rename with it.
            key={selected.id}
            run={selected}
            detail={detail}
            validationFrames={validation?.validation_frames ?? null}
            standing={standing}
            datasetLabels={datasetLabels}
            viewDataset={viewDataset}
            onViewDataset={setViewDataset}
            onResume={resumeSelected}
            resuming={resuming}
            onRename={renameSelected}
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
            everything under <code>outputs/{selected.id}</code>: checkpoint,
            figures, video and metrics.
          </ConfirmDeleteDialog>
        )}
        <div className="tabbar" role="tablist" aria-label="Run outputs">
          {RESULT_TABS.map((tab) => {
            const badge = badges[tab.id];
            return (
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
                title={badge?.title}
                onClick={() => openTab(tab.id)}
              >
                {tab.label}
                {/* What the tab holds, so finding out costs no click. */}
                {badge && (
                  <span className={badge.warn ? "tab-n warn" : "tab-n"}>
                    {badge.text}
                  </span>
                )}
              </button>
            );
          })}
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
                  validationLoading={validationLoading}
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
                  capability={capability!}
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
                  capability={capability!}
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
                  run={selected}
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
                  validationLoading={validationLoading}
                />
              ) : activeTab === "velocity" ? (
                <FrontVelocityTab
                  runId={selected.id}
                  dataset={viewDataset}
                  capability={capability!}
                  noseSpeed={validation?.nose_speed_inferred_mm_s ?? null}
                />
              ) : activeTab === "training" ? (
                <TrainingTab runId={selected.id} />
              ) : activeTab === "compare" ? (
                <CompareTab
                  runs={runs ?? []}
                  currentId={selected.id}
                  alsoCompare={comparedWith}
                  datasetLabels={datasetLabels}
                />
              ) : (
                <ExportTab
                  run={selected}
                  capability={capability!}
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
    </>
  );
}
