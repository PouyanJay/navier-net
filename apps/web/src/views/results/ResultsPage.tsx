import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { Callout, Panel } from "../../components";
import {
  api,
  type ProjectSummary,
  type RunStatus,
  type RunSummary,
} from "../../lib/api";
import { isTrainedRun } from "../../lib/runs";
import { runHeadline, runRowMeta } from "./format";
import "./results.css";

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
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setRuns(null);
    setError(null);
    api
      .listRuns(project.id)
      .then((listed) => {
        if (mounted) setRuns(listed);
      })
      .catch((exc: unknown) => {
        if (mounted) setError(exc instanceof Error ? exc.message : String(exc));
      });
    return () => {
      mounted = false;
    };
  }, [project.id]);

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

  const openRun = (runId: string) =>
    navigate(`${base}/${encodeURIComponent(runId)}`);
  const openTab = (tab: ResultTabId) => {
    if (selected) navigate(`${base}/${encodeURIComponent(selected.id)}/${tab}`);
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
                      <b>{run.id}</b>
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
          {selected ? (
            <Panel
              title={RESULT_TABS.find((tab) => tab.id === activeTab)!.label}
              subtitle={selected.id}
            >
              <p className="res-quiet">
                {/* Walking skeleton: panels land tab by tab in later tasks. */}
                Content for “{activeTab}” arrives in a later task.
              </p>
            </Panel>
          ) : runs !== null ? (
            <div className="res-empty">
              <b>Nothing to show</b>
              Results appear after the project's first run.
            </div>
          ) : null}
        </section>
      </div>
    </div>
  );
}
