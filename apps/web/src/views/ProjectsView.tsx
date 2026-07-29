import { seriesName } from "../lib/series";
import { Fragment, useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { Button, Callout, Chip } from "../components";
import { useToast } from "../components/Toast";
import {
  api,
  type DatasetSummary,
  type ProjectSummary,
  type RunSummary,
} from "../lib/api";
import { errorMessage } from "../lib/errors";
import { hasEvaluation, isTrainedRun } from "../lib/runs";
import "./datasets/datasets.css";
import "./runs.css";

interface ProjectFacts {
  project: ProjectSummary;
  datasets: DatasetSummary[];
  runs: RunSummary[];
}

/** Pipeline progress for one project: which stages have real artifacts. */
function stageDots(facts: ProjectFacts): boolean[] {
  if (facts.datasets.length === 0) return [false, false, false, false];
  const dataReady = facts.datasets.some((dataset) => dataset.processed);
  // The governing equations ship with the platform, so the Model stage is
  // complete as soon as the data it binds to is (dot 2 mirrors dot 1).
  return [
    dataReady,
    dataReady,
    facts.runs.some(isTrainedRun),
    facts.runs.some(hasEvaluation),
  ];
}

interface ProjectsViewProps {
  onOpen: (project: ProjectSummary) => void;
  /** The "+ New project" form is controlled so the page header can open it. */
  creating: boolean;
  onCreatingChange: (open: boolean) => void;
  /** Fired after any create/edit, so the shell's counts stay fresh. */
  onChanged?: () => void;
}

/** The workspace home: every project, editable in place, plus creation. */
export function ProjectsView({
  onOpen,
  creating,
  onCreatingChange,
  onChanged,
}: ProjectsViewProps) {
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.listProjects(), api.listDatasets(), api.listRuns()])
      .then(([ps, ds, rs]) => {
        setProjects(ps);
        setDatasets(ds);
        setRuns(rs);
      })
      .catch((err) => setError(errorMessage(err)));
  }, []);

  if (error) {
    return (
      <Callout tone="error" title="Could not load projects">
        {error}. Is the API running on :8000?
      </Callout>
    );
  }
  if (projects === null) {
    return (
      <p className="state-note" role="status">
        Loading projects…
      </p>
    );
  }

  const replaceProject = (updated: ProjectSummary) => {
    setProjects((current) =>
      (current ?? []).map((p) => (p.id === updated.id ? updated : p)),
    );
    onChanged?.();
  };

  const removeProject = (id: string) => {
    setProjects((current) => (current ?? []).filter((p) => p.id !== id));
    onChanged?.();
  };

  return (
    <div className="project-grid">
      {projects.map((project) => (
        <ProjectCard
          key={project.id}
          facts={{
            project,
            datasets: datasets.filter((d) => project.datasets.includes(d.id)),
            runs: runs.filter(
              (run) =>
                run.dataset != null && project.datasets.includes(run.dataset),
            ),
          }}
          onOpen={onOpen}
          onSaved={replaceProject}
          onDeleted={removeProject}
        />
      ))}
      {creating ? (
        <NewProjectForm
          onCreated={(project) => {
            setProjects((current) => [...(current ?? []), project]);
            onCreatingChange(false);
            onChanged?.();
          }}
          onCancel={() => onCreatingChange(false)}
        />
      ) : (
        <button
          type="button"
          className="newproj"
          onClick={() => onCreatingChange(true)}
        >
          <span className="plus" aria-hidden="true">
            ＋
          </span>
          New project
        </button>
      )}
    </div>
  );
}

// Short stage names for the progress label, matching the mockup's pipe copy.
const STAGE_NAMES = ["Data", "Model", "Solve", "Results"];

function statusChip(facts: ProjectFacts, trained: boolean) {
  if (facts.datasets.length === 0) return <Chip>Awaiting data</Chip>;
  if (trained) return <Chip tone="green">Stage A · trained</Chip>;
  if (!facts.datasets.some((dataset) => dataset.processed)) {
    return <Chip tone="amber">needs preprocess</Chip>;
  }
  return <Chip>No runs yet</Chip>;
}

/** Progress dots with connectors, labelled by the stage currently in progress. */
function StagePipe({ dots }: { dots: boolean[] }) {
  const doneCount = dots.filter(Boolean).length;
  const stageLabel =
    doneCount === 0
      ? "not started"
      : `${STAGE_NAMES[Math.min(doneCount, STAGE_NAMES.length - 1)]} stage`;
  return (
    <div className="pipe" aria-label={`Pipeline progress: ${stageLabel}`}>
      {dots.map((done, i) => (
        <Fragment key={STAGE_NAMES[i]}>
          <span
            className="pd"
            data-done={done || undefined}
            data-act={(!done && i === doneCount) || undefined}
          />
          {i < dots.length - 1 && <span className="pl" aria-hidden="true" />}
        </Fragment>
      ))}
      <span className="plbl">{stageLabel}</span>
    </div>
  );
}

function ProjectCard({
  facts,
  onOpen,
  onSaved,
  onDeleted,
}: {
  facts: ProjectFacts;
  onOpen: (project: ProjectSummary) => void;
  onSaved: (project: ProjectSummary) => void;
  onDeleted: (id: string) => void;
}) {
  const { project } = facts;
  const toast = useToast();
  const [editing, setEditing] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const dots = stageDots(facts);

  if (editing) {
    return (
      <div className="pcard">
        <ProjectMetadataForm
          heading={`Edit ${project.name}`}
          initialName={project.name}
          initialDescription={project.description}
          submitLabel="Save"
          onSubmit={async (name, description) => {
            const updated = await api.updateProject(project.id, {
              name,
              description,
            });
            onSaved(updated);
            setEditing(false);
          }}
          onCancel={() => setEditing(false)}
        />
      </div>
    );
  }

  return (
    // The whole card is a mouse shortcut for its Open button; keyboard and
    // assistive tech use the real buttons inside.
    <div className="pcard pcard-clickable" onClick={() => onOpen(project)}>
      <div className="pcard-top">
        <h3>{project.name}</h3>
        {statusChip(facts, dots[2])}
      </div>
      <p className="purpose">{project.description || "No description yet."}</p>
      <div className="pmeta mono">
        <span>
          <b>{facts.datasets.length}</b> dataset
          {facts.datasets.length === 1 ? "" : "s"}
        </span>
        <span>
          <b>{facts.runs.length}</b> run{facts.runs.length === 1 ? "" : "s"}
        </span>
      </div>
      <StagePipe dots={dots} />
      <div className="pfoot">
        <span className="mono">
          naviernet://
          {facts.datasets[0]
            ? seriesName(facts.datasets[0])
            : project.id.slice(0, 8)}
        </span>
        <span className="pfoot-actions">
          <button
            type="button"
            className="btn ghost danger icon-only"
            aria-label={`Delete ${project.name}`}
            title="Delete project"
            onClick={(e) => {
              e.stopPropagation();
              setConfirmingDelete(true);
            }}
          >
            <TrashIcon />
          </button>
          <button
            type="button"
            className="btn ghost"
            onClick={(e) => {
              e.stopPropagation();
              setEditing(true);
            }}
          >
            Edit
          </button>
          <button
            type="button"
            className="btn ghost"
            onClick={(e) => {
              e.stopPropagation();
              onOpen(project);
            }}
          >
            Open →
          </button>
        </span>
      </div>
      {confirmingDelete && (
        <DeleteProjectDialog
          project={project}
          datasetCount={facts.datasets.length}
          runCount={facts.runs.length}
          onClose={() => setConfirmingDelete(false)}
          onConfirm={async () => {
            await api.deleteProject(project.id);
            toast("Project deleted", project.name, "ok");
            onDeleted(project.id);
          }}
        />
      )}
    </div>
  );
}

/** Recycle-bin glyph, drawn in the house SVG style (16-grid, 1.4 stroke). */
function TrashIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      aria-hidden="true"
      focusable="false"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M2.6 4.1h10.8" />
      <path d="M6.4 4.1V2.7h3.2v1.4" />
      <path d="M4 4.1l.55 8.1a1 1 0 0 0 1 .93h4.9a1 1 0 0 0 1-.93L12 4.1" />
      <path d="M6.5 6.6v4M9.5 6.6v4" />
    </svg>
  );
}

/** Confirm-and-delete overlay: a project's data and runs are removed for good,
 * so the destructive action is gated behind an explicit confirmation. */
function DeleteProjectDialog({
  project,
  datasetCount,
  runCount,
  onClose,
  onConfirm,
}: {
  project: ProjectSummary;
  datasetCount: number;
  runCount: number;
  onClose: () => void;
  onConfirm: () => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, onClose]);

  const owned = [
    datasetCount > 0
      ? `${datasetCount} dataset${datasetCount === 1 ? "" : "s"}`
      : null,
    runCount > 0 ? `${runCount} run${runCount === 1 ? "" : "s"}` : null,
  ].filter(Boolean);

  const runDelete = () => {
    setBusy(true);
    setError(null);
    // On success the card unmounts, taking this dialog with it, so we leave
    // `busy` set; only a failure returns control here to surface the reason.
    onConfirm().catch((err) => {
      setError(errorMessage(err));
      setBusy(false);
    });
  };

  // Portalled to <body> so the fixed overlay is positioned against the viewport,
  // not the card: `.pcard-clickable`'s hover transform makes the card a
  // containing block, which would anchor the modal to it (and jitter with hover).
  return createPortal(
    <div
      className="modal-ov"
      role="presentation"
      // React re-dispatches portal events up the *component* tree, so without
      // this a click here still bubbles to the card's onClick (opening the
      // project). Contain every click to the dialog.
      onClick={(e) => e.stopPropagation()}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !busy) onClose();
      }}
    >
      <div
        className="modal"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="del-title"
        aria-describedby="del-desc"
      >
        <div className="hd">
          <h2 id="del-title">Delete project</h2>
          <span className="sub">permanent</span>
        </div>
        <div className="body">
          <p id="del-desc" className="del-msg">
            Delete <b>{project.name}</b>
            {owned.length > 0 ? <> and its {owned.join(" and ")}</> : null}? Its
            data and outputs are removed from disk. Anything still shared with
            another project is kept.
          </p>
          <Callout tone="caution">This cannot be undone.</Callout>
          {error && <Callout tone="error">{error}</Callout>}
          <div className="pform-actions">
            <Button variant="danger" onClick={runDelete} disabled={busy}>
              {busy ? "Deleting…" : "Delete project"}
            </Button>
            <Button onClick={onClose} disabled={busy} autoFocus>
              Cancel
            </Button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}

function NewProjectForm({
  onCreated,
  onCancel,
}: {
  onCreated: (project: ProjectSummary) => void;
  onCancel: () => void;
}) {
  const toast = useToast();
  return (
    <div className="pcard">
      <ProjectMetadataForm
        heading="New project"
        initialName=""
        initialDescription=""
        submitLabel="Create"
        onSubmit={async (name, description) => {
          const project = await api.createProject(name, description);
          toast("Project created", project.name, "ok");
          onCreated(project);
        }}
        onCancel={onCancel}
      />
    </div>
  );
}

/** Name + description form shared by "create" and "edit in place". */
function ProjectMetadataForm({
  heading,
  initialName,
  initialDescription,
  submitLabel,
  onSubmit,
  onCancel,
}: {
  heading: string;
  initialName: string;
  initialDescription: string;
  submitLabel: string;
  onSubmit: (name: string, description: string) => Promise<void>;
  onCancel: () => void;
}) {
  const [name, setName] = useState(initialName);
  const [description, setDescription] = useState(initialDescription);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return (
    <form
      className="pform"
      aria-label={heading}
      onSubmit={(e) => {
        e.preventDefault();
        setBusy(true);
        setError(null);
        onSubmit(name, description)
          .catch((err) => setError(errorMessage(err)))
          .finally(() => setBusy(false));
      }}
    >
      {/* Limits mirror the API's MAX_NAME_CHARS / MAX_DESCRIPTION_CHARS
          (apps/api services/projects.py); keep the pairs in sync. */}
      <label className="pform-field">
        Name
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
          maxLength={120}
          autoFocus
        />
      </label>
      <label className="pform-field">
        Description
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={3}
          maxLength={2000}
        />
      </label>
      {error && <Callout tone="error">{error}</Callout>}
      <div className="pform-actions">
        <Button type="submit" variant="primary" disabled={busy || !name.trim()}>
          {busy ? "Saving…" : submitLabel}
        </Button>
        <Button type="button" onClick={onCancel} disabled={busy}>
          Cancel
        </Button>
      </div>
    </form>
  );
}
