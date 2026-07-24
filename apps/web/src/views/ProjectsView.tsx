import { Fragment, useEffect, useState } from "react";

import { Button, Chip } from "../components";
import { useToast } from "../components/Toast";
import {
  api,
  type DatasetSummary,
  type ProjectSummary,
  type RunSummary,
} from "../lib/api";
import { errorMessage } from "../lib/errors";
import "./datasets/datasets.css";
import "./runs.css";

interface ProjectFacts {
  project: ProjectSummary;
  dataset: DatasetSummary | null;
  runs: RunSummary[];
}

/** Pipeline progress for one project: which stages have real artifacts. */
function stageDots(facts: ProjectFacts): boolean[] {
  if (!facts.dataset) return [false, false, false, false];
  const trained = facts.runs.some((run) => run.status === "trained");
  const evaluated = facts.runs.some((run) => run.iou_holdout != null);
  return [facts.dataset.processed, facts.dataset.processed, trained, evaluated];
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
      <p className="state-note error" role="alert">
        {error}. Is the API running on :8000?
      </p>
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
    setProjects((current) => (current ?? []).map((p) => (p.id === updated.id ? updated : p)));
    onChanged?.();
  };

  return (
    <div className="project-grid">
      {projects.map((project) => (
        <ProjectCard
          key={project.id}
          facts={{
            project,
            dataset: datasets.find((d) => d.id === project.dataset) ?? null,
            runs: runs.filter((run) => run.dataset === project.dataset),
          }}
          onOpen={onOpen}
          onSaved={replaceProject}
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
        <button type="button" className="newproj" onClick={() => onCreatingChange(true)}>
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
  if (!facts.dataset) return <Chip>Awaiting data</Chip>;
  if (trained) return <Chip tone="green">Stage A · trained</Chip>;
  if (!facts.dataset.processed) return <Chip tone="amber">needs preprocess</Chip>;
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
}: {
  facts: ProjectFacts;
  onOpen: (project: ProjectSummary) => void;
  onSaved: (project: ProjectSummary) => void;
}) {
  const { project, dataset } = facts;
  const [editing, setEditing] = useState(false);
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
            const updated = await api.updateProject(project.id, { name, description });
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
          <b>{dataset?.n_frames ?? 0}</b> frames
        </span>
        <span>
          <b>{facts.runs.length}</b> runs
        </span>
      </div>
      <StagePipe dots={dots} />
      <div className="pfoot">
        <span className="mono">naviernet://{dataset?.id ?? project.id.slice(0, 8)}</span>
        <span className="pfoot-actions">
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
    </div>
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
      {error && (
        <p className="state-note error" role="alert">
          {error}
        </p>
      )}
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
