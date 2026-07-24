import { Fragment, useEffect, useState } from "react";

import { Chip } from "../components";
import { useToast } from "../components/Toast";
import { api, type DatasetSummary, type RunSummary } from "../lib/api";
import { errorMessage } from "../lib/errors";
import "./datasets/datasets.css";
import "./runs.css";

interface ProjectFacts {
  dataset: DatasetSummary;
  runs: RunSummary[];
}

/** Pipeline progress for one dataset-project: which stages have real artifacts. */
function stageDots(facts: ProjectFacts): boolean[] {
  const trained = facts.runs.some((run) => run.status === "trained");
  const evaluated = facts.runs.some((run) => run.iou_holdout != null);
  return [facts.dataset.processed, facts.dataset.processed, trained, evaluated];
}

/** A grid of dataset-projects; opening one jumps into its pipeline. */
export function ProjectsView({ onOpen }: { onOpen: (id: string) => void }) {
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const toast = useToast();

  useEffect(() => {
    Promise.all([api.listDatasets(), api.listRuns()])
      .then(([ds, rs]) => {
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
  if (datasets === null) {
    return (
      <p className="state-note" role="status">
        Loading projects…
      </p>
    );
  }
  if (datasets.length === 0) {
    return <p className="state-note">No datasets yet. Drop image sequences into data/raw/.</p>;
  }

  return (
    <div className="project-grid">
      {datasets.map((dataset) => (
        <ProjectCard
          key={dataset.id}
          facts={{ dataset, runs: runs.filter((run) => run.dataset === dataset.id) }}
          onOpen={onOpen}
        />
      ))}
      <button
        type="button"
        className="newproj"
        onClick={() =>
          toast(
            "Project creation is not available yet",
            "this workspace currently scopes one repository",
          )
        }
      >
        <span className="plus" aria-hidden="true">
          +
        </span>
        New project
      </button>
    </div>
  );
}

// Short stage names for the progress label, matching the mockup's pipe copy.
const STAGE_NAMES = ["Data", "Model", "Solve", "Results"];

function statusChip(facts: ProjectFacts) {
  const trained = facts.runs.some((run) => run.status === "trained");
  if (trained) return <Chip tone="green">Stage A · trained</Chip>;
  if (!facts.dataset.processed) return <Chip tone="amber">needs preprocess</Chip>;
  return <Chip>No runs yet</Chip>;
}

function ProjectCard({
  facts,
  onOpen,
}: {
  facts: ProjectFacts;
  onOpen: (id: string) => void;
}) {
  const { dataset } = facts;
  const dots = stageDots(facts);
  const doneCount = dots.filter(Boolean).length;
  const stageLabel =
    doneCount === 0 ? "not started" : `${STAGE_NAMES[Math.min(doneCount, 3)]} stage`;
  return (
    <button type="button" className="pcard" onClick={() => onOpen(dataset.id)}>
      <div className="pcard-top">
        <h3>{dataset.id}</h3>
        {statusChip(facts)}
      </div>
      <p className="purpose">
        Reconstruct the hidden velocity and volume-fraction fields of a confined vapor slug
        from its high-speed image sequence.
      </p>
      <div className="pmeta mono">
        <span>
          <b>{dataset.n_frames}</b> frames
        </span>
        <span>
          <b>{facts.runs.length}</b> runs
        </span>
      </div>
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
      <div className="pfoot">
        <span className="mono">naviernet://{dataset.id}</span>
        <span className="popen">Open →</span>
      </div>
    </button>
  );
}
