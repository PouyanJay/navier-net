import { useEffect, useState } from "react";

import { Chip, Panel, StatusDot } from "../components";
import { api, type DatasetSummary } from "../lib/api";
import { errorMessage } from "../lib/errors";
import "./datasets/datasets.css";
import "./runs.css";

/** A grid of datasets as project cards; opening one jumps to Datasets & conditions. */
export function ProjectsView({ onOpen }: { onOpen: (id: string) => void }) {
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listDatasets()
      .then(setDatasets)
      .catch((err) => setError(errorMessage(err)));
  }, []);

  if (error) {
    return <p className="state-note error" role="alert">{error}. Is the API running on :8000?</p>;
  }
  if (datasets === null) {
    return <p className="state-note" role="status">Loading projects…</p>;
  }

  return (
    <Panel title="Projects" subtitle="Each dataset is a project — open one to configure and run it">
      {datasets.length === 0 ? (
        <p className="state-note">No datasets yet. Drop image sequences into data/raw/.</p>
      ) : (
        <div className="project-grid">
          {datasets.map((dataset) => (
            <ProjectCard key={dataset.id} dataset={dataset} onOpen={onOpen} />
          ))}
        </div>
      )}
    </Panel>
  );
}

function ProjectCard({
  dataset,
  onOpen,
}: {
  dataset: DatasetSummary;
  onOpen: (id: string) => void;
}) {
  return (
    <button type="button" className="project-card" onClick={() => onOpen(dataset.id)}>
      <div className="name">{dataset.id}</div>
      <div className="meta">
        <Chip tone="accent">{dataset.n_frames} frames</Chip>
        <StatusDot
          tone={dataset.processed ? "green" : "default"}
          label={dataset.processed ? "processed" : "raw"}
        />
      </div>
    </button>
  );
}
