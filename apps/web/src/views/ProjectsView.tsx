import { useEffect, useState } from "react";

import { Chip, Panel, StatusDot } from "../components";
import { api, type DatasetSummary } from "../lib/api";
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
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
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
          {datasets.map((d) => (
            <button key={d.id} type="button" className="project-card" onClick={() => onOpen(d.id)}>
              <div className="name">{d.id}</div>
              <div className="meta">
                <Chip tone="accent">{d.n_frames} frames</Chip>
                <StatusDot
                  tone={d.processed ? "green" : "default"}
                  label={d.processed ? "processed" : "raw"}
                />
              </div>
            </button>
          ))}
        </div>
      )}
    </Panel>
  );
}
