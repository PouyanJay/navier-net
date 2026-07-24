import { useState } from "react";

import { Chip, Panel } from "../../components";
import type { DatasetSummary, ProjectSummary } from "../../lib/api";
import { NewSeriesModal } from "./NewSeriesModal";

interface SeriesLibraryProps {
  project: ProjectSummary;
  series: DatasetSummary[];
  trainedIds: Set<string>;
  selected: string | null;
  onSelect: (id: string) => void;
  /** Called with the updated project after a new series is uploaded+attached. */
  onProjectChanged: (project: ProjectSummary) => void;
}

function seriesChip(summary: DatasetSummary, trained: boolean) {
  if (trained) return <Chip tone="green">trained</Chip>;
  if (!summary.conditions_set)
    return <Chip tone="amber">needs conditions</Chip>;
  if (summary.processed) return <Chip tone="green">tensors ready</Chip>;
  return <Chip>uploaded</Chip>;
}

function seriesMeta(summary: DatasetSummary): string {
  const parts = [`${summary.n_frames}`];
  if (summary.frame_px) {
    const [width, height] = summary.frame_px;
    parts.push(`× ${width === height ? `${width}²` : `${width}×${height}`}`);
  } else {
    parts.push("frames");
  }
  if (summary.dt_frame_ms != null) parts.push(`· Δt ${summary.dt_frame_ms} ms`);
  return parts.join(" ");
}

/** The project's uploaded series; select one to edit it, or add another. */
export function SeriesLibrary({
  project,
  series,
  trainedIds,
  selected,
  onSelect,
  onProjectChanged,
}: SeriesLibraryProps) {
  const [adding, setAdding] = useState(false);

  return (
    <Panel title="Series library" subtitle="per-series conditions">
      {series.length === 0 && (
        <div className="dsempty">
          <b>No series yet</b>
          Upload the first high-speed sequence for this project to begin
          calibration and segmentation.
        </div>
      )}
      <div className="dsrows">
        {series.map((summary) => (
          <button
            key={summary.id}
            type="button"
            className={summary.id === selected ? "dsrow sel" : "dsrow"}
            aria-current={summary.id === selected || undefined}
            onClick={() => onSelect(summary.id)}
          >
            <span className="ic mono" aria-hidden="true">
              TIF
            </span>
            <span className="m">
              <b>{summary.id}</b>
              <span className="mono">{seriesMeta(summary)}</span>
            </span>
            <span className="st">
              {seriesChip(summary, trainedIds.has(summary.id))}
            </span>
          </button>
        ))}
      </div>
      <button type="button" className="addds" onClick={() => setAdding(true)}>
        + Upload new series · TIFF frames
      </button>
      {adding && (
        <NewSeriesModal
          project={project}
          onClose={() => setAdding(false)}
          onAttached={(updated, seriesId) => {
            onProjectChanged(updated);
            onSelect(seriesId);
          }}
        />
      )}
      <p className="note">
        <b>Transfer learning:</b> once two or more series are configured, Stage
        B can train jointly across heat-flux conditions.
      </p>
    </Panel>
  );
}
