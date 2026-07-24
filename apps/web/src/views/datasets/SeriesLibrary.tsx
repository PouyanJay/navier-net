import { useState } from "react";

import { Button, Chip, Panel } from "../../components";
import { useToast } from "../../components/Toast";
import { api, type DatasetSummary, type ProjectSummary } from "../../lib/api";
import { errorMessage } from "../../lib/errors";

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
  if (!summary.conditions_set) return <Chip tone="amber">needs conditions</Chip>;
  if (summary.processed) return <Chip tone="green">tensors ready</Chip>;
  return <Chip>uploaded</Chip>;
}

function seriesMeta(summary: DatasetSummary): string {
  const frames = `${summary.n_frames}`;
  if (!summary.frame_px) return `${frames} frames`;
  const [width, height] = summary.frame_px;
  const size = width === height ? `${width}²` : `${width}×${height}`;
  return `${frames} × ${size} px`;
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
  const [adding, setAdding] = useState(series.length === 0);

  return (
    <Panel title="Series library" subtitle="per-series conditions">
      {series.length === 0 && !adding && (
        <div className="dsempty">
          <b>No series yet</b>
          Upload the first high-speed sequence for this project to begin calibration and
          segmentation.
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
            <span className="st">{seriesChip(summary, trainedIds.has(summary.id))}</span>
          </button>
        ))}
      </div>
      {adding ? (
        <NewSeriesForm
          project={project}
          onDone={(updated, seriesId) => {
            setAdding(false);
            onProjectChanged(updated);
            onSelect(seriesId);
          }}
          onCancel={() => setAdding(false)}
        />
      ) : (
        <button type="button" className="addds" onClick={() => setAdding(true)}>
          + Upload new series — TIFF frames
        </button>
      )}
      <p className="note">
        <b>Transfer learning:</b> once two or more series are configured, Stage B can train
        jointly across heat-flux conditions.
      </p>
    </Panel>
  );
}

const SERIES_ID_RE = /^[A-Za-z0-9._-]+$/;

function NewSeriesForm({
  project,
  onDone,
  onCancel,
}: {
  project: ProjectSummary;
  onDone: (project: ProjectSummary, seriesId: string) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState("");
  const [files, setFiles] = useState<FileList | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const toast = useToast();

  const validName = SERIES_ID_RE.test(name) && !project.datasets.includes(name);

  async function upload() {
    if (!validName || !files?.length) return;
    setBusy(true);
    setError(null);
    try {
      await api.uploadFrames(name, files);
    } catch (err) {
      setError(`Upload failed: ${errorMessage(err)}`);
      setBusy(false);
      return;
    }
    try {
      // Frames are on disk now — a failure past this point is only the link.
      const updated = await api.updateProject(project.id, {
        datasets: [...project.datasets, name],
      });
      toast("Series uploaded", `${name} — ${files.length} frames`, "ok");
      onDone(updated, name);
    } catch (err) {
      setError(`Uploaded, but linking the series failed: ${errorMessage(err)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      className="pform addds-form"
      aria-label="Upload new series"
      onSubmit={(e) => {
        e.preventDefault();
        void upload();
      }}
    >
      <label className="pform-field">
        Series name
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="mid_T"
          required
          pattern="[A-Za-z0-9._\-]+"
          autoFocus
        />
      </label>
      <label className="pform-field">
        Frames
        <input
          type="file"
          accept=".tif,.tiff,image/tiff"
          multiple
          onChange={(e) => setFiles(e.target.files)}
        />
      </label>
      {name.length > 0 && !validName && (
        <p className="state-note error">
          {project.datasets.includes(name)
            ? "A series with this name already exists in the project."
            : "Series names use letters, digits, dots, dashes, and underscores."}
        </p>
      )}
      {error && (
        <p className="state-note error" role="alert">
          {error}
        </p>
      )}
      <div className="pform-actions">
        <Button type="submit" variant="primary" disabled={busy || !validName || !files?.length}>
          {busy ? "Uploading…" : "Upload"}
        </Button>
        <Button type="button" onClick={onCancel} disabled={busy}>
          Cancel
        </Button>
      </div>
    </form>
  );
}
