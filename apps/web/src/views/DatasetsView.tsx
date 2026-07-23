import { Chip } from "../components";
import type { DatasetSummary } from "../lib/api";
import { ConditionsPanel } from "./datasets/ConditionsPanel";
import { GroupsPanel } from "./datasets/GroupsPanel";
import { ImageSequence } from "./datasets/ImageSequence";
import { UploadPreprocess } from "./datasets/UploadPreprocess";
import { useDatasetData } from "./datasets/useDatasetData";
import "./datasets/datasets.css";
import "./runs.css";

export function DatasetsView({ datasetId }: { datasetId?: string | null }) {
  const data = useDatasetData(datasetId);

  if (data.datasets === null) {
    return <p className="state-note" role="status">Loading datasets…</p>;
  }
  if (data.datasets.length === 0) {
    return <p className="state-note">No datasets yet. Drop image sequences into data/raw/.</p>;
  }

  return (
    <div className="stack">
      <DatasetHeader
        datasets={data.datasets}
        selected={data.selected}
        nFrames={data.detail?.n_frames ?? null}
        onSelect={data.setSelected}
      />

      {data.error && (
        <p className="state-note error" role="alert">{data.error}</p>
      )}

      {data.detail && (
        <>
          <UploadPreprocess
            detail={data.detail}
            preprocess={data.preprocess}
            busy={data.busy}
            onUpload={data.upload}
            onPreprocess={data.runPreprocess}
          />
          <ImageSequence datasetId={data.detail.id} nFrames={data.detail.n_frames} />
          <ConditionsPanel conditions={data.detail.conditions} />
          {data.groups && <GroupsPanel groups={data.groups} />}
        </>
      )}
    </div>
  );
}

interface DatasetHeaderProps {
  datasets: DatasetSummary[];
  selected: string | null;
  nFrames: number | null;
  onSelect: (id: string) => void;
}

function DatasetHeader({ datasets, selected, nFrames, onSelect }: DatasetHeaderProps) {
  return (
    <div className="dataset-head">
      <span className="id">{selected}</span>
      {nFrames != null && <Chip tone="accent">{nFrames} frames</Chip>}
      {datasets.length > 1 && (
        <select
          aria-label="Select dataset"
          value={selected ?? ""}
          onChange={(e) => onSelect(e.target.value)}
        >
          {datasets.map((dataset) => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.id}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
