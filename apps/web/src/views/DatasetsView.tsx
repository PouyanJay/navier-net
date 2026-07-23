import { Chip } from "../components";
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
      <div className="dataset-head">
        <span className="id">{data.selected}</span>
        {data.detail && <Chip tone="accent">{data.detail.n_frames} frames</Chip>}
        {data.datasets.length > 1 && (
          <select
            aria-label="Select dataset"
            value={data.selected ?? ""}
            onChange={(e) => data.setSelected(e.target.value)}
          >
            {data.datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.id}
              </option>
            ))}
          </select>
        )}
      </div>

      {data.error && (
        <p className="state-note error" role="alert">
          {data.error}
        </p>
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
