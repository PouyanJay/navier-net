import { Callout } from "../components";
import { type ProjectSummary } from "../lib/api";
import { GroupsPanel } from "./datasets/GroupsPanel";
import { ImageSequence } from "./datasets/ImageSequence";
import { SeriesLibrary } from "./datasets/SeriesLibrary";
import {
  useDatasetData,
  useQcData,
  useTrainedIds,
} from "./datasets/useDatasetData";
import "./datasets/datasets.css";
import "./runs.css";

interface DatasetsViewProps {
  /** The stage is always entered from a project; series live inside it. */
  project: ProjectSummary;
  onProjectChanged: (project: ProjectSummary) => void;
}

/** The datasets stage: the project's series library plus the selected series'
 * frames + preprocessing QC (one card) and its dimensionless groups. Operating
 * conditions are set at upload and editable per series from the library card;
 * editing a tensor-baked condition prompts a re-preprocess. */
export function DatasetsView({ project, onProjectChanged }: DatasetsViewProps) {
  const data = useDatasetData(project.datasets[0] ?? null);
  const trainedIds = useTrainedIds();
  const { qc, qcError } = useQcData(
    data.selected,
    data.detail?.processed ?? false,
  );

  if (data.datasets === null && data.error) {
    return (
      <Callout tone="error" title="Could not load datasets">
        {data.error}. Is the API running on :8000?
      </Callout>
    );
  }
  if (data.datasets === null) {
    return (
      <p className="state-note" role="status">
        Loading datasets…
      </p>
    );
  }

  const series = data.datasets.filter((d) => project.datasets.includes(d.id));
  const inScope =
    data.selected != null && series.some((d) => d.id === data.selected);
  const detail = inScope ? data.detail : null;

  return (
    <div className="dsx">
      <SeriesLibrary
        project={project}
        series={series}
        trainedIds={trainedIds}
        selected={inScope ? data.selected : null}
        detail={detail}
        onSelect={data.setSelected}
        onProjectChanged={(updated) => {
          // The new series must appear in the library without a reload.
          onProjectChanged(updated);
          data.refresh().catch(() => {});
        }}
        onSaveConditions={data.saveConditions}
        onSaveLabel={data.saveLabel}
        onPreprocess={() => void data.runPreprocess()}
        preprocessing={data.preprocess?.state === "running"}
      />

      <div className="dsx-main">
        {data.error && <Callout tone="error">{data.error}</Callout>}
        {detail && (
          <>
            <ImageSequence
              detail={detail}
              qc={qc}
              preprocess={data.preprocess}
              onPreprocess={data.runPreprocess}
              onToggleExcluded={(frame) => void data.toggleExcludedFrame(frame)}
              exclusionError={data.exclusionError}
            />
            {qcError && (
              <Callout tone="error" title="Could not load the preprocessing QC">
                {qcError}
              </Callout>
            )}
            {data.groups && (
              <GroupsPanel datasetId={detail.id} groups={data.groups} />
            )}
          </>
        )}
      </div>
    </div>
  );
}
