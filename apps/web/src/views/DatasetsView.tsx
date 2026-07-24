import { type ProjectSummary } from "../lib/api";
import { ConditionsPanel } from "./datasets/ConditionsPanel";
import { GroupsPanel } from "./datasets/GroupsPanel";
import { ImageSequence } from "./datasets/ImageSequence";
import { QcPanel } from "./datasets/QcPanel";
import { SeriesLibrary } from "./datasets/SeriesLibrary";
import { UploadPreprocess } from "./datasets/UploadPreprocess";
import { useDatasetData, useQcData, useTrainedIds } from "./datasets/useDatasetData";
import "./datasets/datasets.css";
import "./runs.css";

interface DatasetsViewProps {
  /** The stage is always entered from a project; series live inside it. */
  project: ProjectSummary;
  onProjectChanged: (project: ProjectSummary) => void;
}

/** The datasets stage: the project's series library plus the selected series'
 * frames, calibration, conditions, groups, and preprocessing QC. */
export function DatasetsView({ project, onProjectChanged }: DatasetsViewProps) {
  const data = useDatasetData(project.datasets[0] ?? null);
  const trainedIds = useTrainedIds();
  const { qc, qcError } = useQcData(data.selected, data.detail?.processed ?? false);

  if (data.datasets === null && data.error) {
    return (
      <p className="state-note error" role="alert">
        Could not load datasets: {data.error}. Is the API running on :8000?
      </p>
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
  const inScope = data.selected != null && series.some((d) => d.id === data.selected);
  const detail = inScope ? data.detail : null;

  return (
    <div className="dsx">
      <SeriesLibrary
        project={project}
        series={series}
        trainedIds={trainedIds}
        selected={inScope ? data.selected : null}
        onSelect={data.setSelected}
        onProjectChanged={(updated) => {
          // The new series must appear in the library without a reload.
          onProjectChanged(updated);
          data.refresh().catch(() => {});
        }}
      />

      <div className="dsx-main">
        {data.error && (
          <p className="state-note error" role="alert">
            {data.error}
          </p>
        )}
        {detail && (
          <>
            <ImageSequence detail={detail} />
            <UploadPreprocess
              detail={detail}
              preprocess={data.preprocess}
              busy={data.busy}
              onUpload={data.upload}
              onPreprocess={data.runPreprocess}
            />
            {qcError && (
              <p className="state-note error" role="alert">
                Could not load the preprocessing QC: {qcError}
              </p>
            )}
            {qc && <QcPanel qc={qc} />}
            <ConditionsPanel
              datasetId={detail.id}
              conditions={detail.conditions}
              onSaved={data.applyConditions}
            />
            {data.groups && <GroupsPanel datasetId={detail.id} groups={data.groups} />}
          </>
        )}
      </div>
    </div>
  );
}
