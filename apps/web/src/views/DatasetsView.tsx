import { useEffect, useState } from "react";

import { api, type ProjectSummary, type QcData, type RunSummary } from "../lib/api";
import { isTrainedRun } from "../lib/runs";
import { ConditionsPanel } from "./datasets/ConditionsPanel";
import { GroupsPanel } from "./datasets/GroupsPanel";
import { ImageSequence } from "./datasets/ImageSequence";
import { QcPanel } from "./datasets/QcPanel";
import { SeriesLibrary } from "./datasets/SeriesLibrary";
import { UploadPreprocess } from "./datasets/UploadPreprocess";
import { useDatasetData } from "./datasets/useDatasetData";
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
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [qc, setQc] = useState<QcData | null>(null);

  useEffect(() => {
    api
      .listRuns()
      .then(setRuns)
      .catch(() => setRuns([])); // chips only — panels surface real errors
  }, []);

  // The QC charts follow the selected series and refresh after preprocessing.
  const selectedProcessed = data.detail?.processed ?? false;
  useEffect(() => {
    let stale = false;
    setQc(null);
    if (data.selected && selectedProcessed) {
      api
        .getQcData(data.selected)
        .then((payload) => {
          if (!stale) setQc(payload);
        })
        .catch(() => {}); // 404 before preprocessing is expected
    }
    return () => {
      stale = true;
    };
  }, [data.selected, selectedProcessed]);

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
  const trainedIds = new Set(
    runs.filter(isTrainedRun).flatMap((run) => (run.dataset ? [run.dataset] : [])),
  );

  return (
    <div className="grid dsx">
      <SeriesLibrary
        project={project}
        series={series}
        trainedIds={trainedIds}
        selected={inScope ? data.selected : null}
        onSelect={data.setSelected}
        onProjectChanged={onProjectChanged}
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
