import { useRef } from "react";

import { Button, Chip, Panel, StatusDot } from "../../components";
import { artifactUrl, type DatasetDetail, type PreprocessStatus } from "../../lib/api";

interface UploadPreprocessProps {
  detail: DatasetDetail;
  preprocess: PreprocessStatus | null;
  busy: boolean;
  onUpload: (files: FileList) => void;
  onPreprocess: () => void;
}

const STATE_TONE = {
  idle: "default",
  running: "accent",
  done: "green",
  error: "red",
} as const;

/** Upload an image sequence and drive preprocessing; shows the QC preview. */
export function UploadPreprocess({
  detail,
  preprocess,
  busy,
  onUpload,
  onPreprocess,
}: UploadPreprocessProps) {
  const fileInput = useRef<HTMLInputElement>(null);
  const state = preprocess?.state ?? "idle";
  const running = state === "running";
  const hasQc = preprocess?.has_qc ?? detail.has_qc;

  return (
    <Panel
      title="Calibration & segmentation"
      subtitle="Upload frames, then preprocess into training tensors"
    >
      <div className="upload-row">
        <input
          ref={fileInput}
          type="file"
          accept=".tif,.tiff,image/tiff"
          multiple
          aria-label="Image sequence (TIFF frames)"
          onChange={(e) => e.target.files?.length && onUpload(e.target.files)}
        />
        <Button
          variant="primary"
          onClick={onPreprocess}
          disabled={detail.n_frames === 0 || running || busy}
        >
          {running ? "Preprocessing…" : "Run preprocessing"}
        </Button>
        <StatusDot tone={STATE_TONE[state]} label={state} />
        {detail.processed && <Chip tone="green">tensors ready</Chip>}
      </div>

      {state === "error" && preprocess?.message && (
        <p className="state-note error" role="alert">
          Preprocessing failed: {preprocess.message}
        </p>
      )}

      {hasQc && <QcFigure datasetId={detail.id} cacheKey={state} />}
    </Panel>
  );
}

function QcFigure({ datasetId, cacheKey }: { datasetId: string; cacheKey: string }) {
  return (
    <img
      key={cacheKey}
      className="qc-figure"
      src={artifactUrl.datasetQc(datasetId)}
      alt="Preprocessing quality-control: growth curve, interface evolution, signed distance"
    />
  );
}
