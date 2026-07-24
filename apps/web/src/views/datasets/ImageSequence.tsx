import { Button, Panel } from "../../components";
import { ArtifactImage } from "../../components/ArtifactImage";
import { artifactUrl, type DatasetDetail, type PreprocessStatus } from "../../lib/api";

interface ImageSequenceProps {
  detail: DatasetDetail;
  preprocess: PreprocessStatus | null;
  onPreprocess: () => void;
}

function calibrationLine(detail: DatasetDetail): string {
  if (detail.um_per_px == null || detail.frame_px == null) {
    return `${detail.n_frames} raw frame${detail.n_frames === 1 ? "" : "s"} — preprocess to calibrate`;
  }
  const fovMm = (detail.um_per_px * detail.frame_px[0]) / 1000;
  return `auto-calibrated · ${detail.um_per_px.toFixed(3)} µm px⁻¹ · FOV ${fovMm.toFixed(2)} mm`;
}

/** The raw frames as an inline film strip; scrolls when it overflows. The
 * panel also owns (re)running preprocessing for an unprocessed series. */
export function ImageSequence({ detail, preprocess, onPreprocess }: ImageSequenceProps) {
  const frames = Array.from({ length: detail.n_frames }, (_, i) => i + 1);
  const running = preprocess?.state === "running";
  return (
    <Panel
      title={`Image sequence — ${detail.id}`}
      subtitle={calibrationLine(detail)}
      actions={
        !detail.processed && detail.n_frames > 0 ? (
          <Button variant="primary" onClick={onPreprocess} disabled={running}>
            {running ? "Preprocessing…" : "Run preprocessing"}
          </Button>
        ) : undefined
      }
    >
      {detail.n_frames === 0 ? (
        <p className="state-note">
          Upload the first high-speed sequence to begin calibration and segmentation.
        </p>
      ) : (
        <div className="strip" role="list" aria-label={`${detail.n_frames} raw frames`}>
          {frames.map((n) => {
            const holdout = n === detail.holdout_frame;
            return (
              <figure key={n} className={holdout ? "fr hold" : "fr"} role="listitem">
                <ArtifactImage
                  src={artifactUrl.datasetFrame(detail.id, n)}
                  alt={`frame ${n}${holdout ? " (holdout)" : ""}`}
                  loading="lazy"
                />
                <figcaption className="mono">
                  f{String(n).padStart(2, "0")} ·{" "}
                  {((n - 1) * detail.conditions.dt_frame_ms).toFixed(1)} ms
                </figcaption>
              </figure>
            );
          })}
        </div>
      )}
      {running && (
        <div className="modal-progress">
          <div className="meter indeterminate" role="progressbar" aria-label="Preprocessing">
            <i />
          </div>
          <p className="state-note" role="status">
            Preprocessing — calibrating, segmenting, building tensors…
          </p>
        </div>
      )}
      {preprocess?.state === "error" && preprocess.message && (
        <p className="state-note error" role="alert">
          Preprocessing failed: {preprocess.message}
        </p>
      )}
      {detail.notes && (
        <p className="note">
          <b>Auto-detected:</b> {detail.notes}
        </p>
      )}
    </Panel>
  );
}
