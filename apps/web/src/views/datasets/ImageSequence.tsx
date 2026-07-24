import { Panel } from "../../components";
import { ArtifactImage } from "../../components/ArtifactImage";
import { artifactUrl, type DatasetDetail } from "../../lib/api";

function calibrationLine(detail: DatasetDetail): string {
  if (detail.um_per_px == null || detail.frame_px == null) {
    return `${detail.n_frames} raw frame${detail.n_frames === 1 ? "" : "s"} — preprocess to calibrate`;
  }
  const fovMm = (detail.um_per_px * detail.frame_px[0]) / 1000;
  return `auto-calibrated · ${detail.um_per_px.toFixed(3)} µm px⁻¹ · FOV ${fovMm.toFixed(2)} mm`;
}

/** The raw frames as an inline film strip; scrolls when it overflows. */
export function ImageSequence({ detail }: { detail: DatasetDetail }) {
  const frames = Array.from({ length: detail.n_frames }, (_, i) => i + 1);
  return (
    <Panel title={`Image sequence — ${detail.id}`} subtitle={calibrationLine(detail)}>
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
                  f{String(n).padStart(2, "0")} · {((n - 1) * detail.conditions.dt_frame_ms).toFixed(1)} ms
                </figcaption>
              </figure>
            );
          })}
        </div>
      )}
      {detail.notes && (
        <p className="note">
          <b>Auto-detected:</b> {detail.notes}
        </p>
      )}
    </Panel>
  );
}
