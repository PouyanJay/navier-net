import { Panel } from "../../components";
import { artifactUrl } from "../../lib/api";

interface ImageSequenceProps {
  datasetId: string;
  nFrames: number;
}

/** Thumbnails of the raw high-speed frames (TIFFs rendered to PNG by the API). */
export function ImageSequence({ datasetId, nFrames }: ImageSequenceProps) {
  const frames = Array.from({ length: nFrames }, (_, i) => i + 1);
  return (
    <Panel title="Image sequence" subtitle={`${nFrames} raw frame${nFrames === 1 ? "" : "s"}`}>
      {nFrames === 0 ? (
        <p className="state-note">
          Upload the first high-speed sequence to begin calibration and segmentation.
        </p>
      ) : (
        <div className="sequence-grid">
          {frames.map((n) => (
            <figure key={n}>
              <img src={artifactUrl.datasetFrame(datasetId, n)} alt={`frame ${n}`} loading="lazy" />
              <figcaption>frame {n}</figcaption>
            </figure>
          ))}
        </div>
      )}
    </Panel>
  );
}
