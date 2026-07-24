import { useState } from "react";

import { Panel } from "../../components";
import { ArtifactImage } from "../../components/ArtifactImage";
import { artifactUrl, type ArtifactFlags } from "../../lib/api";

interface DeliverablesProps {
  runId: string;
  artifacts: ArtifactFlags;
}

/** Downloadable run outputs: tensors, checkpoint, video, and figure previews. */
export function Deliverables({ runId, artifacts }: DeliverablesProps) {
  return (
    <Panel title="Deliverables" subtitle="Everything the run produced — fully reproducible">
      <div className="deliverables">
        <FileRow
          name="training_data.npz"
          desc="Preprocessed tensors"
          href={artifactUrl.tensors(runId)}
        />
        {artifacts.checkpoint && (
          <FileRow
            name="ckpt.pt"
            desc="Trained checkpoint (resumable)"
            href={artifactUrl.checkpoint(runId)}
          />
        )}
      </div>

      {artifacts.video && <VideoPreview runId={runId} />}

      {artifacts.figures.length > 0 && (
        <FigureGrid runId={runId} figures={artifacts.figures} />
      )}
    </Panel>
  );
}

function FileRow({ name, desc, href }: { name: string; desc: string; href: string }) {
  return (
    <div className="deliverable">
      <div>
        <div className="name">{name}</div>
        <div className="desc">{desc}</div>
      </div>
      <a className="btn" href={href} download>
        Download
      </a>
    </div>
  );
}

function VideoPreview({ runId }: { runId: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <p className="state-note error" role="alert">
        The reconstruction video could not be loaded.
      </p>
    );
  }
  return (
    <video
      className="result-video"
      controls
      preload="metadata"
      onError={() => setFailed(true)}
    >
      <source src={artifactUrl.video(runId)} type="video/mp4" />
      Your browser cannot play this video.
    </video>
  );
}

function FigureGrid({ runId, figures }: { runId: string; figures: string[] }) {
  return (
    <div className="figure-grid">
      {figures.map((name) => (
        <a key={name} href={artifactUrl.figure(runId, name)} target="_blank" rel="noreferrer">
          <figure>
            <ArtifactImage src={artifactUrl.figure(runId, name)} alt={name} loading="lazy" />
            <figcaption>{name}</figcaption>
          </figure>
        </a>
      ))}
    </div>
  );
}
