import { useEffect, useRef } from "react";

import { Button } from "../../components";
import { ArtifactImage } from "../../components/ArtifactImage";
import { artifactUrl, type DatasetDetail } from "../../lib/api";

interface FrameLightboxProps {
  detail: DatasetDetail;
  /** 1-based camera frame on show. */
  frame: number;
  onFrameChange: (frame: number) => void;
  onToggleExcluded: (frame: number) => void;
  onClose: () => void;
}

/** One frame at full size, stepped with the arrow keys. Opened by
 * double-clicking a tile in the sequence strip (or its expand control). */
export function FrameLightbox({
  detail,
  frame,
  onFrameChange,
  onToggleExcluded,
  onClose,
}: FrameLightboxProps) {
  const dialog = useRef<HTMLDivElement>(null);
  const excluded = detail.excluded_frames.includes(frame);
  const holdout = frame === detail.holdout_frame;
  const timeMs = (frame - 1) * detail.conditions.dt_frame_ms;

  useEffect(() => dialog.current?.focus(), []); // focus moves into the dialog

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
      if (event.key === "ArrowLeft" && frame > 1) onFrameChange(frame - 1);
      if (event.key === "ArrowRight" && frame < detail.n_frames)
        onFrameChange(frame + 1);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [frame, detail.n_frames, onFrameChange, onClose]);

  return (
    <div
      className="modal-ov lightbox-ov"
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={dialog}
        tabIndex={-1}
        className="modal lightbox"
        role="dialog"
        aria-modal="true"
        aria-label={`Frame ${frame} of ${detail.id}`}
      >
        <div className="hd">
          <h2 className="mono">f{String(frame).padStart(2, "0")}</h2>
          <span className="sub">
            {timeMs.toFixed(1)} ms · frame {frame} of {detail.n_frames}
            {holdout && " · holdout"}
            {excluded && " · excluded"}
          </span>
        </div>

        <div className="lightbox-view">
          <ArtifactImage
            src={artifactUrl.datasetFrame(detail.id, frame)}
            alt={`Frame ${frame} of ${detail.id} at ${timeMs.toFixed(1)} milliseconds`}
          />
        </div>

        <div className="lightbox-actions">
          <Button
            onClick={() => onFrameChange(frame - 1)}
            disabled={frame <= 1}
            aria-label="Previous frame"
          >
            ← Prev
          </Button>
          <Button
            onClick={() => onFrameChange(frame + 1)}
            disabled={frame >= detail.n_frames}
            aria-label="Next frame"
          >
            Next →
          </Button>
          <Button
            variant={excluded ? "primary" : undefined}
            onClick={() => onToggleExcluded(frame)}
            disabled={holdout}
            title={
              holdout
                ? "The holdout frame is the run's only unsupervised check; move the holdout before excluding it"
                : undefined
            }
          >
            {excluded ? "Include in training" : "Exclude from training"}
          </Button>
          <Button onClick={onClose}>Close</Button>
        </div>
      </div>
    </div>
  );
}
