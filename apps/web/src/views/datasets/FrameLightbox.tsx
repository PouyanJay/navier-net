import { useEffect, useRef, useState } from "react";

import { Button } from "../../components";
import { ArtifactImage } from "../../components/ArtifactImage";
import {
  artifactUrl,
  type DatasetDetail,
  type QcData,
  type QcInterfaceFrame,
} from "../../lib/api";

interface FrameLightboxProps {
  detail: DatasetDetail;
  /** Preprocessing QC, when available: source of the detected boundary. */
  qc: QcData | null;
  /** 1-based camera frame on show. */
  frame: number;
  onFrameChange: (frame: number) => void;
  onToggleExcluded: (frame: number) => void;
  onClose: () => void;
}

/** What it takes to place a ring on the raw frame: the pipeline flips x (raw
 * camera flow runs right to left) and crops the imaged band off the top, so the
 * overlay undoes both. */
interface FrameGeometry {
  /** Raw frame width in pixels. */
  width: number;
  /** Raw frame height in pixels. */
  height: number;
  /** x*,y* are in L_ref units; multiply by this to get pixels. */
  pxPerStar: number;
  /** Top row of the imaged band the rings were cut to. */
  yRoiTop: number;
}

/** A ring of [x*, y*] points as an SVG path in raw-frame pixel space. */
function ringPath(ring: number[][], geo: FrameGeometry): string {
  const points = ring.map(([xStar, yStar]) => {
    const col = geo.width - 0.5 - xStar * geo.pxPerStar;
    const row = yStar * geo.pxPerStar - 0.5 + geo.yRoiTop;
    return `${col.toFixed(2)} ${row.toFixed(2)}`;
  });
  return points.length ? `M${points.join("L")}Z` : "";
}

/** One frame at full size, stepped with the arrow keys. Opened by
 * double-clicking a tile in the sequence strip (or its expand control). */
export function FrameLightbox({
  detail,
  qc,
  frame,
  onFrameChange,
  onToggleExcluded,
  onClose,
}: FrameLightboxProps) {
  const dialog = useRef<HTMLDivElement>(null);
  const [showBoundary, setShowBoundary] = useState(true);
  const excluded = detail.excluded_frames.includes(frame);
  const holdout = frame === detail.holdout_frame;
  const timeMs = (frame - 1) * detail.conditions.dt_frame_ms;

  // The overlay needs both the QC silhouettes and the raw frame's geometry to
  // map x*,y* back to pixels; either missing means no boundary to draw.
  const iface = qc?.interface ?? null;
  const geo: FrameGeometry | null =
    iface && detail.frame_px && detail.um_per_px != null
      ? {
          width: detail.frame_px[0],
          height: detail.frame_px[1],
          pxPerStar: iface.l_ref_um / detail.um_per_px,
          yRoiTop: iface.y_roi_top,
        }
      : null;
  const boundary: QcInterfaceFrame | null =
    iface?.frames.find((f) => f.camera_frame === frame) ?? null;

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
            {geo && showBoundary && !boundary && " · no boundary this frame"}
          </span>
        </div>

        <div className="lightbox-view">
          <div className="lightbox-frame">
            <ArtifactImage
              src={artifactUrl.datasetFrame(detail.id, frame)}
              alt={`Frame ${frame} of ${detail.id} at ${timeMs.toFixed(1)} milliseconds`}
            />
            {geo && showBoundary && boundary && (
              <svg
                className="frame-overlay"
                viewBox={`0 0 ${geo.width} ${geo.height}`}
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                <path
                  d={boundary.rings
                    .map((ring) => ringPath(ring, geo))
                    .join(" ")}
                />
              </svg>
            )}
          </div>
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
          {geo && (
            <Button
              variant={showBoundary ? "primary" : undefined}
              aria-pressed={showBoundary}
              onClick={() => setShowBoundary((on) => !on)}
              title="Overlay the detected bubble boundary on the frame"
            >
              Boundary
            </Button>
          )}
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
