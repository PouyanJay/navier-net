import { useEffect, useId, useRef, useState } from "react";

import {
  Button,
  Callout,
  Panel,
  ScrollBar,
  useScrollExtent,
  useWheelToHorizontal,
} from "../../components";
import { ArtifactImage } from "../../components/ArtifactImage";
import {
  artifactUrl,
  type DatasetDetail,
  type PreprocessStatus,
} from "../../lib/api";
import { FrameLightbox } from "./FrameLightbox";

interface ImageSequenceProps {
  detail: DatasetDetail;
  preprocess: PreprocessStatus | null;
  onPreprocess: () => void;
  onToggleExcluded: (frame: number) => void;
  exclusionError: string | null;
}

/** A double-click also delivers two clicks; hold the exclusion toggle for this
 * long so opening the lightbox does not silently flip the frame on the way. */
const DOUBLE_CLICK_GRACE_MS = 220;

/** The API refuses to drop the holdout; say why here rather than round-tripping
 * to an error the user could have been told about up front. */
const HOLDOUT_BLOCKED =
  "That is the holdout frame, the run's only unsupervised check on the model. " +
  "Move the holdout in the run configuration before excluding it.";

function calibrationLine(detail: DatasetDetail): string {
  if (detail.um_per_px == null || detail.frame_px == null) {
    const plural = detail.n_frames === 1 ? "" : "s";
    return `${detail.n_frames} raw frame${plural} · preprocess to calibrate`;
  }
  const fovMm = (detail.um_per_px * detail.frame_px[0]) / 1000;
  return `auto-calibrated · ${detail.um_per_px.toFixed(3)} µm px⁻¹ · FOV ${fovMm.toFixed(2)} mm`;
}

/** Circular arrow: this rebuilds an artifact that already exists. */
function RerunIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path
        d="M13.2 8a5.2 5.2 0 1 1-1.6-3.7"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <path
        d="M13.4 1.9v3.1h-3.1"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** The raw frames as an inline film strip; scrolls when it overflows. Clicking a
 * frame excludes it from the tensors the model trains on, double-clicking (or
 * the expand control) opens it full size. The panel also owns (re)running
 * preprocessing, which is what actually applies an exclusion. */
export function ImageSequence({
  detail,
  preprocess,
  onPreprocess,
  onToggleExcluded,
  exclusionError,
}: ImageSequenceProps) {
  const frames = Array.from({ length: detail.n_frames }, (_, i) => i + 1);
  const [zoomed, setZoomed] = useState<number | null>(null);
  const [blocked, setBlocked] = useState<string | null>(null);
  const pendingToggle = useRef<number | undefined>(undefined);
  const strip = useRef<HTMLDivElement>(null);
  const stripId = useId();
  const extent = useScrollExtent(strip, detail.n_frames);
  useWheelToHorizontal(strip);

  // A pending single-click must not fire after the strip is gone.
  useEffect(() => () => window.clearTimeout(pendingToggle.current), []);

  function handleClick(frame: number, holdout: boolean) {
    window.clearTimeout(pendingToggle.current);
    if (holdout) {
      setBlocked(HOLDOUT_BLOCKED);
      return;
    }
    setBlocked(null);
    pendingToggle.current = window.setTimeout(
      () => onToggleExcluded(frame),
      DOUBLE_CLICK_GRACE_MS,
    );
  }

  function handleDoubleClick(frame: number) {
    window.clearTimeout(pendingToggle.current); // this was an enlarge, not a toggle
    setZoomed(frame);
  }

  const running = preprocess?.state === "running";
  const excludedCount = detail.excluded_frames.length;
  const pendingRerun = detail.processed && !detail.exclusions_applied;

  return (
    <Panel
      title={`Image sequence · ${detail.id}`}
      subtitle={calibrationLine(detail)}
      actions={
        <PreprocessAction
          {...{ detail, running, pendingRerun, onPreprocess }}
        />
      }
    >
      {detail.n_frames === 0 ? (
        <p className="state-note">
          Upload the first high-speed sequence to begin calibration and
          segmentation.
        </p>
      ) : (
        <>
          <div
            ref={strip}
            id={stripId}
            className="strip"
            role="list"
            aria-label={`${detail.n_frames} raw frames`}
          >
            {frames.map((n) => {
              const holdout = n === detail.holdout_frame;
              const excluded = detail.excluded_frames.includes(n);
              const state = excluded ? "excluded from" : "included in";
              return (
                <figure
                  key={n}
                  className={[
                    "fr",
                    holdout ? "hold" : "",
                    excluded ? "excl" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  role="listitem"
                >
                  <button
                    type="button"
                    className="fr-toggle"
                    aria-pressed={excluded}
                    title={holdout ? HOLDOUT_BLOCKED : undefined}
                    aria-label={`Frame ${n}, ${state} training${holdout ? ", holdout frame" : ""}`}
                    onClick={() => handleClick(n, holdout)}
                    onDoubleClick={() => handleDoubleClick(n)}
                  >
                    <ArtifactImage
                      src={artifactUrl.datasetFrame(detail.id, n)}
                      alt={`Frame ${n} preview`}
                      loading="lazy"
                    />
                    <figcaption className="mono">
                      f{String(n).padStart(2, "0")} ·{" "}
                      {((n - 1) * detail.conditions.dt_frame_ms).toFixed(1)} ms
                    </figcaption>
                  </button>
                  <button
                    type="button"
                    className="fr-zoom mono"
                    aria-label={`Enlarge frame ${n}`}
                    onClick={() => setZoomed(n)}
                  >
                    ⤢
                  </button>
                  {(holdout || excluded) && (
                    <span
                      className={excluded ? "fr-tag mono excl" : "fr-tag mono"}
                    >
                      {excluded ? "EXCLUDED" : "HOLDOUT"}
                    </span>
                  )}
                </figure>
              );
            })}
          </div>
          <ScrollBar
            target={strip}
            extent={extent}
            controls={stripId}
            label="Scroll the frame strip"
          />
          <p className="fr-hint">
            Click a frame to exclude it from training · double-click to enlarge
            {excludedCount > 0 && (
              <>
                {" · "}
                <b className="mono">
                  {excludedCount} excluded ({detail.excluded_frames.join(", ")})
                </b>
              </>
            )}
          </p>
        </>
      )}

      {blocked && (
        <Callout tone="caution" title="Holdout frame">
          {blocked}
        </Callout>
      )}
      {exclusionError && (
        <Callout tone="error" title="Could not change the excluded frames">
          {exclusionError}
        </Callout>
      )}
      {pendingRerun && !running && (
        <Callout tone="caution" title="Preprocessing is out of date">
          The excluded frames have changed since the tensors were built. Re-run
          preprocessing so the solver sees this set.
        </Callout>
      )}
      {running && (
        <div className="modal-progress">
          <div
            className="meter indeterminate"
            role="progressbar"
            aria-label="Preprocessing"
          >
            <i />
          </div>
          <p className="state-note" role="status">
            Preprocessing: calibrating, segmenting, building tensors…
          </p>
        </div>
      )}
      {preprocess?.state === "error" && preprocess.message && (
        <Callout tone="error" title="Preprocessing failed">
          {preprocess.message}
        </Callout>
      )}
      {detail.notes && (
        <p className="note">
          <b>Auto-detected</b> {detail.notes}
        </p>
      )}

      {zoomed !== null && (
        <FrameLightbox
          detail={detail}
          frame={zoomed}
          onFrameChange={setZoomed}
          onToggleExcluded={onToggleExcluded}
          onClose={() => setZoomed(null)}
        />
      )}
    </Panel>
  );
}

interface PreprocessActionProps {
  detail: DatasetDetail;
  running: boolean;
  pendingRerun: boolean;
  onPreprocess: () => void;
}

/** The first run is this panel's primary call to action and stays full size; a
 * re-run is a small corrective action, so it does not outweigh the panel title. */
function PreprocessAction({
  detail,
  running,
  pendingRerun,
  onPreprocess,
}: PreprocessActionProps) {
  if (detail.n_frames === 0 || (detail.processed && !pendingRerun)) return null;

  if (pendingRerun) {
    return (
      <Button
        size="sm"
        onClick={onPreprocess}
        disabled={running}
        title="Rebuild the tensors so the excluded frames take effect"
      >
        <RerunIcon />
        {running ? "Running…" : "Re-run"}
      </Button>
    );
  }
  return (
    <Button variant="primary" onClick={onPreprocess} disabled={running}>
      {running ? "Preprocessing…" : "Run preprocessing"}
    </Button>
  );
}
