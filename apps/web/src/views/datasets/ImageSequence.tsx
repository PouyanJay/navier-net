import { useEffect, useRef, useState, type RefObject } from "react";

import { Button, Panel } from "../../components";
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
  "That is the holdout frame — the run's only unsupervised check on the model. " +
  "Move the holdout in the run configuration before excluding it.";

/** How much of a horizontal scroller is visible, and how far along it sits.
 * `visible` is 1 when everything fits — the caller hides the indicator then. */
function useScrollExtent(
  strip: RefObject<HTMLDivElement | null>,
  itemCount: number,
) {
  const [extent, setExtent] = useState({ visible: 1, progress: 0 });

  useEffect(() => {
    const element = strip.current;
    if (!element) return;
    const measure = () => {
      const scrollable = element.scrollWidth - element.clientWidth;
      setExtent({
        visible:
          element.scrollWidth > 0
            ? element.clientWidth / element.scrollWidth
            : 1,
        progress: scrollable > 0 ? element.scrollLeft / scrollable : 0,
      });
    };
    measure();
    element.addEventListener("scroll", measure, { passive: true });
    const observer = new ResizeObserver(measure); // panel resize changes the fit
    observer.observe(element);
    return () => {
      element.removeEventListener("scroll", measure);
      observer.disconnect();
    };
  }, [strip, itemCount]);

  return extent;
}

function calibrationLine(detail: DatasetDetail): string {
  if (detail.um_per_px == null || detail.frame_px == null) {
    return `${detail.n_frames} raw frame${detail.n_frames === 1 ? "" : "s"} — preprocess to calibrate`;
  }
  const fovMm = (detail.um_per_px * detail.frame_px[0]) / 1000;
  return `auto-calibrated · ${detail.um_per_px.toFixed(3)} µm px⁻¹ · FOV ${fovMm.toFixed(2)} mm`;
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
  const { visible, progress } = useScrollExtent(strip, detail.n_frames);

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
  const showPreprocess =
    detail.n_frames > 0 && (!detail.processed || pendingRerun);

  return (
    <Panel
      title={`Image sequence — ${detail.id}`}
      subtitle={calibrationLine(detail)}
      actions={
        showPreprocess ? (
          <Button variant="primary" onClick={onPreprocess} disabled={running}>
            {running
              ? "Preprocessing…"
              : pendingRerun
                ? "Re-run preprocessing"
                : "Run preprocessing"}
          </Button>
        ) : undefined
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
            className="strip"
            role="list"
            aria-label={`${detail.n_frames} raw frames`}
          >
            {frames.map((n) => {
              const holdout = n === detail.holdout_frame;
              const excluded = detail.excluded_frames.includes(n);
              const label = `Frame ${n}`;
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
                    aria-label={`${label} — ${excluded ? "excluded from" : "included in"} training${holdout ? ", holdout frame" : ""}`}
                    onClick={() => handleClick(n, holdout)}
                    onDoubleClick={() => handleDoubleClick(n)}
                  >
                    <ArtifactImage
                      src={artifactUrl.datasetFrame(detail.id, n)}
                      alt={`${label} preview`}
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
                    aria-label={`Enlarge ${label.toLowerCase()}`}
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
          {visible < 1 && (
            <div className="strip-bar" aria-hidden="true">
              <i
                style={{
                  width: `${visible * 100}%`,
                  left: `${progress * (100 - visible * 100)}%`,
                }}
              />
            </div>
          )}
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
        <p className="state-note caution" role="status">
          {blocked}
        </p>
      )}
      {exclusionError && (
        <p className="state-note error" role="alert">
          Could not change the excluded frames: {exclusionError}
        </p>
      )}
      {pendingRerun && !running && (
        <p className="state-note caution" role="status">
          The excluded frames have changed since the tensors were built — re-run
          preprocessing so the solver sees this set.
        </p>
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
