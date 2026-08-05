import type { DatasetDetail } from "../../lib/api";

/**
 * Every frame in the series, as one tick each, in a bar that never scrolls.
 *
 * The thumbnail strip below has to scroll: `n_frames` is data, and a long
 * acquisition is hundreds of frames, so nothing that shows each one at a
 * readable size can also fit. What scrolling costs is the shape of the
 * sequence — which frames are excluded, where the holdout sits, whether the
 * problems cluster — and that is exactly what a reader needs before deciding
 * anything. So the overview stays put and the detail scrolls beneath it.
 *
 * Ticks flex to the width available, so this holds at 12 frames and at 240.
 */
export function FrameRibbon({
  detail,
  onSelect,
}: {
  detail: DatasetDetail;
  /** Scrolls the thumbnail strip to the clicked frame. */
  onSelect?: (frame: number) => void;
}) {
  if (detail.n_frames === 0) return null;

  const frames = Array.from({ length: detail.n_frames }, (_, i) => i + 1);
  const excluded = new Set(detail.excluded_frames);

  return (
    <div
      className="ribbon"
      role="img"
      aria-label={ribbonSummary(detail)}
      title={ribbonSummary(detail)}
    >
      {frames.map((n) => {
        const role =
          n === detail.holdout_frame
            ? "held"
            : excluded.has(n)
              ? "excl"
              : "train";
        return (
          <button
            key={n}
            type="button"
            className={`ribbon-tick ${role}`}
            aria-hidden="true"
            tabIndex={-1}
            onClick={() => onSelect?.(n)}
          />
        );
      })}
    </div>
  );
}

/**
 * The ribbon's text alternative. It is decorative pixel-by-pixel, so the whole
 * bar carries one honest summary rather than 240 unreadable tick labels — the
 * frames themselves are reachable in the strip below.
 */
function ribbonSummary(detail: DatasetDetail): string {
  const parts = [`${detail.n_frames} frames`];
  if (detail.holdout_frame != null)
    parts.push(`frame ${detail.holdout_frame} held out`);
  if (detail.excluded_frames.length > 0)
    parts.push(
      `${detail.excluded_frames.length} excluded (${detail.excluded_frames.join(", ")})`,
    );
  return parts.join(" · ");
}
