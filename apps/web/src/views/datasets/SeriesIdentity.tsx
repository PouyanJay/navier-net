import { StageHeader } from "../../app/StageHeader";
import type { DatasetDetail, DatasetSummary } from "../../lib/api";
import { seriesName } from "../../lib/series";

/**
 * What the stage is looking at, in one line.
 *
 * This replaces an `<h1>` that repeated the rail's own label and a paragraph
 * explaining the stage — about 140px above the fold, identical on every visit,
 * saying nothing about the series in view. Every token here is about this
 * series: its shape, its fluid, and the state that decides whether it can be
 * trained on.
 */
export function SeriesIdentity({
  series,
  detail,
  trained,
}: {
  series: DatasetSummary | null;
  detail: DatasetDetail | null;
  trained: boolean;
}) {
  if (!series) return null;

  const [w, h] = series.frame_px ?? [];
  const shape = w && h ? (w === h ? `${w}²` : `${w}×${h}`) : null;
  const excluded = detail?.excluded_frames.length ?? 0;
  const heldOut = detail?.holdout_frame != null;
  const dt = series.dt_frame_ms;

  return (
    <StageHeader>
      <div className="ident">
        <h1 className="ident-name">{seriesName(series)}</h1>
        <IdentFact>{series.n_frames} frames</IdentFact>
        {shape && <IdentFact>{shape}</IdentFact>}
        {dt != null && <IdentFact>Δt {dt} ms</IdentFact>}
        {detail?.conditions?.fluid && (
          <IdentFact>{detail.conditions.fluid}</IdentFact>
        )}

        <span className="ident-state">
          {series.processed ? (
            <span className="chip" data-tone={trained ? "green" : undefined}>
              {trained ? "trained" : "preprocessed"}
            </span>
          ) : (
            <span className="chip" data-tone="amber">
              not preprocessed
            </span>
          )}
          {/* Both are the reason a frame is absent from training, and they are
              different reasons, so they are never merged into one count. */}
          {heldOut && (
            <span className="chip" data-tone="amber">
              1 held out
            </span>
          )}
          {excluded > 0 && (
            <span className="chip">
              {excluded} excluded
              {detail?.exclusions_applied === false ? " · needs re-run" : ""}
            </span>
          )}
        </span>
      </div>
    </StageHeader>
  );
}

/** One mono fact, preceded by the separator that joins it to the last. */
function IdentFact({ children }: { children: React.ReactNode }) {
  return (
    <>
      <span className="ident-sep" aria-hidden="true">
        ·
      </span>
      <span className="ident-fact">{children}</span>
    </>
  );
}
