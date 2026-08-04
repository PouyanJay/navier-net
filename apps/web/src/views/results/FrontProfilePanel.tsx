import { useEffect, useMemo, useRef, useState } from "react";

import { Panel } from "../../components";
import {
  CompareChart,
  SERIES_SLOT,
  type CompareBand,
  type ComparePoint,
} from "../../components/charts/CompareChart";
import type { FrontProfile, FrontSegment } from "../../lib/api";
import { ChartCard } from "./ChartCard";
import { cmap, cssGradient } from "./fieldColormaps";

/** Human labels for the four segments of the closed front, in traversal order. */
const SEGMENT_LABEL: Record<FrontSegment["name"], string> = {
  root_cap: "root cap",
  upper_body: "upper body",
  nose_cap: "nose cap",
  lower_body: "lower body",
};

/** The kymograph is signed — a receding front reads negative — so it takes the
 * diverging ramp around a neutral zero, never a sequential one. */
const KYMOGRAPH_CMAP = "div" as const;

function bandsOf(segments: FrontSegment[]): CompareBand[] {
  return segments.map((segment) => ({
    start: segment.s_start,
    end: segment.s_end,
    label: SEGMENT_LABEL[segment.name],
    muted: !segment.measured,
  }));
}

/**
 * The (position, time) speed field as RGBA pixels, symmetric about zero.
 *
 * The report is indexed [time][position] but the image runs time along x and
 * position along y, so the two are transposed here -- and that transposition is
 * why this is a pure function with its own test rather than arithmetic buried
 * in a canvas call jsdom cannot execute.
 *
 * A null cell is transparent, not zero: zero is a perfectly good speed and
 * would read as "the front stood still here" rather than "nothing was computed".
 */
export function kymographImage(rows: (number | null)[][], limit: number) {
  const width = rows.length; // time
  const height = rows[0]?.length ?? 0; // position along the front
  const data = new Uint8ClampedArray(width * height * 4);
  rows.forEach((row, t) =>
    row.forEach((value, s) => {
      const offset = (s * width + t) * 4;
      const [r, g, b] = cmap(KYMOGRAPH_CMAP, (value ?? 0) / (2 * limit) + 0.5);
      data[offset] = r;
      data[offset + 1] = g;
      data[offset + 2] = b;
      data[offset + 3] = value == null ? 0 : 255;
    }),
  );
  return { width, height, data };
}

function paintKymograph(
  canvas: HTMLCanvasElement,
  rows: (number | null)[][],
  limit: number,
) {
  const context = canvas.getContext("2d");
  if (!context) return; // jsdom
  const { width, height, data } = kymographImage(rows, limit);
  if (!width || !height) return;
  canvas.width = width;
  canvas.height = height;
  context.putImageData(new ImageData(data, width, height), 0, 0);
}

function Kymograph({ profile }: { profile: FrontProfile }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const rows = profile.kymograph.v_um_per_ms;
  const limit = useMemo(() => {
    const largest = Math.max(
      ...rows.flatMap((row) =>
        row.map((value) => (value == null ? 0 : Math.abs(value))),
      ),
      1e-6,
    );
    return largest;
  }, [rows]);

  useEffect(() => {
    if (ref.current) paintKymograph(ref.current, rows, limit);
  }, [rows, limit]);

  return (
    <div className="kymo">
      <canvas
        ref={ref}
        className="kymo-canvas"
        role="img"
        aria-label={
          "Kymograph: the model's normal speed along the front (vertical) " +
          "against time (horizontal). Warm is outward, cool is inward."
        }
      />
      <div className="kymo-scale">
        <span className="kin-unit">−{limit.toPrecision(3)}</span>
        <span
          className="kymo-ramp"
          style={{ background: cssGradient(KYMOGRAPH_CMAP) }}
          aria-hidden="true"
        />
        <span className="kin-unit">+{limit.toPrecision(3)} µm/ms</span>
      </div>
    </div>
  );
}

interface FrontProfilePanelProps {
  profile: FrontProfile;
  /** File stem for the charts' downloads. */
  stem: string;
  /** Formats a speed for the readouts (both units). */
  readout: (value: number) => string;
}

/**
 * How fast each point of the interface is moving -- the answer to "not just the
 * nose, the whole front".
 *
 * Only the NORMAL component appears here, and that is not a limitation of the
 * method but of the data: a curve sliding along itself looks identical between
 * frames, so the tangential component is unobservable from masks. It is also
 * the only component that matters for the shape, which tangential motion does
 * not change at all.
 */
export function FrontProfilePanel({
  profile,
  stem,
  readout,
}: FrontProfilePanelProps) {
  const [index, setIndex] = useState(0);
  const frames = profile.times;
  const frame = frames[Math.min(index, frames.length - 1)];
  const bands = useMemo(() => bandsOf(profile.segments), [profile.segments]);
  const noseCap = profile.segments.find((s) => !s.measured);

  const series = (values: (number | null)[]): ComparePoint[] =>
    profile.s.map((s, i) => ({ x: s, y: values[i] ?? null }));

  return (
    <>
      <Panel
        title="Normal speed along the front"
        subtitle="every point of the interface, not just the nose"
      >
        {frames.length === 0 ? (
          <p className="state-note">
            No two camera frames of this series are consecutive, so no interface
            velocity can be differenced from the masks. The model's own profile
            is in the kymograph below.
          </p>
        ) : (
          <>
            <ChartCard
              title={`Frames ${frame.frames[0]}–${frame.frames[1]}`}
              unit="µm/ms"
              name={`${stem}-profile-${frame.frames[0]}-${frame.frames[1]}`}
              rows={profile.s.map((s, i) => ({
                s,
                model_um_per_ms: frame.model[i],
                measured_um_per_ms: frame.measured[i],
              }))}
              render={() => (
                <CompareChart
                  series={[
                    {
                      id: "model",
                      points: series(frame.model),
                      slot: SERIES_SLOT.primary,
                    },
                    {
                      id: "measured",
                      points: series(frame.measured),
                      slot: frame.heldout
                        ? SERIES_SLOT.heldout
                        : SERIES_SLOT.measured,
                    },
                  ]}
                  bands={bands}
                  xLabel="s"
                  yLabel="normal speed · µm/ms"
                  ariaLabel={
                    `Normal speed along the front between camera frames ` +
                    `${frame.frames[0]} and ${frame.frames[1]}, model against ` +
                    "the masks, walking once around the closed interface."
                  }
                  yFormat={readout}
                />
              )}
            />
            <div className="field-ctl">
              <label htmlFor="fv-profile-time">Frame pair</label>
              <input
                id="fv-profile-time"
                type="range"
                min={0}
                max={frames.length - 1}
                value={Math.min(index, frames.length - 1)}
                onChange={(event) => setIndex(Number(event.target.value))}
              />
              <span className="field-t">
                {frame.t_ms} ms{frame.heldout ? " · held out" : ""}
              </span>
            </div>
          </>
        )}
        <p className="figcap">
          <b>Figure 2.</b> <i>s</i> walks once around the closed interface: root
          cap, up the upper body to the nose, around the nose cap, back down the
          lower body. These are <b>normal</b> speeds — a curve sliding along
          itself looks identical between frames, so the tangential component is
          unobservable from masks, and it is also the one component that does
          not change the shape.
          {noseCap && (
            <>
              {" "}
              The measurement is deliberately absent across the{" "}
              <b>{SEGMENT_LABEL[noseCap.name]}</b>: the level-set estimate is
              first-order in the distance the front travels, and there that
              distance is largest against the smallest radius of curvature. The
              model's own curve continues through it — that disagreement is a
              real finding, not a rendering gap.
            </>
          )}
        </p>
      </Panel>

      <Panel
        title="Kymograph"
        subtitle="the whole history of the front's motion, at once"
      >
        <Kymograph profile={profile} />
        <p className="figcap">
          <b>Figure 3.</b> The model's normal speed over position (vertical, the
          same <i>s</i> as above) against time (horizontal). Model only: the
          parameterisation is continuous in time while the measurement exists
          only at frame pairs, and stacking nine measured rows beside a dense
          model field would invite a comparison the sampling cannot support.
        </p>
      </Panel>
    </>
  );
}
