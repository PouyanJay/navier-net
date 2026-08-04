import { useEffect, useMemo, useState } from "react";

import type { FrontArrow, InterfaceData, InterfaceFrame } from "../lib/api";

const PLAYBACK_FPS = 14;

/** How long the fastest arrow is drawn, as a fraction of the domain's width.
 * Every arrow is scaled against the fastest one IN THE SAME FRAME, so the
 * overlay shows how the speed is distributed around the front at that instant.
 * A fixed scale would leave the early frames invisible. */
const ARROW_SPAN = 0.06;

/** Arrows shorter than this fraction of the longest are dropped rather than
 * drawn as specks that read like noise on the contour. Unrelated to
 * `ARROW_SPAN` above -- that one is a fraction of the DOMAIN, this one a
 * fraction of the frame's fastest ARROW -- so the two move independently. */
const ARROW_FLOOR = 0.05;

const toPath = (contour: number[][], flipY: (y: number) => number) =>
  contour
    .map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x} ${flipY(y)}`)
    .join(" ");

function nearestByTime(
  frames: InterfaceFrame[],
  tMs: number,
): InterfaceFrame | null {
  if (frames.length === 0) return null;
  return frames.reduce((best, frame) =>
    Math.abs(frame.t_ms - tMs) < Math.abs(best.t_ms - tMs) ? frame : best,
  );
}

function noseLength(frame: InterfaceFrame, xPin: number): number | null {
  const xs = frame.contours.flatMap((contour) => contour.map(([x]) => x));
  return xs.length > 0 ? Math.max(...xs) - xPin : null;
}

/**
 * One arrow's line, scaled so the frame's fastest point spans `ARROW_SPAN`.
 *
 * The arrow is `v * n`: the NORMAL component of the front's motion. That is
 * everything two masks can say about a generic interface point -- a curve
 * sliding along itself looks identical between frames -- and it is also the only
 * component that changes the shape. The viewport labels it so no one reads these
 * as material velocities.
 */
function arrowLine(
  [x, y, nx, ny, v]: FrontArrow,
  fastest: number,
  span: number,
  flipY: (y: number) => number,
) {
  const length = (v / fastest) * span;
  return {
    x1: x,
    y1: flipY(y),
    x2: x + nx * length,
    // The arrow is a vector in the same space as the point it starts from, so
    // it flips with it -- adding the raw component would point it the wrong way
    // on a y-inverted axis.
    y2: flipY(y + ny * length),
  };
}

interface ViewportProps {
  data: InterfaceData;
}

/**
 * The reconstruction viewport: the model's continuous interface, animated
 * between camera instants, with the measured contour as a dashed overlay.
 * All geometry is real (served contours in µm); playback is user-initiated.
 */
export function ReconstructionViewport({ data }: ViewportProps) {
  const [playing, setPlaying] = useState(false);
  const [index, setIndex] = useState(0);
  const [showInterface, setShowInterface] = useState(true);
  const [showMeasured, setShowMeasured] = useState(true);
  // Off by default: the arrows annotate the contour, and a reader who came for
  // the shape should not have to turn them off first.
  const [showVelocity, setShowVelocity] = useState(false);

  const total = data.frames.length;
  const frame = data.frames[Math.min(index, total - 1)];
  const lastT = data.frames[total - 1]?.t_ms ?? 0;

  useEffect(() => {
    if (!playing || total === 0) return;
    const id = window.setInterval(
      () => setIndex((i) => (i + 1) % total),
      1000 / PLAYBACK_FPS,
    );
    return () => window.clearInterval(id);
  }, [playing, total]);

  const [x0, x1] = data.domain.x_um;
  const [y0, y1] = data.domain.y_um;
  const width = x1 - x0;
  const height = y1 - y0;
  const flipY = useMemo(() => (y: number) => y1 + y0 - y, [y0, y1]);
  const measured = showMeasured
    ? nearestByTime(data.measured, frame?.t_ms ?? 0)
    : null;
  const length = frame ? noseLength(frame, data.domain.x_pin_um) : null;
  const gridLines = useMemo(
    () => Array.from({ length: 23 }, (_, i) => x0 + ((i + 1) * width) / 24),
    [x0, width],
  );

  const arrows = frame?.front ?? null;
  const fastest = useMemo(
    () => Math.max(...(arrows ?? []).map(([, , , , v]) => Math.abs(v)), 1e-9),
    [arrows],
  );

  if (!frame) return null;

  return (
    <div className="viewport">
      <div className="vhd">
        <span className="vt">Reconstruction viewport · flow →</span>
        <div className="layerbtns" role="group" aria-label="Layers">
          <button
            type="button"
            className="lbtn"
            aria-pressed={showInterface}
            onClick={() => setShowInterface((on) => !on)}
          >
            interface
          </button>
          <button
            type="button"
            className="lbtn"
            aria-pressed={showMeasured}
            onClick={() => setShowMeasured((on) => !on)}
          >
            measured
          </button>
          <button
            type="button"
            className="lbtn"
            aria-pressed={showVelocity}
            // aria-disabled, not disabled: a disabled button drops out of the
            // tab order, taking the reason it is unavailable with it.
            aria-disabled={arrows == null}
            aria-label={
              arrows == null
                ? "velocity — unavailable: this run has no explicit front (model.front_geometry), so it has no per-point interface velocity"
                : "velocity — normal component of the front's motion; the tangential component is unobservable from masks"
            }
            onClick={() => arrows && setShowVelocity((on) => !on)}
          >
            velocity
          </button>
        </div>
        <span className="hudright">
          {showVelocity && arrows && (
            // Visible, not just a tooltip: an arrow that looks like a velocity
            // must say which velocity it is, and carry the scale that tells a
            // reader what its length is worth.
            <span>
              normal component · longest {fastest.toPrecision(3)} µm/ms ·{" "}
            </span>
          )}
          t = {frame.t_ms.toFixed(2)} ms
          {length != null && ` · L = ${Math.round(length)} µm`}
        </span>
      </div>
      <svg
        viewBox={`${x0} ${y0} ${width} ${height}`}
        style={{ aspectRatio: `${width} / ${height}` }}
        role="img"
        aria-label="Reconstructed vapor interface inside the microchannel over time."
      >
        <defs>
          <marker
            id="vp-arrowhead"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="5"
            markerHeight="5"
            markerUnits="strokeWidth"
            orient="auto"
          >
            <path className="vp-arrowhead" d="M0 1.5 L8 4 L0 6.5 z" />
          </marker>
        </defs>
        <g className="vp-grid">
          {gridLines.map((x) => (
            <line key={x} x1={x} x2={x} y1={y0} y2={y1} />
          ))}
        </g>
        <line className="vp-wall" x1={x0} x2={x1} y1={y0 + 0.5} y2={y0 + 0.5} />
        <line className="vp-wall" x1={x0} x2={x1} y1={y1 - 0.5} y2={y1 - 0.5} />
        {measured?.contours.map((contour, i) => (
          <path
            key={`m${i}`}
            className="vp-measured"
            d={toPath(contour, flipY)}
          />
        ))}
        {showInterface &&
          frame.contours.map((contour, i) => (
            <path key={i} className="vp-interface" d={toPath(contour, flipY)} />
          ))}
        {showVelocity && arrows && (
          <g
            className="vp-velocity"
            role="img"
            aria-label={
              `Normal velocity at ${arrows.length} points around the front; ` +
              `the longest arrow is ${fastest.toPrecision(3)} µm per ms. ` +
              "Only the normal component is observable from masks."
            }
          >
            {arrows.map((arrow, i) =>
              Math.abs(arrow[4]) / fastest < ARROW_FLOOR ? null : (
                <line
                  key={i}
                  markerEnd="url(#vp-arrowhead)"
                  {...arrowLine(arrow, fastest, width * ARROW_SPAN, flipY)}
                />
              ),
            )}
          </g>
        )}
        <circle
          className="vp-cavity"
          cx={data.domain.x_pin_um}
          cy={flipY(y0 + height * 0.08)}
          r={height * 0.03}
        />
        <text
          className="vp-label"
          x={data.domain.x_pin_um}
          y={flipY(y0 + height * 0.16)}
        >
          nucleation cavity · pinned
        </text>
        <text
          className="vp-label"
          x={x0 + width * 0.01}
          y={flipY(y1 - height * 0.12)}
        >
          inlet
        </text>
        <text
          className="vp-label vp-label-end"
          x={x1 - width * 0.01}
          y={flipY(y1 - height * 0.12)}
        >
          outlet →
        </text>
      </svg>
      <div className="vctl">
        <button
          type="button"
          className="vp-play"
          onClick={() => setPlaying((on) => !on)}
          aria-label={playing ? "Pause playback" : "Play reconstruction"}
        >
          <svg viewBox="0 0 12 12" aria-hidden="true">
            {playing ? (
              <path d="M2.5 1.5h2.6v9H2.5zM6.9 1.5h2.6v9H6.9z" />
            ) : (
              <path d="M3 1.5l7 4.5-7 4.5z" />
            )}
          </svg>
        </button>
        <input
          type="range"
          min={0}
          max={total - 1}
          value={Math.min(index, total - 1)}
          onChange={(event) => {
            setPlaying(false);
            setIndex(Number(event.target.value));
          }}
          aria-label="Scrub reconstruction time"
        />
        <span className="tval">
          t {frame.t_ms.toFixed(2)} / {lastT.toFixed(2)} ms
        </span>
      </div>
    </div>
  );
}
