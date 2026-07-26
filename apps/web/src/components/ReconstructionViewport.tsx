import { useEffect, useMemo, useState } from "react";

import type { InterfaceData, InterfaceFrame } from "../lib/api";

const PLAYBACK_FPS = 14;

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
        </div>
        <span className="hudright">
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
