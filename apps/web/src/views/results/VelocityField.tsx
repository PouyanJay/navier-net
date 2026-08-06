import { useMemo, useState } from "react";

import type { VelocityFieldMap } from "../../lib/api";
import { cmap } from "./fieldColormaps";

/**
 * The inferred velocity field: a quiver over the interface it flows around.
 *
 * This is the platform's strongest single claim, so the drawing is built to
 * carry it rather than to look busy. Three decisions do that work:
 *
 * 1. Length encodes speed and colour encodes it again. Redundant on purpose:
 *    the arrows nearest the wall are short enough that their ink is the only
 *    thing a reader can judge them by, and a length-only quiver reads as noise
 *    wherever the flow is slow.
 * 2. The front is drawn ON TOP, filled. An arrow field without the boundary it
 *    is flowing around cannot be read at all, and the vapour side is where the
 *    velocity means something different in kind.
 * 3. The scale is stated. A quiver with no reference arrow is a picture of a
 *    direction field; with one it is a measurement.
 */

const PAD = { left: 34, right: 16, top: 26, bottom: 30 };
/** Floor on the colour ramp: the slowest arrows must still be visible against
 * the canvas, or the field looks empty exactly where it is stagnant. */
const INK_FLOOR = 0.42;

/** Arrow length as a fraction of the lattice pitch, at the fastest vector. */
const MAX_ARROW = 1.35;
/** Below this fraction of the peak, an arrow is a dot: drawing a head on a
 * two-pixel shaft makes stagnant regions look like fast ones. */
const STILL = 0.06;

interface VelocityFieldProps {
  field: VelocityFieldMap;
  /** viewBox width; the aspect follows the channel. */
  width?: number;
  ariaLabel: string;
}

interface Arrow {
  x: number;
  y: number;
  dx: number;
  dy: number;
  speed: number;
  u: number;
  v: number;
  xUm: number;
  yUm: number;
}

export function VelocityField({
  field,
  width = 720,
  ariaLabel,
}: VelocityFieldProps) {
  const [hover, setHover] = useState<{
    arrow: Arrow;
    clientX: number;
    clientY: number;
  } | null>(null);

  const [x0, x1, y0, y1] = field.domain_um;
  const plotW = width - PAD.left - PAD.right;
  // The plot's aspect IS the channel's. Stretching it would rotate every
  // vector on the page: a 45° arrow in a 2×-tall drawing is not 45° in the
  // channel, and direction is most of what a quiver says.
  const plotH = plotW * ((y1 - y0) / (x1 - x0));
  const height = PAD.top + PAD.bottom + plotH;
  const sx = (xUm: number) => PAD.left + ((xUm - x0) / (x1 - x0)) * plotW;
  const sy = (yUm: number) => PAD.top + ((yUm - y0) / (y1 - y0)) * plotH;

  const { arrows, pitch } = useMemo(() => {
    const columns = field.x_um.length;
    const step = columns > 1 ? plotW / (columns - 1) : plotW;
    const peak = field.speed_max || 1;
    const list: Arrow[] = [];
    field.y_um.forEach((yUm, row) => {
      field.x_um.forEach((xUm, column) => {
        const u = field.u[row][column];
        const v = field.v[row][column];
        const speed = Math.hypot(u, v);
        // Scaled by the LATTICE, not by the value: arrows that overlap their
        // neighbours stop being a field and become a mess of overlapping ink.
        const length = (speed / peak) * step * MAX_ARROW;
        const norm = speed || 1;
        list.push({
          x: sx(xUm),
          y: sy(yUm),
          dx: (u / norm) * length,
          dy: (v / norm) * length,
          speed,
          u,
          v,
          xUm,
          yUm,
        });
      });
    });
    return { arrows: list, pitch: step };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [field, plotW, plotH]);

  const peak = field.speed_max || 1;
  const ink = (speed: number) => {
    const [r, g, b] = cmap(
      "blue",
      INK_FLOOR + (1 - INK_FLOOR) * (speed / peak),
    );
    return `rgb(${r} ${g} ${b})`;
  };

  return (
    <div className="vfield-wrap">
      {hover && (
        <div
          className="chart-tip iou-tip"
          style={{ left: hover.clientX + 12, top: hover.clientY + 12 }}
          role="status"
        >
          x {hover.arrow.xUm.toFixed(0)} µm · y {hover.arrow.yUm.toFixed(0)} µm
          · <b>{hover.arrow.speed.toFixed(1)}</b> {field.unit}
          <span className="vfield-tip-uv">
            u {hover.arrow.u.toFixed(1)} · v {hover.arrow.v.toFixed(1)}
          </span>
        </div>
      )}
      <svg
        className="vfield"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={ariaLabel}
      >
        <defs>
          <clipPath id="vfield-clip">
            <rect x={PAD.left} y={PAD.top} width={plotW} height={plotH} />
          </clipPath>
        </defs>

        {/* The channel: the walls the flow is bounded by. */}
        <rect
          x={PAD.left}
          y={PAD.top}
          width={plotW}
          height={plotH}
          className="vfield-channel"
        />

        <g clipPath="url(#vfield-clip)">
          {/* The vapour region, under the arrows: inside it the velocity is the
              vapour's, which is a different quantity from the liquid's. */}
          {field.interface.map((line, index) => (
            <path
              key={`fill-${index}`}
              className="vfield-vapour"
              d={`M ${line.map(([x, y]) => `${sx(x).toFixed(1)} ${sy(y).toFixed(1)}`).join(" L ")} Z`}
            />
          ))}

          {arrows.map((arrow, index) => {
            const fraction = arrow.speed / peak;
            if (fraction < STILL) {
              return (
                <circle
                  key={index}
                  cx={arrow.x}
                  cy={arrow.y}
                  r={1.1}
                  className="vfield-still"
                />
              );
            }
            const tipX = arrow.x + arrow.dx;
            const tipY = arrow.y + arrow.dy;
            const length = Math.hypot(arrow.dx, arrow.dy);
            const head = Math.min(4.6, Math.max(2.2, length * 0.34));
            const ux = arrow.dx / length;
            const uy = arrow.dy / length;
            // A triangular head drawn from the shaft's own direction: a marker
            // would not rotate with it, and a stroked chevron would thicken
            // with the shaft and swamp the short arrows.
            const points = [
              `${tipX.toFixed(1)},${tipY.toFixed(1)}`,
              `${(tipX - ux * head - uy * head * 0.55).toFixed(1)},${(tipY - uy * head + ux * head * 0.55).toFixed(1)}`,
              `${(tipX - ux * head + uy * head * 0.55).toFixed(1)},${(tipY - uy * head - ux * head * 0.55).toFixed(1)}`,
            ].join(" ");
            return (
              <g
                key={index}
                className={
                  hover?.arrow === arrow ? "vfield-arrow hot" : "vfield-arrow"
                }
                onPointerEnter={(event) =>
                  setHover({
                    arrow,
                    clientX: event.clientX,
                    clientY: event.clientY,
                  })
                }
                onPointerLeave={() => setHover(null)}
              >
                <line
                  x1={arrow.x}
                  y1={arrow.y}
                  x2={(tipX - ux * head * 0.75).toFixed(1)}
                  y2={(tipY - uy * head * 0.75).toFixed(1)}
                  stroke={ink(arrow.speed)}
                  strokeWidth={1 + 0.9 * fraction}
                  strokeLinecap="round"
                />
                <polygon points={points} fill={ink(arrow.speed)} />
                {/* A generous invisible target: the arrows are 2px of ink. */}
                <circle
                  cx={arrow.x + arrow.dx / 2}
                  cy={arrow.y + arrow.dy / 2}
                  r={Math.max(5, pitch * 0.45)}
                  fill="transparent"
                />
              </g>
            );
          })}

          {/* The front itself, over everything: the boundary the field is
              flowing around is what makes the field legible. */}
          {field.interface.map((line, index) => (
            <path
              key={`edge-${index}`}
              className="vfield-front"
              d={`M ${line.map(([x, y]) => `${sx(x).toFixed(1)} ${sy(y).toFixed(1)}`).join(" L ")} Z`}
            />
          ))}
        </g>

        {/* The scale, stated: without it this is a direction field. */}
        <g transform={`translate(${PAD.left}, ${height - 10})`}>
          <line
            x1={0}
            y1={-4}
            x2={pitch * MAX_ARROW}
            y2={-4}
            className="vfield-ref"
          />
          <polygon
            points={`${(pitch * MAX_ARROW).toFixed(1)},-4 ${(pitch * MAX_ARROW - 4).toFixed(1)},-6.2 ${(pitch * MAX_ARROW - 4).toFixed(1)},-1.8`}
            className="vfield-ref-head"
          />
          <text x={pitch * MAX_ARROW + 8} y={-1} className="vfield-label">
            {peak.toFixed(0)} {field.unit} · peak
          </text>
        </g>
        <text x={PAD.left} y={14} className="vfield-label">
          inferred velocity · no velocity data was ever supplied
        </text>
        <text
          x={width - PAD.right}
          y={14}
          textAnchor="end"
          className="vfield-label"
        >
          t = {field.t_ms.toFixed(2)} ms
        </text>
        <text
          x={width - PAD.right}
          y={height - 11}
          textAnchor="end"
          className="vfield-label"
        >
          mean {field.speed_mean.toFixed(0)} {field.unit}
        </text>
      </svg>
    </div>
  );
}
