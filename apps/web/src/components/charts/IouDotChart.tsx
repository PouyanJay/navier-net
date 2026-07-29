import { useState } from "react";

/** Per-frame IoU as positioned dots: honest about its zoomed axis (labeled
 * range), with each frame's validation role encoded by color AND geometry
 * (holdout = open ring); hovering a dot reads out frame, IoU, and role. */

export type IouRole = "supervised" | "validation" | "holdout" | "transfer";

export interface IouFramePoint {
  /** 1-based camera frame. */
  frame: number;
  iou: number;
  role: IouRole;
}

interface IouDotChartProps {
  frames: IouFramePoint[];
  mean: number | null;
  ariaLabel: string;
  /** viewBox width — wider when the chart has a row to itself. */
  width?: number;
}

const HEIGHT = 152;
const PAD = { left: 46, right: 14, top: 20, bottom: 28 };
const TICKS = [0.85, 0.9, 0.95, 1.0];

export function IouDotChart({
  frames,
  mean,
  ariaLabel,
  width = 340,
}: IouDotChartProps) {
  const [hover, setHover] = useState<{
    index: number;
    clientX: number;
    clientY: number;
  } | null>(null);
  if (frames.length === 0) return null;

  const yMin = Math.min(0.84, ...frames.map((p) => p.iou - 0.02));
  const yMax = 1.004;
  const x = (i: number) =>
    PAD.left +
    (frames.length === 1
      ? (width - PAD.left - PAD.right) / 2
      : (i / (frames.length - 1)) * (width - PAD.left - PAD.right));
  const y = (v: number) =>
    HEIGHT -
    PAD.bottom -
    ((v - yMin) / (yMax - yMin)) * (HEIGHT - PAD.top - PAD.bottom);

  const hovered = hover ? frames[hover.index] : null;

  return (
    <div className="iou-dots-wrap">
      {hovered && hover && (
        <div
          className="chart-tip iou-tip"
          style={{ left: hover.clientX + 12, top: hover.clientY + 12 }}
          role="status"
        >
          f{String(hovered.frame).padStart(2, "0")} · IoU{" "}
          <b>{hovered.iou.toFixed(3)}</b> · {hovered.role}
        </div>
      )}
      <svg
        viewBox={`0 0 ${width} ${HEIGHT}`}
        role="img"
        aria-label={ariaLabel}
        className="iou-dots"
      >
        {TICKS.filter((t) => t >= yMin).map((t) => (
          <g key={t}>
            <line
              x1={PAD.left}
              x2={width - PAD.right}
              y1={y(t)}
              y2={y(t)}
              className="iou-grid"
            />
            <text
              x={PAD.left - 8}
              y={y(t) + 3}
              textAnchor="end"
              className="iou-tick"
            >
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        <line
          x1={PAD.left}
          x2={width - PAD.right}
          y1={HEIGHT - PAD.bottom}
          y2={HEIGHT - PAD.bottom}
          className="iou-axis"
        />
        {mean != null && (
          <>
            <line
              x1={PAD.left}
              x2={width - PAD.right}
              y1={y(mean)}
              y2={y(mean)}
              className="iou-mean"
            />
            <text x={PAD.left + 4} y={y(mean) - 5} className="iou-tick">
              mean {mean.toFixed(3)}
            </text>
          </>
        )}
        <text x={PAD.left} y={12} className="iou-tick iou-caption">
          IoU · axis {yMin.toFixed(2)}–1.00
        </text>
        <text
          x={width - PAD.right}
          y={12}
          textAnchor="end"
          className="iou-tick iou-caption"
        >
          camera frame →
        </text>
        {frames.map((point, i) => (
          <g key={point.frame}>
            <circle
              cx={x(i)}
              cy={y(point.iou)}
              r={hover?.index === i ? 6 : 4.5}
              className={
                point.role === "holdout"
                  ? "iou-dot hold"
                  : `iou-dot ${point.role}`
              }
              onPointerMove={(event) =>
                setHover({
                  index: i,
                  clientX: event.clientX,
                  clientY: event.clientY,
                })
              }
              onPointerLeave={() => setHover(null)}
            >
              <title>{`f${String(point.frame).padStart(2, "0")} · IoU ${point.iou.toFixed(3)} · ${point.role === "holdout" ? "holdout, never supervised" : point.role}`}</title>
            </circle>
            {point.role === "holdout" && (
              <text
                x={x(i)}
                y={y(point.iou) - 9}
                textAnchor="middle"
                className="iou-holdout-tag"
              >
                HOLDOUT
              </text>
            )}
            <text
              x={x(i)}
              y={HEIGHT - 8}
              textAnchor="middle"
              className="iou-tick"
            >
              {point.frame}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
