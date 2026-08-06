/** Where the objective's weight actually sits, at the last recorded step.
 *
 * The loss curves answer "did it converge"; this answers "which term is the
 * optimiser spending itself on", which is a different question and the one a
 * re-weighting decision is made from. Ten curves inside three decades cannot be
 * read for that — the ordering is the point, so it is a ranked bar.
 */

export interface ResidualBar {
  key: string;
  value: number;
  /** Which family the term belongs to; sets the hue. */
  family: "core" | "interface" | "extra";
  /** Whether the previous chart already showed it, drawn ghosted so what is
   * NEW here reads without a legend saying so. */
  known?: boolean;
}

const HUE = {
  core: "var(--series-1)",
  interface: "var(--series-4)",
  extra: "var(--series-2)",
} as const;

const ROW = 17;
const PAD = { left: 92, right: 54, top: 10, bottom: 22 };

interface ResidualBarsProps {
  bars: ResidualBar[];
  ariaLabel: string;
  width?: number;
}

export function ResidualBars({
  bars,
  ariaLabel,
  width = 620,
}: ResidualBarsProps) {
  const positive = bars.filter((bar) => bar.value > 0);
  if (positive.length === 0) return null;

  const height = PAD.top + positive.length * ROW + PAD.bottom;
  const values = positive.map((bar) => Math.log10(bar.value));
  const hi = Math.max(...values) + 0.35;
  const lo = Math.min(...values) - 0.35;
  const span = width - PAD.left - PAD.right;
  // Log scale: these terms live three or four decades apart, and a linear bar
  // would render everything but the largest as an invisible sliver.
  const length = (value: number) =>
    Math.max(2, ((Math.log10(value) - lo) / (hi - lo)) * span);

  return (
    <svg
      className="resbars"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={ariaLabel}
    >
      {positive.map((bar, index) => {
        const y = PAD.top + index * ROW;
        const w = length(bar.value);
        return (
          <g key={bar.key}>
            <text
              x={PAD.left - 8}
              y={y + 9}
              textAnchor="end"
              className={bar.known ? "resbar-key known" : "resbar-key"}
            >
              {bar.key}
            </text>
            <rect
              x={PAD.left}
              y={y}
              width={w}
              height={11}
              rx={2}
              fill={HUE[bar.family]}
              opacity={bar.known ? 0.32 : 1}
            />
            <text x={PAD.left + w + 7} y={y + 9} className="resbar-val">
              {bar.value.toExponential(1)}
            </text>
          </g>
        );
      })}
      <text x={PAD.left} y={height - 6} className="resbar-foot">
        final residual · log scale · solid = not on the curve chart before
      </text>
    </svg>
  );
}
