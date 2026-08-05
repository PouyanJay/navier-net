/**
 * The NavierNet mark: a vortex drawn as a phyllotactic spiral.
 *
 * The geometry is generated from the golden angle rather than hand-placed, so
 * the spacing is principled and the mark can be re-derived at any point count
 * instead of being re-drawn. Everything is `currentColor`, so the mark tints
 * with whatever ink it sits in — chrome, a disabled control, a favicon — and
 * needs no per-theme swap.
 */

/** One point of the spiral, in the mark's 24×24 user space. */
export interface MarkPoint {
  cx: number;
  cy: number;
  r: number;
  opacity: number;
}

/** The mark's user-space box. Every consumer scales this, never re-measures it. */
export const MARK_VIEWBOX = 24;

const CENTRE = MARK_VIEWBOX / 2;
/** π(3−√5): the divergence angle that makes a phyllotactic spiral fill evenly. */
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
const CORE_RADIUS = 1.5;
const ARM_GROWTH = 0.83;
const CORE_DOT = 2.15;
const DOT_TAPER = 0.1;
const FADE = 0.055;

/**
 * The spiral, from the dense bright core outward to the faint decaying tail.
 *
 * Dots shrink and fade as they travel out, which is what reads as rotation even
 * when the mark is standing still — the direction it fails in is the one where
 * only the animation carries it.
 */
export function vorticityPoints(count = 11): MarkPoint[] {
  return Array.from({ length: count }, (_, i) => {
    const angle = i * GOLDEN_ANGLE;
    const radius = CORE_RADIUS + i * ARM_GROWTH;
    return {
      cx: CENTRE + radius * Math.cos(angle),
      cy: CENTRE + radius * Math.sin(angle),
      r: Math.max(0.4, CORE_DOT - i * DOT_TAPER),
      opacity: Math.max(0.15, 1 - i * FADE),
    };
  });
}

export interface BrandMarkProps {
  /**
   * Turns the mark into the platform's working indicator. Reserved for real
   * work — a solver run, a preprocess job — never the resting chrome.
   */
  working?: boolean;
  /** Accessible name. Omit for a decorative mark beside a visible wordmark. */
  title?: string;
  className?: string;
}

export function BrandMark({ working, title, className }: BrandMarkProps) {
  const classes = ["brandmark", working ? "working" : "", className]
    .filter(Boolean)
    .join(" ");

  return (
    <svg
      className={classes}
      viewBox={`0 0 ${MARK_VIEWBOX} ${MARK_VIEWBOX}`}
      role={title ? "img" : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
      focusable="false"
    >
      {/* One group, so the spin has a single transform origin at the core. */}
      <g className="brandmark-spiral">
        {vorticityPoints().map((point, i) => (
          <circle
            key={i}
            cx={point.cx}
            cy={point.cy}
            r={point.r}
            fill="currentColor"
            opacity={point.opacity}
          />
        ))}
      </g>
    </svg>
  );
}
