/** Colormaps for the field maps, tuned for the always-dark view canvas.
 * Sequential ramps start at the canvas color so "zero" recedes; the diverging
 * map keeps a dark neutral midpoint (never a hue) per the dataviz rules. */

export type ColormapName = "blue" | "thermal" | "div" | "amber";

const STOPS: Record<ColormapName, number[][]> = {
  blue: [
    [13, 15, 21],
    [30, 58, 138],
    [37, 99, 235],
    [147, 197, 253],
    [224, 242, 254],
  ],
  thermal: [
    [13, 15, 21],
    [80, 7, 36],
    [180, 83, 9],
    [251, 191, 36],
    [254, 243, 199],
  ],
  div: [
    [96, 165, 250],
    [27, 30, 40],
    [251, 113, 133],
  ],
  amber: [
    [13, 15, 21],
    [120, 53, 15],
    [251, 191, 36],
  ],
};

/** RGB at t ∈ [0, 1]. */
export function cmap(name: ColormapName, t: number): [number, number, number] {
  const stops = STOPS[name];
  const clamped = Math.max(0, Math.min(1, t));
  const x = clamped * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  return [0, 1, 2].map((c) =>
    Math.round(stops[i][c] + (stops[i + 1][c] - stops[i][c]) * f),
  ) as [number, number, number];
}

/** The same ramp as a CSS gradient, for the colorbar. */
export function cssGradient(name: ColormapName): string {
  const samples = [0, 0.25, 0.5, 0.75, 1]
    .map((t) => `rgb(${cmap(name, t).join(",")})`)
    .join(",");
  return `linear-gradient(90deg, ${samples})`;
}
