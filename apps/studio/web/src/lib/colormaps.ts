/** Colormaps as functions of t in [0, 1] and as 256-entry RGBA lookup tables.
 *
 * The perceptual maps (viridis, inferno, magma, plasma) are the degree-6
 * polynomial fits by Matt Zucker (CC0), accurate to well under one 8-bit step.
 * Diverging maps are built around a neutral midpoint that follows the canvas:
 * black on the dark canvas, white on the light one, so zero recedes.
 */

export type RGB = [number, number, number];

export type ColormapName =
  "viridis" | "inferno" | "magma" | "plasma" | "cividis" | "grey" | "diverging" | "diverging-light";

export const LUT_SIZE = 256;

const clamp01 = (t: number) => Math.min(1, Math.max(0, t));

function poly(t: number, c: number[][]): RGB {
  const x = clamp01(t);
  const out: RGB = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    let v = 0;
    for (let i = c.length - 1; i >= 0; i--) v = v * x + c[i][k];
    out[k] = clamp01(v);
  }
  return out;
}

const VIRIDIS = [
  [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
  [0.1050930431085774, 1.404613529898575, 1.384590162594685],
  [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
  [-4.634230498983486, -5.799100973351585, -19.33244095627987],
  [6.228269936347081, 14.17993336680509, 56.69055260068105],
  [4.776384997670288, -13.74514537774601, -65.35303263337234],
  [-5.435455855934631, 4.645852612178535, 26.3124352495832],
];
const INFERNO = [
  [0.0002189403691192265, 0.001651004631001012, -0.01948089843709184],
  [0.1065134194856116, 0.5639564367884091, 3.932712388889277],
  [11.60249308247187, -3.972853965665698, -15.9423941062914],
  [-41.70399613139459, 17.43639888205313, 44.35414519872813],
  [77.162935699427, -33.40235894210092, -81.80730925738993],
  [-71.31942824499214, 32.62606426397723, 73.20951985803202],
  [25.13112622477341, -12.24266895238567, -23.07032500287172],
];
const MAGMA = [
  [-0.002136485053939582, -0.000749655052795221, -0.005386127855323933],
  [0.2516605407371642, 0.6775232436837668, 2.494026599312351],
  [8.353717279216625, -3.577719514958484, 0.3144679030132573],
  [-27.66873308576866, 14.26473078096533, -13.64921318813922],
  [52.17613981234068, -27.94360607168351, 12.94416944238394],
  [-50.76852536473588, 29.04658282127291, 4.23415299384598],
  [18.65570506591883, -11.48977351997711, -5.601961508734096],
];
const PLASMA = [
  [0.05873234392399702, 0.02333670892565664, 0.5433401826748754],
  [2.176514634195958, 0.2383834171260182, 0.7539604599784036],
  [-2.689460476458034, -7.455851135738909, 3.110799939717086],
  [6.130348345893603, 42.3461881477227, -28.51885465332158],
  [-11.10743619062271, -82.66631109428045, 60.13984767418263],
  [10.02306557647065, 71.41361770095349, -54.07218655560067],
  [-3.658713842777788, -22.93153465461149, 18.19190778539828],
];

/** Cividis, from its published control points, linearly interpolated. */
const CIVIDIS: RGB[] = [
  [0.0, 0.135, 0.304],
  [0.0, 0.17, 0.372],
  [0.133, 0.218, 0.411],
  [0.247, 0.272, 0.41],
  [0.337, 0.325, 0.413],
  [0.414, 0.379, 0.428],
  [0.487, 0.434, 0.448],
  [0.561, 0.49, 0.464],
  [0.637, 0.548, 0.469],
  [0.717, 0.607, 0.46],
  [0.8, 0.669, 0.432],
  [0.887, 0.734, 0.375],
  [0.974, 0.804, 0.283],
  [1.0, 0.86, 0.19],
];

function ramp(t: number, stops: RGB[]): RGB {
  const x = clamp01(t) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

const DIVERGING_DARK: RGB[] = [
  [0.4, 0.75, 1.0],
  [0.16, 0.42, 0.85],
  [0.0, 0.0, 0.0],
  [0.85, 0.25, 0.3],
  [1.0, 0.72, 0.55],
];
const DIVERGING_LIGHT: RGB[] = [
  [0.1, 0.28, 0.65],
  [0.45, 0.65, 0.92],
  [0.97, 0.97, 0.97],
  [0.95, 0.55, 0.5],
  [0.7, 0.08, 0.15],
];

export function colormap(name: ColormapName, t: number): RGB {
  switch (name) {
    case "viridis":
      return poly(t, VIRIDIS);
    case "inferno":
      return poly(t, INFERNO);
    case "magma":
      return poly(t, MAGMA);
    case "plasma":
      return poly(t, PLASMA);
    case "cividis":
      return ramp(t, CIVIDIS);
    case "grey": {
      const v = 0.06 + 0.9 * clamp01(t);
      return [v, v, v];
    }
    case "diverging":
      return ramp(t, DIVERGING_DARK);
    case "diverging-light":
      return ramp(t, DIVERGING_LIGHT);
  }
}

/** A 256 x RGBA8 table, ready to upload as a texture. */
export function buildLut(name: ColormapName): Uint8Array {
  const out = new Uint8Array(LUT_SIZE * 4);
  for (let i = 0; i < LUT_SIZE; i++) {
    const [r, g, b] = colormap(name, i / (LUT_SIZE - 1));
    out[i * 4] = Math.round(r * 255);
    out[i * 4 + 1] = Math.round(g * 255);
    out[i * 4 + 2] = Math.round(b * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

export function cssGradient(name: ColormapName, steps = 16): string {
  const stops: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const [r, g, b] = colormap(name, i / steps);
    stops.push(
      `rgb(${Math.round(r * 255)} ${Math.round(g * 255)} ${Math.round(b * 255)}) ${(100 * i) / steps}%`,
    );
  }
  return `linear-gradient(90deg, ${stops.join(", ")})`;
}

export const SEQUENTIAL: ColormapName[] = [
  "viridis",
  "inferno",
  "magma",
  "plasma",
  "cividis",
  "grey",
];

/** The default map for a catalogue family on a theme. */
export function defaultColormap(
  family: "sequential" | "diverging" | "thermal" | "magma",
  theme: "dark" | "light",
): ColormapName {
  if (family === "diverging") return theme === "dark" ? "diverging" : "diverging-light";
  if (family === "thermal") return "inferno";
  if (family === "magma") return "magma";
  return "viridis";
}

export function luminance([r, g, b]: RGB): number {
  const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

/** The boundary must read on every ramp: the stroke takes the tone the map
 * lacks around its middle and carries a halo of the opposite tone. */
export function boundaryTone(name: ColormapName): "light" | "dark" {
  const mid = luminance(colormap(name, 0.5));
  const hi = luminance(colormap(name, 0.85));
  return (mid + hi) / 2 > 0.45 ? "dark" : "light";
}
