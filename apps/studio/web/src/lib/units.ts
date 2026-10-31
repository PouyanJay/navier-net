/** Number formatting with stated precision; nulls stay blank. */

export function fmt(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "";
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude >= 1e5 || magnitude < 1e-3) return value.toExponential(Math.max(0, digits - 1));
  const decimals = Math.max(0, digits - 1 - Math.floor(Math.log10(magnitude)));
  return value.toFixed(Math.min(decimals, 6));
}

export function withUnit(value: number | null | undefined, unit: string, digits = 3): string {
  const text = fmt(value, digits);
  return text && unit ? `${text} ${unit}` : text;
}

/** A tidy axis range: symmetric about zero for signed quantities. */
export function niceRange(min: number, max: number, signed: boolean): [number, number] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (signed) {
    const m = Math.max(Math.abs(min), Math.abs(max)) || 1;
    return [-m, m];
  }
  if (min === max) return [min - 0.5, max + 0.5];
  return [min, max];
}

/** Percentile clipping of a plane (ignores NaN); robust ranges for maps. */
export function percentileRange(plane: Float32Array, lo = 0.01, hi = 0.99): [number, number] {
  const finite: number[] = [];
  for (let i = 0; i < plane.length; i += Math.max(1, Math.floor(plane.length / 50000))) {
    const v = plane[i];
    if (Number.isFinite(v)) finite.push(v);
  }
  if (finite.length === 0) return [0, 1];
  finite.sort((a, b) => a - b);
  const at = (p: number) =>
    finite[Math.min(finite.length - 1, Math.max(0, Math.floor(p * (finite.length - 1))))];
  return [at(lo), at(hi)];
}
