/** The studio's shareable state lives in the URL hash as a query string. */

export type UrlState = Record<string, string>;

export function readHash(hash: string = location.hash): UrlState {
  const params = new URLSearchParams(hash.replace(/^#\??/, ""));
  const out: UrlState = {};
  params.forEach((value, key) => {
    out[key] = value;
  });
  return out;
}

export function writeHash(state: UrlState): void {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(state)) if (value !== "") params.set(key, value);
  const next = `#${params.toString()}`;
  if (next !== location.hash) history.replaceState(null, "", next);
}

export function parseFrames(text: string | undefined): number[] {
  if (!text) return [];
  return text
    .split(",")
    .map((v) => Number.parseInt(v, 10))
    .filter((v) => Number.isFinite(v));
}
