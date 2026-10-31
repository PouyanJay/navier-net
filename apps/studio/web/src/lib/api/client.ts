/** Typed client for the studio service, with a small in-memory cache so
 * scrubbing frames never refetches a grid it already holds. */

import type {
  Agreement,
  Audit,
  EdgeResponse,
  FluxBudget,
  GapBudget,
  Grid,
  GridHeader,
  HeatBudget,
  InterfaceContour,
  ProbeRequest,
  ProbeResponse,
  Quantity,
  RunDetail,
  RunSummary,
  Transform,
} from "./types";

const MAGIC = "NVST";
const MAGIC_BYTES = 4;
const LENGTH_BYTES = 4;
const CACHE_LIMIT = 400;

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

const cache = new Map<string, Promise<unknown>>();

function remember<T>(key: string, load: () => Promise<T>): Promise<T> {
  const hit = cache.get(key);
  if (hit) return hit as Promise<T>;
  const pending = load().catch((error) => {
    cache.delete(key);
    throw error;
  });
  cache.set(key, pending);
  while (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value as string);
  return pending;
}

async function detail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    return response.statusText;
  }
}

async function getJson<T>(url: string): Promise<T> {
  return remember(url, async () => {
    const response = await fetch(url);
    if (!response.ok) throw new ApiError(response.status, await detail(response));
    return (await response.json()) as T;
  });
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const key = `${url}#${JSON.stringify(body)}`;
  return remember(key, async () => {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new ApiError(response.status, await detail(response));
    return (await response.json()) as T;
  });
}

/** Decode a packed grid: magic, header length, JSON header, float32 planes. */
export function decodeGrid(buffer: ArrayBuffer): Grid {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.subarray(0, MAGIC_BYTES));
  if (magic !== MAGIC) throw new Error(`not a studio grid (magic ${magic})`);
  const view = new DataView(buffer);
  const length = view.getUint32(MAGIC_BYTES, true);
  const start = MAGIC_BYTES + LENGTH_BYTES;
  const header = JSON.parse(
    new TextDecoder().decode(bytes.subarray(start, start + length)),
  ) as GridHeader;
  const [ny, nx] = header.shape;
  const planeBytes = ny * nx * Float32Array.BYTES_PER_ELEMENT;
  const planes = header.quantities.map((_, i) => {
    const offset = start + length + i * planeBytes;
    // Copy into an aligned buffer: the header length need not be a multiple of 4.
    return new Float32Array(buffer.slice(offset, offset + planeBytes));
  });
  return { header, planes };
}

const run = (id: string) => `/api/runs/${encodeURIComponent(id)}`;

export const api = {
  listRuns: () => getJson<RunSummary[]>("/api/runs"),
  getRun: (id: string) => getJson<RunDetail>(run(id)),
  getCatalogue: (id: string) => getJson<Quantity[]>(`/api/catalogue?run=${encodeURIComponent(id)}`),
  getAudit: (id: string) => getJson<Audit>(`${run(id)}/artifacts/audit`),
  getArtifact: <T>(id: string, name: string) => getJson<T>(`${run(id)}/artifacts/${name}`),
  getInterface: (id: string, frame: number) =>
    getJson<InterfaceContour>(`${run(id)}/interface?frame=${frame}`),
  getTransform: (id: string) => getJson<Transform>(`${run(id)}/geometry/transform`),
  getAgreement: (id: string, frame: number) =>
    getJson<Agreement>(`${run(id)}/geometry/agreement?frame=${frame}`),
  frameUrl: (id: string, frame: number, clean = false) =>
    `${run(id)}/geometry/frame/${frame}.png?crop=true&clean=${clean}`,
  getGrid: (id: string, quantities: string[], frame: number, nx: number, ny: number) => {
    const url = `${run(id)}/grid?q=${quantities.join(",")}&frame=${frame}&nx=${nx}&ny=${ny}`;
    return remember(url, async () => {
      const response = await fetch(url);
      if (!response.ok) throw new ApiError(response.status, await detail(response));
      return decodeGrid(await response.arrayBuffer());
    });
  },
  gridCsvUrl: (id: string, quantities: string[], frame: number, nx: number, ny: number) =>
    `${run(id)}/grid?q=${quantities.join(",")}&frame=${frame}&nx=${nx}&ny=${ny}&fmt=csv`,
  probe: (id: string, request: ProbeRequest) =>
    postJson<ProbeResponse>(`${run(id)}/probe`, request),
  getEdge: (id: string, quantities: string[], frame: number) =>
    getJson<EdgeResponse>(`${run(id)}/edge?q=${quantities.join(",")}&frame=${frame}`),
  edgeCsvUrl: (id: string, quantities: string[], frame: number) =>
    `${run(id)}/edge?q=${quantities.join(",")}&frame=${frame}&fmt=csv`,
  getGapBudget: (id: string, frame: number, side: "upper" | "lower") =>
    getJson<GapBudget>(`${run(id)}/budgets/gap?frame=${frame}&side=${side}`),
  getFluxBudget: (id: string, frame: number) =>
    getJson<FluxBudget>(`${run(id)}/budgets/flux?frame=${frame}`),
  getHeatBudget: (id: string, frame: number) =>
    getJson<HeatBudget>(`${run(id)}/budgets/heat?frame=${frame}`),
  budgetCsvUrl: (id: string, kind: "gap" | "flux" | "heat", frame: number) =>
    `${run(id)}/budgets/${kind}?frame=${frame}&fmt=csv`,
};

export async function postCsv(url: string, body: unknown): Promise<string> {
  const response = await fetch(`${url}${url.includes("?") ? "&" : "?"}fmt=csv`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new ApiError(response.status, await detail(response));
  return response.text();
}

export const probeUrl = (id: string) => `${run(id)}/probe`;
