/** Typed client for the naviernet API. Types mirror the backend response models. */

export interface RunSummary {
  id: string;
  dataset: string | null;
  status: "trained" | "empty";
  steps: number | null;
  iou_holdout: number | null;
}

export interface ArtifactFlags {
  checkpoint: boolean;
  metrics: boolean;
  groups: boolean;
  video: boolean;
  figures: string[];
}

export interface RunDetail {
  id: string;
  dataset: string | null;
  status: "trained" | "empty";
  steps: number | null;
  metrics: Record<string, unknown> | null;
  config: Record<string, unknown> | null;
  artifacts: ArtifactFlags;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new Error(`GET ${path} failed: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export const api = {
  listRuns: () => getJson<RunSummary[]>("/api/runs"),
  getRun: (id: string) => getJson<RunDetail>(`/api/runs/${encodeURIComponent(id)}`),
};
