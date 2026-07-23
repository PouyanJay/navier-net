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

export interface RunMetrics {
  iou_per_frame?: Record<string, number>;
  iou_mean?: number;
  iou_holdout?: number | null;
  holdout_frame?: number | null;
  nose_speed_mm_s?: number;
  dataset?: string;
  run_name?: string;
}

export interface RunDetail {
  id: string;
  dataset: string | null;
  status: "trained" | "empty";
  steps: number | null;
  metrics: RunMetrics | null;
  config: Record<string, unknown> | null;
  artifacts: ArtifactFlags;
}

export interface PhysicsValidation {
  nose_speed_inferred_mm_s: number | null;
  nose_speed_measured_mm_s: number | null;
  nose_speed_error_pct: number | null;
  bretherton_film_um: number | null;
  hele_shaw: number | null;
  reynolds: number | null;
  weber: number | null;
  capillary: number | null;
  prandtl: number | null;
  iou_mean: number | null;
  iou_holdout: number | null;
  holdout_frame: number | null;
}

export interface OperatingConditions {
  fluid: string;
  T_sat_C: number;
  q_wall_W_cm2: number;
  flow_rate_mL_hr: number;
  channel_width_um: number;
  channel_height_um: number;
  dt_frame_ms: number;
  flow_direction: string;
  n_frames_raw: number;
  n_frames_usable: number;
  n_frames_event: number;
}

export interface DatasetSummary {
  id: string;
  n_frames: number;
  processed: boolean;
}

export interface DatasetDetail extends DatasetSummary {
  has_qc: boolean;
  conditions: OperatingConditions;
}

export type DimensionlessGroups = Record<string, number>;

export interface PreprocessStatus {
  dataset: string;
  state: "idle" | "running" | "done" | "error";
  message: string | null;
  has_qc: boolean;
}

export interface ModelArchitecture {
  fields: string[];
  hidden: number;
  layers: number;
  fourier_feats: number;
  fourier_scale: number;
  alpha_eps: number;
  nodewise_activation: boolean;
}

/** Initial loss-term weights for a run (`cfg.training.weights`). */
export interface LossWeightsInput {
  data: number;
  vof: number;
  div: number;
  src: number;
  bc: number;
}

/** A request to start (or resume) a training run. Mirrors the backend model. */
export interface RunLaunchRequest {
  dataset?: string | null;
  resume?: boolean;
  run_id?: string | null;
  steps: number;
  lr: number;
  lr_halflife: number;
  n_data: number;
  n_coll: number;
  n_bc: number;
  holdout_frame: number;
  rebalance_every: number;
  log_every: number;
  weights: LossWeightsInput;
  render: boolean;
}

export interface RunJobStatus {
  run_id: string;
  dataset: string | null;
  state: "running" | "done" | "error";
  stage: string | null;
  message: string | null;
  steps_done: number;
  steps_total: number;
}

/** One per-log-step loss record, streamed live over SSE (`hist` events). */
export interface LossRecord {
  step: number;
  lr: number;
  data: number;
  vof: number;
  div: number;
  src: number;
  bc: number;
}

/** One solver-console line (`log` events). */
export interface ConsoleLine {
  line: string;
  tone: "ok" | "em" | "dim" | "err" | null;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new Error(`GET ${path} failed: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

async function send<T>(path: string, method: string, body?: BodyInit): Promise<T> {
  const response = await fetch(path, { method, body });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      detail = ((await response.json()) as { detail?: string }).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

async function sendJson<T>(path: string, method: string, payload: unknown): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      detail = ((await response.json()) as { detail?: string }).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

const runPath = (id: string) => `/api/runs/${encodeURIComponent(id)}`;

const datasetPath = (id: string) => `/api/datasets/${encodeURIComponent(id)}`;

export const api = {
  listRuns: () => getJson<RunSummary[]>("/api/runs"),
  getRun: (id: string) => getJson<RunDetail>(runPath(id)),
  getValidation: (id: string) => getJson<PhysicsValidation>(`${runPath(id)}/validation`),

  startRun: (request: RunLaunchRequest) =>
    sendJson<RunJobStatus>("/api/runs", "POST", request),
  getRunStatus: (id: string) => getJson<RunJobStatus>(`${runPath(id)}/status`),
  getActiveRun: () => getJson<RunJobStatus | null>("/api/runs/active"),

  listDatasets: () => getJson<DatasetSummary[]>("/api/datasets"),
  getDataset: (id: string) => getJson<DatasetDetail>(datasetPath(id)),
  getDatasetGroups: (id: string) => getJson<DimensionlessGroups>(`${datasetPath(id)}/groups`),
  getPreprocessStatus: (id: string) =>
    getJson<PreprocessStatus>(`${datasetPath(id)}/preprocess`),
  startPreprocess: (id: string) =>
    send<PreprocessStatus>(`${datasetPath(id)}/preprocess`, "POST"),
  uploadFrames: (id: string, files: FileList | File[]) => {
    const form = new FormData();
    for (const file of Array.from(files)) form.append("files", file);
    return send<DatasetSummary>(`${datasetPath(id)}/upload`, "POST", form);
  },

  getModel: (id: string) => getJson<ModelArchitecture>(`/api/model/${encodeURIComponent(id)}`),
};

/** Direct artifact URLs (used as href / img src / video src, not fetched as JSON). */
export const artifactUrl = {
  figure: (id: string, name: string) =>
    `${runPath(id)}/figures/${encodeURIComponent(name)}`,
  video: (id: string) => `${runPath(id)}/video`,
  checkpoint: (id: string) => `${runPath(id)}/checkpoint`,
  tensors: (id: string) => `${runPath(id)}/tensors`,
  datasetQc: (id: string) => `${datasetPath(id)}/qc`,
  datasetFrame: (id: string, n: number) => `${datasetPath(id)}/frames/${n}`,
};
