/** Typed client for the naviernet API. Types mirror the backend response models. */

export type RunStatus = "running" | "trained" | "failed" | "empty";

export interface RunSummary {
  id: string;
  dataset: string | null;
  status: RunStatus;
  steps: number | null;
  iou_holdout: number | null;
  /** Every dataset the run spans (one for single runs, several for joint). */
  datasets?: string[];
  /** Axis-B conditions held out of training. */
  heldout_datasets?: string[];
  /** Joint runs' in-distribution (axis-A) validation IoU. */
  val_iou_mean?: number | null;
  /** ISO timestamp of the run directory. */
  date?: string | null;
  /** Live progress while the server is training this run. */
  steps_done?: number | null;
  steps_total?: number | null;
}

export interface ArtifactFlags {
  checkpoint: boolean;
  metrics: boolean;
  groups: boolean;
  video: boolean;
  figures: string[];
}

/** A joint run's held-out transfer scores (metrics.json v2). */
export interface TransferMetrics {
  per_dataset?: Record<string, number>;
  mean?: number | null;
  per_frame?: Record<string, Record<string, number>>;
}

export interface RunMetrics {
  iou_per_frame?: Record<string, number>;
  iou_mean?: number;
  iou_holdout?: number | null;
  holdout_frame?: number | null;
  nose_speed_mm_s?: number;
  dataset?: string;
  run_name?: string;
  // Two-axis validation fields.
  iou_val?: number | null; // single-series: in-distribution validation IoU
  validation_frames?: number[];
  training_datasets?: string[];
  heldout_datasets?: string[];
  val_iou_mean?: number | null; // joint: mean of training series' validation IoU
  transfer?: TransferMetrics | null; // held-out series, scored over every frame
}

/** The single generalization number to headline for a run: the single-dataset
 * holdout-frame IoU, or a joint run's in-distribution validation IoU (metrics.json
 * v2 has no `iou_holdout`). Mirrors the backend's `_headline_iou` so every surface
 * shows the same number for the same run. */
export function headlineIou(
  metrics: RunMetrics | null | undefined,
): number | null {
  return metrics?.iou_holdout ?? metrics?.val_iou_mean ?? null;
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

/** One dataset's agreement block inside a joint run's v2 metrics. */
export interface DatasetAgreement {
  iou_mean: number | null;
  iou_val: number | null;
  validation_frames: number[];
  iou_per_frame: Record<string, number>;
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
  // Two-axis validation (metrics.json v2); null/empty on legacy runs.
  val_iou_mean: number | null;
  iou_val: number | null;
  validation_frames: number[];
  transfer_iou_mean: number | null;
  transfer_per_dataset: Record<string, number> | null;
  transfer_per_frame: Record<string, Record<string, number>> | null;
  per_dataset: Record<string, DatasetAgreement> | null;
  training_datasets: string[] | null;
  heldout_datasets: string[] | null;
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
  U_ref_m_s: number | null;
}

export interface ProjectSummary {
  id: string;
  name: string;
  description: string;
  datasets: string[];
  created_at: string;
}

export interface DatasetSummary {
  id: string; // immutable filesystem key
  n_frames: number;
  processed: boolean;
  conditions_set: boolean;
  /** Editable display name; null = show the id. */
  label: string | null;
  frame_px: [number, number] | null;
  dt_frame_ms: number | null;
}

/** A selectable working fluid with its atmospheric T_sat and saturated
 * properties (mirrors the `configs/fluid/<id>.yaml` group). */
export interface Fluid {
  id: string; // config-group id and the fluid= override value, e.g. "fc72"
  name: string; // display label, e.g. "FC-72"
  T_sat_C: number; // saturation temperature at 1 atm
  rho_l: number;
  rho_v: number;
  mu_l: number;
  mu_v: number;
  k_l: number;
  k_v: number;
  cp_l: number;
  cp_v: number;
  sigma: number;
  h_lv: number;
}

/** Editable per-series conditions (subset of OperatingConditions). T_sat is not
 * here: it is derived from the selected fluid, not typed. */
export interface ConditionsUpdate {
  fluid?: string; // fluid config-group id (allow-listed server-side)
  dt_frame_ms?: number;
  channel_width_um?: number;
  channel_height_um?: number;
  flow_rate_mL_hr?: number;
  q_wall_W_cm2?: number;
  U_ref?: number;
}

export interface ConditionsResponse {
  conditions: OperatingConditions;
  groups: DimensionlessGroups;
}

export interface QcKinematics {
  t_ms: number[];
  length_um: number[];
  fit_slope_mm_s: number;
  fit_intercept_um: number;
}

export interface QcInterfaceFrame {
  index: number;
  /** 1-based camera frame this silhouette belongs to, for overlaying it on the
   * raw frame image (not index + 1 once frames are excluded). */
  camera_frame: number;
  t_ms: number;
  /** Closed [x*, y*] rings: the outer silhouette first, then any holes. */
  rings: number[][][];
}

export interface QcData {
  dataset: string;
  n_frames_event: number;
  kinematics: QcKinematics;
  interface: {
    x_pin_star: number;
    x_range: [number, number];
    y_range: [number, number];
    /** x* · l_ref_um = µm; the charts label their axes in physical units. */
    l_ref_um: number;
    /** Top row of the imaged band in raw-frame pixels; rings are cut to this
     * ROI, so a frame overlay shifts them down by this much. */
    y_roi_top: number;
    frames: QcInterfaceFrame[];
  };
  sdf: {
    frame_index: number;
    t_ms: number;
    x_range: [number, number];
    y_range: [number, number];
    values: number[][];
  };
}

export interface DatasetDetail extends DatasetSummary {
  has_qc: boolean;
  conditions: OperatingConditions;
  holdout_frame: number | null;
  um_per_px: number | null;
  notes: string | null;
  /** 1-based camera frames kept out of the tensors the model trains on. */
  excluded_frames: number[];
  /** False when the exclusions above still need a preprocessing re-run. */
  exclusions_applied: boolean;
  /** False when a tensor-baked condition (frame interval, channel width, or
   * reference velocity) was edited and the tensors need a re-preprocess. */
  conditions_applied: boolean;
}

export type DimensionlessGroups = Record<string, number>;

export interface PreprocessStatus {
  dataset: string;
  state: "idle" | "running" | "done" | "error";
  message: string | null;
  has_qc: boolean;
}

/** A per-field width/depth override; unset keys fall back to the globals. */
export interface PerFieldArch {
  hidden?: number | null;
  layers?: number | null;
}

export interface ModelArchitecture {
  fields: string[];
  hidden: number;
  layers: number;
  fourier_feats: number;
  fourier_scale: number;
  alpha_eps: number;
  nodewise_activation: boolean;
  per_field: Record<string, PerFieldArch>;
}

/** A model-architecture edit; any unset global keeps its current value. */
export interface ModelUpdate {
  hidden?: number;
  layers?: number;
  fourier_feats?: number;
  fourier_scale?: number;
  alpha_eps?: number;
  per_field?: Record<string, PerFieldArch>;
}

/** One governing equation as the Physics view shows it (from the registry). */
export interface EquationState {
  id: string;
  name: string;
  stage: "A" | "B";
  tex: string;
  weight_key: string;
  fields_required: string[];
  fields_added: string[];
  groups: string[];
  core: boolean;
  enabled: boolean;
  weight: number;
}

export interface PhysicsState {
  dataset: string;
  equations: EquationState[];
  fields: string[];
  groups: Record<string, number>;
}

/** A physics edit: which toggleable Stage-B equations train, and weight overrides. */
export interface PhysicsUpdate {
  enabled: string[];
  weights: Record<string, number>;
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
/** On resume the server applies only `steps` and `render`; every other value
 * is fixed by the run's own config snapshot, so none are sent. */
export interface RunResumeRequest {
  resume: true;
  run_id: string;
  steps: number;
  render?: boolean;
}

export interface RunLaunchRequest {
  dataset?: string | null;
  /** Joint (transfer-learning) training: one model across these datasets. */
  datasets?: string[] | null;
  /** Whole datasets kept OUT of training and scored as a transfer test (axis B). */
  heldout_datasets?: string[] | null;
  resume?: boolean;
  run_id?: string | null;
  steps: number;
  lr: number;
  lr_halflife: number;
  n_data: number;
  n_coll: number;
  n_bc: number;
  holdout_frame: number;
  /** Per-dataset fraction of frames held out as an in-distribution validation set. */
  val_fraction: number;
  /** How the split picks frames: "tail" (extrapolation) | "scatter" (interpolation). */
  val_strategy: "tail" | "scatter";
  rebalance_every: number;
  log_every: number;
  weights: LossWeightsInput;
  render: boolean;
}

export interface RunJobStatus {
  run_id: string;
  dataset: string | null;
  state: "queued" | "running" | "done" | "error";
  stage: string | null;
  message: string | null;
  steps_done: number;
  steps_total: number;
}

/** A request to run the same configuration across several seeds. */
export interface SweepLaunchRequest extends RunLaunchRequest {
  seeds: number[];
}

export interface SweepStatus {
  sweep_id: string;
  dataset: string;
  state: "running" | "done" | "error";
  message: string | null;
  seeds: number[];
  children: RunJobStatus[];
}

/** One per-log-step loss record, streamed live over SSE (`hist` events). */
export interface LossRecord {
  step: number;
  /** Absent in histories from CLI-era checkpoints. */
  lr?: number | null;
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

/** One kinematics series; null marks an instant with no resolvable value
 * (e.g. an empty predicted mask early in growth). */
export type KinematicsSeries = (number | null)[];

/** Growth kinematics written by the evaluate stage (physical units). */
export interface Trajectory {
  t_ms: KinematicsSeries;
  nose_um: KinematicsSeries;
  area_um2: KinematicsSeries;
  measured: {
    t_ms: KinematicsSeries;
    nose_um: KinematicsSeries;
    area_um2: KinematicsSeries;
  };
}

/** One reconstructed instant: interface contour polylines in µm. */
export interface InterfaceFrame {
  t_ms: number;
  contours: number[][][];
}

/** One predicted field evaluated on a grid at t* (the /field endpoint). */
export interface FieldMap {
  run_id: string;
  dataset: string | null;
  name: string;
  unit: string;
  t_star: number;
  t_min_star: number;
  t_max_star: number;
  x_um: number[];
  y_um: number[];
  values: number[][];
  vmin: number;
  vmax: number;
  fields_available: string[];
}

export interface InterfaceData {
  run_id: string;
  domain: { x_um: [number, number]; y_um: [number, number]; x_pin_um: number };
  frames: InterfaceFrame[];
  measured: InterfaceFrame[];
}

/** A failed API response: the server's `detail` plus the HTTP status, so
 * callers can distinguish "not there yet" (404) from a real failure. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** One entry of FastAPI's 422 validation `detail` array. */
interface ValidationIssue {
  loc?: (string | number)[];
  msg?: string;
}

/** Flatten a FastAPI error `detail` into one readable line.
 *
 * `detail` is a plain string for our own HTTPExceptions, but a request that
 * fails request-body validation comes back as an array of `{loc, msg}` issues.
 * Rendering that array directly gives the useless "[object Object]"; here we
 * name the offending field(s) and the reason instead. */
function formatDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const lines = detail
      .map((issue: ValidationIssue) => {
        // Drop the leading "body"/"query" scope so the field reads plainly.
        const field = (issue.loc ?? [])
          .filter((part) => part !== "body" && part !== "query")
          .join(".");
        const msg = issue.msg ?? "invalid value";
        return field ? `${field}: ${msg}` : msg;
      })
      .filter(Boolean);
    if (lines.length > 0) return lines.join("; ");
  }
  return fallback;
}

/** Fetch + shared error handling: failures throw the API's `detail` when the
 * error body carries one, so every caller surfaces the actionable reason. */
async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const fallback = `${response.status} ${response.statusText}`;
    let detail = fallback;
    try {
      const body = (await response.json()) as { detail?: unknown };
      detail = formatDetail(body.detail, fallback);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, response.status);
  }
  if (response.status === 204) return undefined as T; // no-content (e.g. DELETE)
  return (await response.json()) as T;
}

const getJson = <T>(path: string) =>
  request<T>(path, { headers: { Accept: "application/json" } });

const send = <T>(path: string, method: string, body?: BodyInit) =>
  request<T>(path, { method, body });

const sendJson = <T>(path: string, method: string, payload: unknown) =>
  request<T>(path, {
    method,
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });

const runPath = (id: string) => `/api/runs/${encodeURIComponent(id)}`;

const datasetPath = (id: string) => `/api/datasets/${encodeURIComponent(id)}`;

export const api = {
  listRuns: (project?: string) =>
    getJson<RunSummary[]>(
      project
        ? `/api/runs?project=${encodeURIComponent(project)}`
        : "/api/runs",
    ),
  getRun: (id: string) => getJson<RunDetail>(runPath(id)),
  /** Delete a run and everything under its output directory. Rejected (409) while the
   * run is training. */
  deleteRun: (id: string) => send<void>(runPath(id), "DELETE"),
  getValidation: (id: string) =>
    getJson<PhysicsValidation>(`${runPath(id)}/validation`),

  startRun: (request: RunLaunchRequest | RunResumeRequest) =>
    sendJson<RunJobStatus>("/api/runs", "POST", request),
  getRunStatus: (id: string) => getJson<RunJobStatus>(`${runPath(id)}/status`),
  getActiveRun: () => getJson<RunJobStatus | null>("/api/runs/active"),
  getLossHistory: (id: string) =>
    getJson<LossRecord[]>(`${runPath(id)}/loss-history`),
  getTrajectory: (id: string, dataset?: string) =>
    getJson<Trajectory>(
      `${runPath(id)}/trajectory` +
        (dataset ? `?dataset=${encodeURIComponent(dataset)}` : ""),
    ),
  getField: (id: string, name: string, t: number, dataset?: string) =>
    getJson<FieldMap>(
      `${runPath(id)}/field?name=${encodeURIComponent(name)}&t=${t.toFixed(3)}` +
        (dataset ? `&dataset=${encodeURIComponent(dataset)}` : ""),
    ),
  getInterface: (id: string, frames = 48) =>
    getJson<InterfaceData>(`${runPath(id)}/interface?frames=${frames}`),

  startSweep: (request: SweepLaunchRequest) =>
    sendJson<SweepStatus>("/api/sweeps", "POST", request),
  getSweep: (id: string) =>
    getJson<SweepStatus>(`/api/sweeps/${encodeURIComponent(id)}`),
  getActiveSweep: () => getJson<SweepStatus | null>("/api/sweeps/active"),

  listProjects: () => getJson<ProjectSummary[]>("/api/projects"),
  getProject: (id: string) =>
    getJson<ProjectSummary>(`/api/projects/${encodeURIComponent(id)}`),
  createProject: (name: string, description: string) =>
    sendJson<ProjectSummary>("/api/projects", "POST", { name, description }),
  updateProject: (
    id: string,
    fields: Partial<Pick<ProjectSummary, "name" | "description" | "datasets">>,
  ) =>
    sendJson<ProjectSummary>(
      `/api/projects/${encodeURIComponent(id)}`,
      "PATCH",
      fields,
    ),
  deleteProject: (id: string) =>
    send<ProjectSummary>(`/api/projects/${encodeURIComponent(id)}`, "DELETE"),

  listDatasets: () => getJson<DatasetSummary[]>("/api/datasets"),
  getDataset: (id: string) => getJson<DatasetDetail>(datasetPath(id)),
  getDatasetGroups: (id: string) =>
    getJson<DimensionlessGroups>(`${datasetPath(id)}/groups`),
  updateConditions: (id: string, fields: ConditionsUpdate) =>
    sendJson<ConditionsResponse>(
      `${datasetPath(id)}/conditions`,
      "PATCH",
      fields,
    ),
  /** Replace the series' excluded frames (full set, not a delta). */
  setExcludedFrames: (id: string, frames: number[]) =>
    sendJson<DatasetDetail>(`${datasetPath(id)}/excluded-frames`, "PUT", {
      excluded_frames: frames,
    }),
  /** Set (or, with a blank string, clear) the series' display name. The id is
   * immutable; this only changes what the UI shows. */
  setSeriesLabel: (id: string, label: string) =>
    sendJson<DatasetDetail>(`${datasetPath(id)}/label`, "PUT", { label }),
  getQcData: (id: string) => getJson<QcData>(`${datasetPath(id)}/qc-data`),
  getPreprocessStatus: (id: string) =>
    getJson<PreprocessStatus>(`${datasetPath(id)}/preprocess`),
  startPreprocess: (id: string) =>
    send<PreprocessStatus>(`${datasetPath(id)}/preprocess`, "POST"),
  uploadFrames: (id: string, files: FileList | File[]) => {
    const form = new FormData();
    for (const file of Array.from(files)) form.append("files", file);
    return send<DatasetSummary>(`${datasetPath(id)}/upload`, "POST", form);
  },

  getModel: (id: string) =>
    getJson<ModelArchitecture>(`/api/model/${encodeURIComponent(id)}`),
  updateModel: (id: string, update: ModelUpdate) =>
    sendJson<ModelArchitecture>(
      `/api/model/${encodeURIComponent(id)}`,
      "PUT",
      update,
    ),
  getPhysics: (id: string) =>
    getJson<PhysicsState>(`/api/physics/${encodeURIComponent(id)}`),
  updatePhysics: (id: string, update: PhysicsUpdate) =>
    sendJson<PhysicsState>(
      `/api/physics/${encodeURIComponent(id)}`,
      "PUT",
      update,
    ),

  /** The catalogue of characterised working fluids for the new-series form. */
  listFluids: () => getJson<Fluid[]>("/api/fluids"),
};

/** Direct artifact URLs (used as href / img src / video src, not fetched as JSON). */
export const artifactUrl = {
  figure: (id: string, name: string) =>
    `${runPath(id)}/figures/${encodeURIComponent(name)}`,
  video: (id: string) => `${runPath(id)}/video`,
  checkpoint: (id: string) => `${runPath(id)}/checkpoint`,
  tensors: (id: string) => `${runPath(id)}/tensors`,
  datasetFrame: (id: string, n: number) => `${datasetPath(id)}/frames/${n}`,
};
