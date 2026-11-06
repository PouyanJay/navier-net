/** Types mirroring the studio service's responses. */

export type FrameRole = "train" | "holdout" | "truncated" | "excluded";

export interface FrameInfo {
  frame: number;
  t_ms: number | null;
  t_star: number | null;
  role: FrameRole;
  row: number | null;
}

export interface RunSummary {
  id: string;
  dataset: string | null;
  iou_mean: number | null;
  iou_holdout: number | null;
  nose_speed_mm_s: number | null;
  n_frames: number;
  verdicts: Record<string, boolean | null>;
  flow_rate_mL_hr: number | null;
  T_init_C: number | null;
  q_wall_W_cm2?: number | null;
  has_config: boolean;
}

export interface RunDetail {
  id: string;
  dataset: string;
  fields: string[];
  has_front: boolean;
  has_footprint: boolean;
  sharp_thermal: boolean;
  equations: string[];
  inertia: boolean;
  frames: FrameInfo[];
  scales: Record<string, number>;
  groups: Record<string, number>;
  conditions: {
    flow_rate_mL_hr: number;
    T_init_C: number | null;
    T_inlet_C: number | null;
    T_sat_C: number;
    q_wall_W_cm2: number;
    channel_width_um: number;
    channel_height_um: number;
    fluid: string;
  };
  domain_um: {
    x: [number, number];
    y_walls: [number, number];
    y_image: [number, number];
    centreline: number;
  };
}

export type QuantityKind = "volume" | "derived" | "residual" | "term" | "edge";
export type ColormapFamily = "sequential" | "diverging" | "thermal" | "magma";

export interface Quantity {
  id: string;
  label: string;
  symbol: string;
  unit: string;
  kind: QuantityKind;
  group: string;
  tex: string;
  plain: string;
  how: string;
  scale: string;
  needs_grad: boolean;
  signed: boolean;
  cmap: ColormapFamily;
  needs: string[];
  caveats: string[];
  components: string[];
  available?: boolean;
  factor?: number;
}

export interface GridQuantity {
  id: string;
  unit: string;
  signed: boolean;
  cmap: ColormapFamily;
  vmin: number | null;
  vmax: number | null;
}

export interface GridHeader {
  run: string;
  frame: number | null;
  t_ms: number;
  t_star: number;
  shape: [number, number];
  extent_um: [number, number, number, number];
  quantities: GridQuantity[];
  boundary_um: number[][][];
  nose_um: number | null;
}

export interface Grid {
  header: GridHeader;
  planes: Float32Array[];
}

export interface InterfaceContour {
  t_star: number;
  t_ms: number;
  rings_um: number[][][];
  nose_um: number | null;
  root_um: number | null;
  vapour_area_mm2: number;
}

export type ProbeKind = "line" | "points" | "section";
export type SectionKind =
  | "centreline"
  | "wall_top"
  | "wall_bottom"
  | "gap_top"
  | "gap_bottom"
  | "mid_gap_top"
  | "mid_gap_bottom";

export interface ProbeRequest {
  kind: ProbeKind;
  quantities: string[];
  frames?: number[];
  t_ms?: number[];
  points_um?: [number, number][];
  n?: number;
  section?: SectionKind;
  behind_nose_um?: number;
  span_um?: [number, number];
  edge_offset?: number;
}

export interface ProbeFrame {
  frame: number | null;
  t_ms: number;
  t_star: number;
  stations: { label: string; s_um: number[]; x_um: number[]; y_um: number[] };
  values: Record<string, (number | null)[]>;
}

export interface ProbeResponse {
  kind: ProbeKind;
  section: SectionKind | null;
  units: Record<string, string>;
  frames: ProbeFrame[];
  provenance: Record<string, unknown>;
}

export interface EdgeResponse {
  frame: number | null;
  t_ms: number;
  t_star: number;
  n_samples: number;
  segments: string[];
  units: Record<string, string>;
  values: Record<string, (number | null)[]>;
  provenance: Record<string, unknown>;
}

export interface Transform {
  frame_px: [number | null, number | null];
  um_per_px: number;
  L_ref_um: number;
  y_roi_rows: [number, number];
  wall_rows: [number, number];
  crop_rows: [number, number];
  x_flipped: boolean;
  walls_um: [number, number];
  centreline_um: number;
  domain_um: [number, number, number, number];
  x_pin_um: number;
}

export interface Agreement {
  frame: number;
  t_ms: number;
  role: FrameRole;
  iou: number;
  iou_fullres: number;
  nose_model_um: number | null;
  nose_measured_um: number | null;
  nose_error_um: number | null;
  area_model_mm2: number;
  area_measured_mm2: number;
  only_model_mm2: number;
  only_measured_mm2: number;
  measured_rings_um: number[][][];
  model_rings_um: number[][][];
  valid_fraction: number;
}

export interface GapBudget {
  side: "upper" | "lower";
  frame: number | null;
  t_ms: number;
  stations: Record<string, number[]>;
  bins: Record<string, number>[];
}

export interface FluxBudget {
  frame: number | null;
  t_ms: number;
  x_um: number[];
  q_total_mL_h: number[];
  q_liquid_mL_h: number[];
  q_required_mL_h: number[];
  q_inlet_mL_h: number;
  upstream_growth_mL_h: number[];
  defect: number[];
}

export type HeatBudget = Record<string, number | string>;

export interface AuditFrame {
  frame: number;
  t_star: number;
  residual_rms: Record<string, number>;
  flux: { stations: number[]; defect: number[]; max_abs: number };
  heat: Record<string, number | string>;
  outline: { iou: number; nose_model: number; nose_measured: number; nose_error: number };
  relative_error?: Record<string, Record<string, number> | number>;
  front?: Record<string, number | Record<string, number>>;
  mirror: Record<string, number>;
}

export interface Audit {
  run_name: string;
  thresholds: Record<string, number>;
  verdicts: Record<string, { worst: number; pass: boolean; min?: number; max?: number }>;
  frames: AuditFrame[];
  scan?: Record<string, number[]>;
}
