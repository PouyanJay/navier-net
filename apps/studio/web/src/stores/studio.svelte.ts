/** The studio's state: one run, its catalogue, the frames in view, the
 * quantity on the canvas, the probes, and the look. Runes make every piece
 * reactive; App syncs the shareable part with the URL hash. */

import { api } from "../lib/api/client";
import type { Quantity, RunDetail, RunSummary, SectionKind } from "../lib/api/types";
import type { ColormapName } from "../lib/colormaps";

export type ViewId = "geometry" | "fields" | "edge" | "residuals" | "probes" | "budgets" | "export";
export type Theme = "dark" | "light";
export type Palette = "aurora" | "ember" | "tide";
export type RangeMode = "frame" | "all" | "custom";
export type Tool = "pan" | "line" | "point";

export const VIEWS: { id: ViewId; label: string }[] = [
  { id: "geometry", label: "Geometry" },
  { id: "fields", label: "Fields" },
  { id: "edge", label: "Edge" },
  { id: "residuals", label: "Residuals" },
  { id: "probes", label: "Probes" },
  { id: "budgets", label: "Budgets" },
  { id: "export", label: "Export" },
];

export interface Probe {
  id: string;
  name: string;
  kind: "line" | "points" | "section";
  points?: [number, number][];
  section?: SectionKind;
  behindNoseUm?: number;
  spanUm?: [number, number];
  n: number;
  visible: boolean;
}

export const SECTION_LABELS: Record<SectionKind, string> = {
  centreline: "Centreline",
  wall_top: "Upper wall",
  wall_bottom: "Lower wall",
  gap_top: "Upper gap cross-section",
  gap_bottom: "Lower gap cross-section",
  mid_gap_top: "Upper mid-gap line",
  mid_gap_bottom: "Lower mid-gap line",
};

const STORAGE_THEME = "naviernet-studio:theme";
const STORAGE_PALETTE = "naviernet-studio:palette";
const STORAGE_PROBES = "naviernet-studio:probes";
const DEFAULT_QUANTITY = "speed";
export const GRID_NX = 512;
export const GRID_NY = 171;

function stored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function store(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage may be unavailable */
  }
}

class Studio {
  runs = $state<RunSummary[]>([]);
  runId = $state<string | null>(null);
  run = $state<RunDetail | null>(null);
  catalogue = $state<Quantity[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);

  view = $state<ViewId>("fields");
  frame = $state(1);
  frames = $state<number[]>([]);
  playing = $state(false);

  quantity = $state(DEFAULT_QUANTITY);
  colormap = $state<ColormapName | "auto">("auto");
  rangeMode = $state<RangeMode>("frame");
  customRange = $state<[number, number]>([0, 1]);
  layers = $state({
    field: true,
    image: false,
    boundary: true,
    measured: false,
    walls: true,
    probes: true,
    grid: false,
  });
  imageOpacity = $state(0.85);
  /** Remove the set-up's heaters and walls from the camera frame. */
  cleanImage = $state(false);
  tool = $state<Tool>("pan");

  probes = $state<Probe[]>([]);
  activeProbe = $state<string | null>(null);

  theme = $state<Theme>((stored(STORAGE_THEME) as Theme) ?? "dark");
  palette = $state<Palette>((stored(STORAGE_PALETTE) as Palette) ?? "aurora");

  readonly byId = $derived(new Map(this.catalogue.map((q) => [q.id, q])));
  readonly current = $derived(this.byId.get(this.quantity) ?? null);
  readonly eventFrames = $derived(
    (this.run?.frames ?? []).filter((f) => f.role === "train" || f.role === "holdout"),
  );
  readonly frameInfo = $derived(
    (this.run?.frames ?? []).find((f) => f.frame === this.frame) ?? null,
  );
  readonly selectedFrames = $derived(
    this.frames.length > 0 ? this.frames : this.frameInfo ? [this.frame] : [],
  );

  constructor() {
    const saved = stored(STORAGE_PROBES);
    if (saved) {
      try {
        this.probes = JSON.parse(saved) as Probe[];
      } catch {
        this.probes = [];
      }
    }
  }

  async loadRuns(): Promise<void> {
    this.runs = await api.listRuns();
    if (!this.runId && this.runs.length > 0) await this.selectRun(this.runs[0].id);
  }

  async selectRun(id: string): Promise<void> {
    if (id === this.runId && this.run) return;
    this.loading = true;
    this.error = null;
    try {
      const [run, catalogue] = await Promise.all([api.getRun(id), api.getCatalogue(id)]);
      this.runId = id;
      this.run = run;
      this.catalogue = catalogue;
      const event = run.frames.filter((f) => f.role === "train" || f.role === "holdout");
      if (!event.some((f) => f.frame === this.frame)) this.frame = event[0]?.frame ?? 1;
      this.frames = this.frames.filter((n) => event.some((f) => f.frame === n));
      if (!catalogue.some((q) => q.id === this.quantity && q.available))
        this.quantity = DEFAULT_QUANTITY;
    } catch (error) {
      this.error = error instanceof Error ? error.message : String(error);
    } finally {
      this.loading = false;
    }
  }

  setTheme(theme: Theme): void {
    this.theme = theme;
    document.documentElement.dataset.theme = theme;
    store(STORAGE_THEME, theme);
  }

  setPalette(palette: Palette): void {
    this.palette = palette;
    document.documentElement.dataset.palette = palette;
    store(STORAGE_PALETTE, palette);
  }

  toggleFrame(frame: number): void {
    this.frames = this.frames.includes(frame)
      ? this.frames.filter((f) => f !== frame)
      : [...this.frames, frame].sort((a, b) => a - b);
  }

  stepFrame(delta: number): void {
    const list = this.eventFrames.map((f) => f.frame);
    if (list.length === 0) return;
    const index = Math.max(0, list.indexOf(this.frame));
    this.frame = list[(index + delta + list.length) % list.length];
  }

  addProbe(probe: Omit<Probe, "id" | "visible">): Probe {
    const created: Probe = { ...probe, id: `p${Date.now().toString(36)}`, visible: true };
    this.probes = [...this.probes, created];
    this.activeProbe = created.id;
    this.persistProbes();
    return created;
  }

  updateProbe(id: string, patch: Partial<Probe>): void {
    this.probes = this.probes.map((p) => (p.id === id ? { ...p, ...patch } : p));
    this.persistProbes();
  }

  removeProbe(id: string): void {
    this.probes = this.probes.filter((p) => p.id !== id);
    if (this.activeProbe === id) this.activeProbe = null;
    this.persistProbes();
  }

  private persistProbes(): void {
    store(STORAGE_PROBES, JSON.stringify(this.probes));
  }

  quantitiesOfKind(...kinds: Quantity["kind"][]): Quantity[] {
    return this.catalogue.filter((q) => kinds.includes(q.kind));
  }
}

export const studio = new Studio();
