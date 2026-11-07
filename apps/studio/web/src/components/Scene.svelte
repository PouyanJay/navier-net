<script lang="ts">
  /**
   * The canvas: a WebGL2 field (and camera frame) under an SVG overlay that
   * draws the bubble boundary, the measured contour, the walls, the probes and
   * a readout. One µm-to-screen view is shared by both layers; wheel zooms,
   * drag pans, and the probe tools draw on top.
   */
  import type { Grid, Transform } from "../lib/api/types";
  import { boundaryTone, buildLut, type ColormapName } from "../lib/colormaps";
  import { sceneToPng } from "../lib/export";
  import { FieldRenderer } from "../lib/gl/renderer";
  import { Icon, icons } from "../lib/icons";
  import {
    cropExtent,
    fitView,
    nearestOnRings,
    pan,
    polylinePath,
    sampleGrid,
    svgPath,
    toScreen,
    toUm,
    zoomAt,
    type Extent,
    type View,
  } from "../lib/transform";
  import { fmt } from "../lib/units";
  import { arrows, streamlines, type Streamline, type VectorField } from "../lib/vectors";
  import type { Probe, Tool, VectorMode } from "../stores/studio.svelte";

  interface Props {
    grid?: Grid | null;
    plane?: number;
    colormap: ColormapName;
    range: [number, number];
    unit?: string;
    label?: string;
    extent: Extent;
    boundary?: number[][][];
    measured?: number[][][];
    walls?: [number, number] | null;
    centreline?: number | null;
    imageUrl?: string | null;
    transform?: Transform | null;
    imageOpacity?: number;
    showField?: boolean;
    /** How opaque the field is over the camera frame (1 hides the frame). */
    fieldOpacity?: number;
    showBoundary?: boolean;
    showMeasured?: boolean;
    showWalls?: boolean;
    showGrid?: boolean;
    probes?: Probe[];
    probeColors?: Record<string, string>;
    tool?: Tool;
    snap?: boolean;
    onDraw?: (kind: "line" | "point", points: [number, number][]) => void;
    onHover?: (xUm: number, yUm: number, value: number) => void;
    markers?: { x: number; y: number; color: string; label?: string }[];
    rings?: { points: number[][]; colors: string[] } | null;
    /** The vector drawn over the field as arrows or streamlines, if any. */
    vectors?: VectorField | null;
    vectorMode?: VectorMode;
    height?: number;
    background?: string;
  }

  let {
    grid = null,
    plane = 0,
    colormap,
    range,
    unit = "",
    label = "",
    extent,
    boundary = [],
    measured = [],
    walls = null,
    centreline = null,
    imageUrl = null,
    transform = null,
    imageOpacity = 0.85,
    showField = true,
    fieldOpacity = 1,
    showBoundary = true,
    showMeasured = false,
    showWalls = true,
    showGrid = false,
    probes = [],
    probeColors = {},
    tool = "pan",
    snap = true,
    onDraw,
    onHover,
    markers = [],
    rings = null,
    vectors = null,
    vectorMode = "map",
    height = 0,
    background = "",
  }: Props = $props();

  let canvas = $state<HTMLCanvasElement | null>(null);
  let overlay = $state<SVGSVGElement | null>(null);
  let width = $state(800);
  let boxHeight = $state(400);
  let view = $state<View>({ scale: 1, tx: 0, ty: 0 });
  let renderer: FieldRenderer | null = null;
  let glMissing = $state(false);
  let draft = $state<[number, number][]>([]);
  let cursor = $state<{ x: number; y: number; value: number } | null>(null);
  let dragging: { sx: number; sy: number; view: View } | null = null;
  let image = $state<HTMLImageElement | null>(null);

  // Without an explicit height the scene takes most of the viewport, so the
  // size never depends on what the canvas holds.
  const DEFAULT_HEIGHT = "min(72dvh, 820px)";
  // Presses closer than this (screen px) to the last draft vertex are the
  // same click, as in the pair that makes a double-click.
  const DOUBLE_CLICK_PX = 4;

  const dpr = () => Math.max(1, window.devicePixelRatio || 1);
  const sceneHeight = $derived(height > 0 ? height : boxHeight);
  const tone = $derived(boundaryTone(colormap));
  const strokeMain = $derived(tone === "light" ? "var(--boundary-light)" : "var(--boundary-dark)");
  const strokeHalo = $derived(tone === "light" ? "var(--boundary-dark)" : "var(--boundary-light)");

  export function fit(): void {
    view = fitView(extent, { width, height: sceneHeight });
  }

  export async function toPng(): Promise<Blob> {
    if (!canvas || !overlay) throw new Error("scene not mounted");
    const bg =
      background || getComputedStyle(document.documentElement).getPropertyValue("--canvas").trim();
    return sceneToPng(canvas, overlay, bg);
  }

  $effect(() => {
    if (!canvas) return;
    renderer = new FieldRenderer(canvas);
    glMissing = !renderer.available;
    return () => {
      renderer?.dispose();
      renderer = null;
    };
  });

  // Fit whenever the extent or the canvas size changes, so the view is right
  // on first paint and after a resize; pan and zoom keep their place otherwise.
  let fitKey = "";
  $effect(() => {
    const key = `${extent.x0}|${extent.x1}|${extent.y0}|${extent.y1}|${width}|${sceneHeight}`;
    if (width > 0 && sceneHeight > 0 && key !== fitKey) {
      fitKey = key;
      fit();
    }
  });

  $effect(() => {
    if (!imageUrl) {
      image = null;
      return;
    }
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => (image = img);
    // A frame the service cannot give (a ghost frame, an ungenerated cleaned
    // frame) must not leave the previous picture on the canvas.
    img.onerror = () => (image = null);
    img.src = imageUrl;
  });

  const imageExtent = $derived.by((): [number, number, number, number] | null => {
    if (!transform || transform.frame_px[0] === null) return null;
    const map = {
      width: transform.frame_px[0],
      umPerPx: transform.um_per_px,
      roiTop: transform.y_roi_rows[0],
      cropTop: transform.crop_rows[0],
    };
    const e = cropExtent(map, transform.crop_rows[1]);
    return [e.x0, e.x1, e.y0, e.y1];
  });

  $effect(() => {
    if (!renderer) return;
    renderer.setLut(buildLut(colormap));
  });

  $effect(() => {
    if (!renderer) return;
    if (grid && showField) {
      const [ny, nx] = grid.header.shape;
      renderer.setField({ plane: grid.planes[plane], nx, ny, extent: grid.header.extent_um });
    } else {
      renderer.setField(null);
    }
  });

  $effect(() => {
    if (!renderer) return;
    if (image && imageExtent) renderer.setImage({ image, extent: imageExtent, flip: true });
    else renderer.setImage(null);
  });

  $effect(() => {
    if (!renderer) return;
    renderer.setRange(range);
    renderer.setOpacity(showField && grid ? fieldOpacity : 0, image ? imageOpacity : 0);
    renderer.draw(view, width, sceneHeight, dpr());
  });

  function onWheel(event: WheelEvent) {
    event.preventDefault();
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    const factor = Math.exp(-event.deltaY * 0.0015);
    view = zoomAt(view, event.clientX - rect.left, event.clientY - rect.top, factor);
  }

  function local(event: PointerEvent): [number, number] {
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  function snapped(xUm: number, yUm: number): [number, number] {
    if (!snap) return [xUm, yUm];
    const tolerance = 10 / view.scale;
    if (walls) {
      if (Math.abs(yUm - walls[1]) < tolerance) return [xUm, walls[1]];
      if (Math.abs(yUm - walls[0]) < tolerance) return [xUm, walls[0]];
    }
    if (centreline !== null && Math.abs(yUm - centreline) < tolerance) return [xUm, centreline];
    const near = nearestOnRings(boundary, xUm, yUm);
    if (near && near.distance < tolerance) return [near.x, near.y];
    return [xUm, yUm];
  }

  function onPointerDown(event: PointerEvent) {
    const [sx, sy] = local(event);
    if (tool === "pan" || event.button === 1) {
      dragging = { sx, sy, view };
      (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
      return;
    }
    const [x, y] = snapped(...toUm(view, sx, sy));
    if (tool === "point") {
      onDraw?.("point", [[x, y]]);
      return;
    }
    // The second press of a double-click lands on the last vertex: it ends
    // the line (see onDoubleClick) and must not duplicate that vertex.
    const last = draft.at(-1);
    if (last) {
      const [lx, ly] = toScreen(view, last[0], last[1]);
      if (Math.hypot(lx - sx, ly - sy) < DOUBLE_CLICK_PX) return;
    }
    draft = [...draft, [x, y]];
  }

  // Leaving the line tool abandons the vertices placed so far, so switching
  // to pan or point never leaves a dashed draft nobody can remove.
  $effect(() => {
    if (tool !== "line") draft = [];
  });

  function onDoubleClick() {
    if (tool === "line" && draft.length >= 2) {
      onDraw?.("line", draft);
      draft = [];
    }
  }

  function onKey(event: KeyboardEvent) {
    if (event.key === "Escape") draft = [];
    if (event.key === "Enter" && tool === "line" && draft.length >= 2) {
      onDraw?.("line", draft);
      draft = [];
    }
    if (event.key === "f") fit();
  }

  function onPointerMove(event: PointerEvent) {
    const [sx, sy] = local(event);
    if (dragging) {
      view = pan(dragging.view, sx - dragging.sx, sy - dragging.sy);
      return;
    }
    const [x, y] = toUm(view, sx, sy);
    let value = NaN;
    if (grid) {
      const [ny, nx] = grid.header.shape;
      value = sampleGrid(grid.planes[plane], nx, ny, grid.header.extent_um, x, y);
    }
    cursor = { x, y, value };
    onHover?.(x, y, value);
  }

  function onPointerUp() {
    dragging = null;
  }

  function probePath(probe: Probe): string {
    if (!probe.points) return "";
    return polylinePath(probe.points, view);
  }

  const gridLines = $derived.by(() => {
    if (!showGrid) return [];
    const step = view.scale > 2 ? 50 : view.scale > 0.8 ? 100 : 250;
    const out: {
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      label: string;
      axis: "x" | "y";
    }[] = [];
    for (let x = Math.ceil(extent.x0 / step) * step; x <= extent.x1; x += step) {
      const [sx, sy0] = toScreen(view, x, extent.y0);
      const [, sy1] = toScreen(view, x, extent.y1);
      out.push({ x1: sx, y1: sy0, x2: sx, y2: sy1, label: `${x}`, axis: "x" });
    }
    for (let y = Math.ceil(extent.y0 / step) * step; y <= extent.y1; y += step) {
      const [sx0, sy] = toScreen(view, extent.x0, y);
      const [sx1] = toScreen(view, extent.x1, y);
      out.push({ x1: sx0, y1: sy, x2: sx1, y2: sy, label: `${y}`, axis: "y" });
    }
    return out;
  });

  // --- vectors ---------------------------------------------------------------
  // Arrows and streamlines sit on a lattice a fixed number of screen pixels
  // apart, so zooming in shows more of them rather than bigger ones.
  const LATTICE_PX = 24;
  const ARROW_MAX_PX = 0.9 * LATTICE_PX;
  const ARROW_HEAD_PX = 5;
  const ARROW_MIN_PX = 1.5;
  const STREAM_HEAD_PX = 4;
  // The magnitude that gets the longest arrow: a high percentile rather than
  // the maximum, so one hot cell does not shrink every other arrow.
  const ARROW_REFERENCE_PERCENTILE = 0.98;

  /** The part of the field extent on screen (µm). */
  const visible = $derived.by((): Extent => {
    const [left, top] = toUm(view, 0, 0);
    const [right, bottom] = toUm(view, width, sceneHeight);
    return {
      x0: Math.max(extent.x0, left),
      x1: Math.min(extent.x1, right),
      y0: Math.max(extent.y0, bottom),
      y1: Math.min(extent.y1, top),
    };
  });
  const latticeUm = $derived(LATTICE_PX / view.scale);

  function segment(x0: number, y0: number, x1: number, y1: number): string {
    return `M${x0.toFixed(1)} ${y0.toFixed(1)}L${x1.toFixed(1)} ${y1.toFixed(1)}`;
  }

  /** Two barbs meeting at (hx, hy), pointing along the screen direction (dx, dy). */
  function head(hx: number, hy: number, dx: number, dy: number, size: number): string {
    const bx = hx - dx * size;
    const by = hy - dy * size;
    const nx = -dy * size * 0.5;
    const ny = dx * size * 0.5;
    return (
      `M${(bx + nx).toFixed(1)} ${(by + ny).toFixed(1)}` +
      `L${hx.toFixed(1)} ${hy.toFixed(1)}` +
      `L${(bx - nx).toFixed(1)} ${(by - ny).toFixed(1)}`
    );
  }

  const arrowPath = $derived.by((): string => {
    if (!vectors || vectorMode !== "arrows") return "";
    const list = arrows(vectors, visible, latticeUm);
    if (list.length === 0) return "";
    const sorted = list.map((a) => a.magnitude).sort((a, b) => a - b);
    const at = Math.min(sorted.length - 1, Math.floor(ARROW_REFERENCE_PERCENTILE * sorted.length));
    const reference = sorted[at] || 1;
    const parts: string[] = [];
    for (const a of list) {
      const length = ARROW_MAX_PX * Math.min(1, a.magnitude / reference);
      if (length < ARROW_MIN_PX) continue;
      const [cx, cy] = toScreen(view, a.x, a.y);
      // Screen y points down, so the y component flips.
      const dx = a.dx;
      const dy = -a.dy;
      const hx = cx + (dx * length) / 2;
      const hy = cy + (dy * length) / 2;
      parts.push(
        segment(cx - (dx * length) / 2, cy - (dy * length) / 2, hx, hy),
        head(hx, hy, dx, dy, Math.min(ARROW_HEAD_PX, 0.4 * length)),
      );
    }
    return parts.join("");
  });

  // Streamlines are traced once per lattice spacing and visible window
  // (quantised to the lattice), then only projected as the view moves.
  let traced: { field: VectorField; key: string; lines: Streamline[] } | null = null;
  const streamPaths = $derived.by((): { lines: string; heads: string } | null => {
    if (!vectors || vectorMode !== "streamlines") return null;
    const spacing = latticeUm;
    const cell = (value: number) => Math.floor(value / spacing);
    const key = [
      spacing.toPrecision(3),
      cell(visible.x0),
      cell(visible.x1),
      cell(visible.y0),
      cell(visible.y1),
    ].join("|");
    if (!traced || traced.field !== vectors || traced.key !== key) {
      traced = { field: vectors, key, lines: streamlines(vectors, visible, spacing) };
    }
    const lines: string[] = [];
    const heads: string[] = [];
    for (const line of traced.lines) {
      lines.push(polylinePath(line.points, view));
      if (line.head) {
        const [hx, hy] = toScreen(view, line.head.x, line.head.y);
        heads.push(head(hx, hy, line.head.dx, -line.head.dy, STREAM_HEAD_PX));
      }
    }
    return { lines: lines.join(""), heads: heads.join("") };
  });
</script>

<div
  class="scene"
  bind:clientWidth={width}
  bind:clientHeight={boxHeight}
  style:height={height > 0 ? `${height}px` : DEFAULT_HEIGHT}
  data-tool={tool}
>
  <canvas bind:this={canvas} style:width="{width}px" style:height="{sceneHeight}px"></canvas>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex a11y_no_noninteractive_element_interactions -->
  <div
    class="interact"
    role="application"
    aria-label="{label} field map with the bubble boundary. Arrow keys step frames, f fits the view"
    tabindex="0"
    onwheel={onWheel}
    onpointerdown={onPointerDown}
    onpointermove={onPointerMove}
    onpointerup={onPointerUp}
    onpointerleave={() => {
      cursor = null;
      dragging = null;
    }}
    ondblclick={onDoubleClick}
    onkeydown={onKey}
  >
    <svg bind:this={overlay} class="overlay" {width} height={sceneHeight} aria-hidden="true">
      {#if showGrid}
        <g class="grid">
          {#each gridLines as g (g.axis + g.label)}
            <line x1={g.x1} y1={g.y1} x2={g.x2} y2={g.y2} />
            <text x={g.axis === "x" ? g.x1 + 3 : g.x1 + 3} y={g.axis === "x" ? g.y2 + 11 : g.y1 - 3}
              >{g.label}</text
            >
          {/each}
        </g>
      {/if}
      {#if showWalls && walls}
        {@const [, top] = toScreen(view, extent.x0, walls[1])}
        {@const [, bottom] = toScreen(view, extent.x0, walls[0])}
        {@const [x0] = toScreen(view, extent.x0, 0)}
        {@const [x1] = toScreen(view, extent.x1, 0)}
        <g class="walls">
          <line x1={x0} x2={x1} y1={top} y2={top} />
          <line x1={x0} x2={x1} y1={bottom} y2={bottom} />
          {#if centreline !== null}
            {@const [, mid] = toScreen(view, extent.x0, centreline)}
            <line class="centre" x1={x0} x2={x1} y1={mid} y2={mid} />
          {/if}
        </g>
      {/if}
      {#if showMeasured}
        <g class="measured">
          {#each measured as ring, i (i)}
            <path d={svgPath(ring, view)} />
          {/each}
        </g>
      {/if}
      {#if showBoundary}
        <g class="boundary">
          {#each boundary as ring, i (i)}
            <path class="halo" d={svgPath(ring, view)} style:stroke={strokeHalo} />
            <path class="main" d={svgPath(ring, view)} style:stroke={strokeMain} />
          {/each}
        </g>
      {/if}
      {#if vectorMode === "arrows" && arrowPath}
        <g class="vectors">
          <path class="halo" d={arrowPath} />
          <path class="main" d={arrowPath} />
        </g>
      {:else if streamPaths}
        <g class="vectors">
          <path class="halo" d={streamPaths.lines} />
          <path class="main" d={streamPaths.lines} />
          <path class="halo" d={streamPaths.heads} />
          <path class="main" d={streamPaths.heads} />
        </g>
      {/if}
      {#if rings}
        <g class="rings">
          {#each rings.points as [x, y], i (i)}
            {@const [sx, sy] = toScreen(view, x, y)}
            <circle cx={sx} cy={sy} r="2.6" fill={rings.colors[i]} />
          {/each}
        </g>
      {/if}
      <g class="probes">
        {#each probes.filter((p) => p.visible) as probe (probe.id)}
          {@const color = probeColors[probe.id] ?? "var(--probe)"}
          {#if probe.kind === "points" && probe.points}
            {#each probe.points as [x, y], i (i)}
              {@const [sx, sy] = toScreen(view, x, y)}
              <circle class="handle" cx={sx} cy={sy} r="5" stroke={color} />
              <text x={sx + 8} y={sy - 6} fill={color}
                >{probe.name}{probe.points.length > 1 ? ` ${i + 1}` : ""}</text
              >
            {/each}
          {:else if probe.points}
            <path class="line" d={probePath(probe)} stroke={color} />
            {#each probe.points as [x, y], i (i)}
              {@const [sx, sy] = toScreen(view, x, y)}
              <circle class="handle" cx={sx} cy={sy} r="4" stroke={color} />
            {/each}
            {@const [lx, ly] = toScreen(view, probe.points[0][0], probe.points[0][1])}
            <text x={lx + 8} y={ly - 8} fill={color}>{probe.name}</text>
          {/if}
        {/each}
        {#if draft.length > 0}
          <path class="line draft" d={polylinePath(draft, view)} />
          {#each draft as [x, y], i (i)}
            {@const [sx, sy] = toScreen(view, x, y)}
            <circle class="handle" cx={sx} cy={sy} r="4" />
          {/each}
        {/if}
      </g>
      <g class="markers">
        {#each markers as m, i (i)}
          {@const [sx, sy] = toScreen(view, m.x, m.y)}
          <circle cx={sx} cy={sy} r="5" fill={m.color} />
          {#if m.label}<text x={sx + 8} y={sy + 4} fill={m.color}>{m.label}</text>{/if}
        {/each}
      </g>
    </svg>
  </div>
  {#if cursor}
    <div class="hud mono" aria-live="polite">
      <span>x {fmt(cursor.x, 4)} µm</span>
      <span>y {fmt(cursor.y, 4)} µm</span>
      {#if Number.isFinite(cursor.value)}<span class="value">{fmt(cursor.value, 4)} {unit}</span
        >{/if}
    </div>
  {/if}
  {#if tool === "line"}
    <div class="hint">Click to add vertices, double-click or Enter to finish, Esc to cancel</div>
  {/if}
  <button class="icon-btn fit" aria-label="Fit to view" title="Fit (f)" onclick={fit}>
    <Icon icon={icons.fit} size={16} />
  </button>
  {#if glMissing}
    <div class="missing">WebGL2 is not available here; the field cannot be drawn.</div>
  {/if}
</div>

<style>
  .scene {
    position: relative;
    width: 100%;
    min-height: 240px;
    background: var(--canvas);
    border-radius: var(--radius);
    overflow: hidden;
    touch-action: none;
  }
  canvas {
    display: block;
    position: absolute;
    inset: 0;
  }
  .interact {
    position: absolute;
    inset: 0;
  }
  .overlay {
    position: absolute;
    inset: 0;
    display: block;
    cursor: grab;
    outline: none;
  }
  .scene[data-tool="line"] .overlay,
  .scene[data-tool="point"] .overlay {
    cursor: crosshair;
  }
  .overlay:focus-visible {
    box-shadow: inset 0 0 0 2px var(--focus);
  }
  .grid line {
    stroke: var(--canvas-grid);
  }
  .grid text {
    fill: var(--canvas-ink);
    font-family: var(--font-mono);
    font-size: 9px;
    opacity: 0.7;
  }
  .walls line {
    stroke: var(--canvas-wall);
    stroke-width: 1.5;
  }
  .walls .centre {
    stroke-dasharray: 4 6;
    stroke-width: 1;
  }
  .measured path {
    fill: none;
    stroke: var(--measured);
    stroke-width: 1.4;
    stroke-dasharray: 5 4;
  }
  .boundary path {
    fill: none;
    stroke-linejoin: round;
  }
  .boundary .halo {
    stroke-width: 3.4;
    stroke-opacity: 0.55;
  }
  .boundary .main {
    stroke-width: 1.5;
  }
  .vectors path {
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .vectors .halo {
    stroke: var(--canvas);
    stroke-width: 2.8;
    stroke-opacity: 0.55;
  }
  .vectors .main {
    stroke: var(--canvas-ink-strong);
    stroke-width: 1.1;
    stroke-opacity: 0.9;
  }
  .probes .line {
    fill: none;
    stroke-width: 1.8;
  }
  .probes .draft {
    stroke: var(--probe);
    stroke-dasharray: 4 4;
  }
  .probes .handle {
    fill: var(--canvas);
    stroke: var(--probe-handle);
    stroke-width: 1.5;
  }
  .probes text,
  .markers text {
    font-family: var(--font-ui);
    font-size: 11px;
    font-weight: 600;
    paint-order: stroke;
    stroke: var(--canvas);
    stroke-width: 3px;
  }
  .hud {
    position: absolute;
    left: var(--s3);
    bottom: var(--s3);
    display: flex;
    gap: var(--s3);
    padding: 6px 10px;
    border-radius: var(--radius-sm);
    background: color-mix(in srgb, var(--canvas) 70%, transparent);
    color: var(--canvas-ink);
    font-size: 11px;
    pointer-events: none;
    backdrop-filter: blur(6px);
  }
  .hud .value {
    color: var(--canvas-ink-strong);
    font-weight: 600;
  }
  .hint {
    position: absolute;
    top: var(--s3);
    left: 50%;
    transform: translateX(-50%);
    padding: 4px 10px;
    border-radius: 999px;
    background: color-mix(in srgb, var(--canvas) 70%, transparent);
    color: var(--canvas-ink);
    font-size: 11px;
    pointer-events: none;
  }
  .fit {
    position: absolute;
    right: var(--s3);
    bottom: var(--s3);
    background: color-mix(in srgb, var(--canvas) 70%, transparent);
    color: var(--canvas-ink);
  }
  .missing {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    color: var(--canvas-ink);
  }
</style>
