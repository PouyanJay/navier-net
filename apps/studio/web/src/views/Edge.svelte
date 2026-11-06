<script lang="ts">
  import { api } from "../lib/api/client";
  import type { EdgeResponse } from "../lib/api/types";
  import Colorbar from "../components/Colorbar.svelte";
  import Plot, { type Series } from "../components/Plot.svelte";
  import QuantityCard from "../components/QuantityCard.svelte";
  import QuantityPicker from "../components/QuantityPicker.svelte";
  import Scene from "../components/Scene.svelte";
  import { buildLut, defaultColormap, type ColormapName } from "../lib/colormaps";
  import { Icon, icons } from "../lib/icons";
  import { RunData, seriesColor } from "../lib/runData.svelte";
  import { fmt, niceRange } from "../lib/units";
  import { studio } from "../stores/studio.svelte";

  const data = new RunData();
  const quantities = $derived(
    studio
      .quantitiesOfKind("edge")
      .filter(
        (q) => !["edge_x", "edge_y", "arclength", "segment", "normal_x", "normal_y"].includes(q.id),
      ),
  );
  let quantity = $state("yl_jump");
  let responses = $state<Map<number, EdgeResponse>>(new Map());
  let error = $state<string | null>(null);

  const meta = $derived(studio.byId.get(quantity) ?? null);
  const frames = $derived(studio.selectedFrames);

  $effect(() => {
    const run = studio.runId;
    const q = quantity;
    const list = frames;
    if (!run || !studio.run?.has_front || list.length === 0) return;
    let cancelled = false;
    Promise.all(list.map((f) => api.getEdge(run, [q], f).then((r) => [f, r] as const)))
      .then((pairs) => {
        if (cancelled) return;
        responses = new Map(pairs);
        error = null;
      })
      .catch((e: Error) => {
        if (!cancelled) error = e.message;
      });
    return () => {
      cancelled = true;
    };
  });

  const series = $derived<Series[]>(
    frames.map((f, i) => {
      const r = responses.get(f);
      const s = (r?.values.arclength ?? []) as number[];
      const y = (r?.values[quantity] ?? []) as (number | null)[];
      const order = s.map((_, k) => k).sort((a, b) => s[a] - s[b]);
      return {
        name: `frame ${f}`,
        color: seriesColor(i),
        x: order.map((k) => s[k]),
        y: order.map((k) => y[k]),
      };
    }),
  );

  const current = $derived(responses.get(studio.frame) ?? null);
  const colormap = $derived<ColormapName>(defaultColormap(meta?.cmap ?? "diverging", studio.theme));
  const range = $derived.by((): [number, number] => {
    const values = series
      .flatMap((s) => s.y)
      .filter((v): v is number => v !== null && Number.isFinite(v));
    if (values.length === 0) return [0, 1];
    return niceRange(Math.min(...values), Math.max(...values), !!meta?.signed);
  });
  const lut = $derived(buildLut(colormap));

  function colorOf(value: number | null): string {
    if (value === null || !Number.isFinite(value)) return "transparent";
    const t = Math.min(1, Math.max(0, (value - range[0]) / (range[1] - range[0] || 1)));
    const i = Math.round(t * 255) * 4;
    return `rgb(${lut[i]} ${lut[i + 1]} ${lut[i + 2]})`;
  }

  const ring = $derived.by(() => {
    if (!current) return null;
    const xs = current.values.edge_x as number[];
    const ys = current.values.edge_y as number[];
    const vs = current.values[quantity] as (number | null)[];
    return { points: xs.map((x, i) => [x, ys[i]]), colors: vs.map(colorOf) };
  });

  const segmentStats = $derived.by(() => {
    if (!current) return [];
    const codes = current.values.segment as number[];
    const vs = current.values[quantity] as (number | null)[];
    return current.segments.map((name, k) => {
      const picked = vs.filter(
        (v, i) => codes[i] === k && v !== null && Number.isFinite(v),
      ) as number[];
      const rms = picked.length
        ? Math.sqrt(picked.reduce((a, v) => a + v * v, 0) / picked.length)
        : null;
      const mean = picked.length ? picked.reduce((a, v) => a + v, 0) / picked.length : null;
      return { name, n: picked.length, rms, mean };
    });
  });

  const walls = $derived(studio.run ? studio.run.domain_um.y_walls : null);
</script>

<div class="view">
  <div class="stage">
    <div class="toolbar">
      <div class="title">
        <strong>{meta?.label ?? quantity} along the bubble edge</strong>
        <span class="muted">frames {frames.join(", ")} (shift-click frames to compare)</span>
        {#if error}<span class="chip bad">{error}</span>{/if}
        {#if studio.run && !studio.run.has_front}<span class="chip bad"
            >this run has no explicit front</span
          >{/if}
      </div>
      <div class="actions">
        {#if studio.runId}
          <a class="btn" href={api.edgeCsvUrl(studio.runId, [quantity], studio.frame)} download>
            <Icon icon={icons.download} size={14} /> CSV (frame {studio.frame})
          </a>
        {/if}
      </div>
    </div>
    <div class="canvas">
      <Scene
        grid={null}
        {colormap}
        {range}
        unit={meta?.unit ?? ""}
        label={meta?.label ?? ""}
        extent={data.extent}
        boundary={data.contour?.rings_um ?? []}
        {walls}
        centreline={studio.run?.domain_um.centreline ?? null}
        showField={false}
        showBoundary={true}
        showWalls={studio.layers.walls}
        showGrid={studio.layers.grid}
        probes={[]}
        rings={ring}
        height={300}
      />
      <div class="legend">
        <Colorbar {colormap} {range} unit={meta?.unit ?? ""} label={meta?.label ?? ""} />
      </div>
    </div>
    <Plot
      title="{meta?.label ??
        quantity} against arc length from the root apex (upper edge, nose, lower edge)"
      xLabel="s (µm)"
      yLabel={meta?.symbol ?? quantity}
      unit={meta?.unit ?? ""}
      {series}
      height={260}
      filename="{studio.runId}_edge_{quantity}"
      meta={{ run: studio.runId, quantity }}
    />
  </div>
  <aside class="side">
    <QuantityPicker {quantities} selected={[quantity]} onSelect={(id) => (quantity = id)} />
    {#if current}
      <table class="table mono">
        <thead><tr><th>segment</th><th>n</th><th>mean</th><th>rms</th></tr></thead>
        <tbody>
          {#each segmentStats as s (s.name)}
            <tr><td>{s.name}</td><td>{s.n}</td><td>{fmt(s.mean, 3)}</td><td>{fmt(s.rms, 3)}</td></tr
            >
          {/each}
        </tbody>
      </table>
    {/if}
    {#if meta}<QuantityCard quantity={meta} />{/if}
  </aside>
</div>

<style>
  .view {
    display: grid;
    grid-template-columns: minmax(0, 1fr) var(--inspector-width);
    height: 100%;
    min-height: 0;
  }
  .stage {
    display: grid;
    grid-template-rows: auto auto minmax(0, 1fr);
    gap: var(--s3);
    padding: var(--s3);
    min-width: 0;
    overflow: auto;
  }
  .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--s3);
  }
  .title {
    display: flex;
    gap: var(--s3);
    align-items: center;
  }
  .canvas {
    position: relative;
  }
  .legend {
    position: absolute;
    top: var(--s3);
    right: var(--s3);
  }
  .side {
    display: grid;
    grid-template-rows: minmax(160px, 1fr) auto auto;
    gap: var(--s3);
    padding: var(--s3);
    border-left: 1px solid var(--line);
    background: var(--surface);
    overflow: auto;
    min-height: 0;
  }
</style>
