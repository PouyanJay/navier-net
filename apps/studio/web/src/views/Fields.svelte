<script lang="ts">
  import { api } from "../lib/api/client";
  import Colorbar from "../components/Colorbar.svelte";
  import Inspector from "../components/Inspector.svelte";
  import Scene from "../components/Scene.svelte";
  import { download } from "../lib/export";
  import { FieldState } from "../lib/fieldState.svelte";
  import { Icon, icons } from "../lib/icons";
  import { RunData } from "../lib/runData.svelte";
  import { fmt } from "../lib/units";
  import { GRID_NX, GRID_NY, studio } from "../stores/studio.svelte";

  const field = new FieldState();
  const data = new RunData();
  let scene = $state<Scene | null>(null);
  let multiples = $state(false);

  const quantities = $derived(studio.quantitiesOfKind("volume", "derived"));
  const walls = $derived(studio.run ? studio.run.domain_um.y_walls : null);
  const imageUrl = $derived(
    studio.layers.image && studio.runId
      ? api.frameUrl(studio.runId, studio.frame, studio.imageMode)
      : null,
  );

  async function exportPng() {
    if (!scene) return;
    const blob = await scene.toPng();
    download(`${studio.runId}_${studio.quantity}_f${studio.frame}.png`, blob);
  }

  function exportCsv() {
    if (!studio.runId) return;
    window.open(
      api.gridCsvUrl(studio.runId, [studio.quantity], studio.frame, GRID_NX, GRID_NY),
      "_blank",
    );
  }
</script>

<div class="view">
  <div class="stage">
    <div class="toolbar">
      <div class="title">
        <strong>{studio.current?.label ?? studio.quantity}</strong>
        <span class="muted mono"
          >frame {studio.frame}{studio.frameInfo?.t_ms != null
            ? ` · ${fmt(studio.frameInfo.t_ms, 3)} ms`
            : ""}</span
        >
        {#if studio.frameInfo?.role === "holdout"}<span class="chip holdout">held out</span>{/if}
        {#if field.busy}<span class="muted">evaluating…</span>{/if}
        {#if field.error}<span class="chip bad">{field.error}</span>{/if}
      </div>
      <div class="actions">
        <div class="segmented" role="group" aria-label="Layout">
          <button aria-pressed={!multiples} onclick={() => (multiples = false)}>one frame</button>
          <button aria-pressed={multiples} onclick={() => (multiples = true)}>all frames</button>
        </div>
        <button class="btn" onclick={exportCsv}><Icon icon={icons.download} size={14} /> CSV</button
        >
        <button class="btn" onclick={exportPng}><Icon icon={icons.download} size={14} /> PNG</button
        >
      </div>
    </div>
    {#if !multiples}
      <div class="canvas">
        <Scene
          bind:this={scene}
          grid={field.grid}
          colormap={field.colormap}
          range={field.range}
          unit={studio.current?.unit ?? ""}
          label={studio.current?.label ?? ""}
          extent={data.extent}
          boundary={field.grid?.header.boundary_um ?? data.contour?.rings_um ?? []}
          measured={data.measured}
          {walls}
          centreline={studio.run?.domain_um.centreline ?? null}
          {imageUrl}
          transform={data.transform}
          imageOpacity={studio.imageOpacity}
          showField={studio.layers.field}
          showBoundary={studio.layers.boundary}
          showMeasured={studio.layers.measured}
          showWalls={studio.layers.walls}
          showGrid={studio.layers.grid}
          probes={studio.layers.probes ? studio.probes : []}
        />
        <div class="legend">
          <Colorbar
            colormap={field.colormap}
            range={field.range}
            unit={studio.current?.unit ?? ""}
            label={studio.current?.label ?? ""}
          />
        </div>
      </div>
    {:else}
      <div class="multiples scroll">
        {#each studio.eventFrames as f (f.frame)}
          {#await studio.runId ? api.getGrid(studio.runId, [studio.quantity], f.frame, 256, 86) : Promise.reject()}
            <div class="tile pending">frame {f.frame}</div>
          {:then grid}
            <div class="tile">
              <div class="tile-title mono">
                <span>frame {f.frame} · {fmt(f.t_ms, 2)} ms</span>
                {#if f.role === "holdout"}<span class="chip holdout">held out</span>{/if}
              </div>
              <Scene
                {grid}
                colormap={field.colormap}
                range={studio.rangeMode === "frame" ? field.range : field.range}
                unit={studio.current?.unit ?? ""}
                label={studio.current?.label ?? ""}
                extent={data.extent}
                boundary={grid.header.boundary_um}
                {walls}
                centreline={studio.run?.domain_um.centreline ?? null}
                showBoundary={studio.layers.boundary}
                showWalls={studio.layers.walls}
                height={170}
              />
            </div>
          {:catch}
            <div class="tile pending">frame {f.frame}: unavailable</div>
          {/await}
        {/each}
      </div>
    {/if}
  </div>
  <Inspector {quantities} range={field.range} colormap={field.colormap} />
</div>

<style>
  .view {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    height: 100%;
    min-height: 0;
  }
  .stage {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    min-width: 0;
    padding: var(--s3);
    gap: var(--s3);
  }
  .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--s3);
  }
  .title {
    display: flex;
    align-items: center;
    gap: var(--s3);
    font-size: 13px;
  }
  .actions {
    display: flex;
    gap: var(--s2);
    align-items: center;
  }
  .canvas {
    position: relative;
    min-height: 0;
  }
  .legend {
    position: absolute;
    top: var(--s3);
    right: var(--s3);
  }
  .multiples {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: var(--s3);
    align-content: start;
    min-height: 0;
  }
  .tile {
    display: grid;
    gap: 4px;
  }
  .tile-title {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--muted);
  }
  .pending {
    height: 170px;
    display: grid;
    place-items: center;
    color: var(--muted);
    background: var(--canvas);
    border-radius: var(--radius);
  }
</style>
