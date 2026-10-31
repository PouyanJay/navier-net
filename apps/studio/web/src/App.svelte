<script lang="ts">
  import Rail from "./components/Rail.svelte";
  import TopBar from "./components/TopBar.svelte";
  import { parseFrames, readHash, writeHash } from "./lib/url";
  import { studio, VIEWS, type ViewId } from "./stores/studio.svelte";
  import Budgets from "./views/Budgets.svelte";
  import Edge from "./views/Edge.svelte";
  import Export from "./views/Export.svelte";
  import Fields from "./views/Fields.svelte";
  import Geometry from "./views/Geometry.svelte";
  import Probes from "./views/Probes.svelte";
  import Residuals from "./views/Residuals.svelte";

  const PLAY_INTERVAL_MS = 450;
  let ready = $state(false);

  // Restore the shareable state from the URL, then load the runs.
  $effect(() => {
    const state = readHash();
    if (state.view && VIEWS.some((v) => v.id === state.view)) studio.view = state.view as ViewId;
    if (state.q) studio.quantity = state.q;
    if (state.frame) studio.frame = Number(state.frame);
    if (state.frames) studio.frames = parseFrames(state.frames);
    if (state.cmap) studio.colormap = state.cmap as typeof studio.colormap;
    if (state.range === "all" || state.range === "frame") studio.rangeMode = state.range;
    const run = state.run;
    studio.loadRuns().then(() => {
      if (run) studio.selectRun(run);
      ready = true;
    });
  });

  // Write it back whenever it changes.
  $effect(() => {
    if (!ready) return;
    writeHash({
      run: studio.runId ?? "",
      view: studio.view,
      q: studio.quantity,
      frame: String(studio.frame),
      frames: studio.frames.join(","),
      cmap: studio.colormap === "auto" ? "" : studio.colormap,
      range: studio.rangeMode === "frame" ? "" : studio.rangeMode,
    });
  });

  $effect(() => {
    if (!studio.playing) return;
    const timer = setInterval(() => studio.stepFrame(1), PLAY_INTERVAL_MS);
    return () => clearInterval(timer);
  });

  function onKey(event: KeyboardEvent) {
    const target = event.target as HTMLElement | null;
    if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
    if (event.key === "ArrowRight") studio.stepFrame(1);
    if (event.key === "ArrowLeft") studio.stepFrame(-1);
    if (event.key === " ") {
      event.preventDefault();
      studio.playing = !studio.playing;
    }
  }
</script>

<svelte:window onkeydown={onKey} />

<div class="app">
  <Rail />
  <div class="main">
    <TopBar />
    <main class="content">
      {#if studio.error}
        <div class="notice bad">{studio.error}</div>
      {:else if !studio.run}
        <div class="notice">
          {#if studio.runs.length === 0 && ready}
            No runs with a checkpoint under outputs/. Train one, then reload.
          {:else}
            Loading the run…
          {/if}
        </div>
      {:else if studio.view === "geometry"}
        <Geometry />
      {:else if studio.view === "fields"}
        <Fields />
      {:else if studio.view === "edge"}
        <Edge />
      {:else if studio.view === "residuals"}
        <Residuals />
      {:else if studio.view === "probes"}
        <Probes />
      {:else if studio.view === "budgets"}
        <Budgets />
      {:else}
        <Export />
      {/if}
    </main>
  </div>
</div>

<style>
  .app {
    display: grid;
    grid-template-columns: var(--rail-width) minmax(0, 1fr);
    height: 100dvh;
    background: var(--bg);
  }
  .main {
    display: grid;
    grid-template-rows: var(--topbar-height) minmax(0, 1fr);
    min-width: 0;
  }
  .content {
    min-height: 0;
    min-width: 0;
  }
  .notice {
    margin: var(--s6);
    padding: var(--s4);
    border-radius: var(--radius);
    border: 1px solid var(--line);
    color: var(--text-soft);
  }
  .notice.bad {
    border-color: color-mix(in srgb, var(--bad) 50%, var(--line));
    color: var(--bad);
  }
</style>
