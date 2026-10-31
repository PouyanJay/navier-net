<script lang="ts">
  /** The right-hand panel of the field views: quantity, look, layers, card. */
  import type { Quantity } from "../lib/api/types";
  import { cssGradient, SEQUENTIAL, type ColormapName } from "../lib/colormaps";
  import { fmt } from "../lib/units";
  import { studio } from "../stores/studio.svelte";
  import QuantityCard from "./QuantityCard.svelte";
  import QuantityPicker from "./QuantityPicker.svelte";

  let {
    quantities,
    range,
    colormap,
  }: { quantities: Quantity[]; range: [number, number]; colormap: ColormapName } = $props();

  const maps: ColormapName[] = [...SEQUENTIAL, "diverging", "diverging-light"];

  function setCustom(index: 0 | 1, text: string) {
    const value = Number.parseFloat(text);
    if (!Number.isFinite(value)) return;
    const next: [number, number] = [...studio.customRange];
    next[index] = value;
    studio.customRange = next;
    studio.rangeMode = "custom";
  }
</script>

<aside class="inspector">
  <section class="block picker">
    <QuantityPicker
      {quantities}
      selected={[studio.quantity]}
      onSelect={(id) => (studio.quantity = id)}
    />
  </section>

  <section class="block">
    <h4>Colour</h4>
    <div class="maps" role="radiogroup" aria-label="Colormap">
      <button
        class="map"
        role="radio"
        aria-checked={studio.colormap === "auto"}
        title="The catalogue's default for this quantity"
        onclick={() => (studio.colormap = "auto")}
      >
        <span class="swatch" style:background={cssGradient(colormap)}></span><span>auto</span>
      </button>
      {#each maps as m (m)}
        <button
          class="map"
          role="radio"
          aria-checked={studio.colormap === m}
          title={m}
          onclick={() => (studio.colormap = m)}
        >
          <span class="swatch" style:background={cssGradient(m)}></span><span>{m}</span>
        </button>
      {/each}
    </div>
    <h4>Range</h4>
    <div class="segmented" role="group" aria-label="Range rule">
      <button
        aria-pressed={studio.rangeMode === "frame"}
        onclick={() => (studio.rangeMode = "frame")}>this frame</button
      >
      <button aria-pressed={studio.rangeMode === "all"} onclick={() => (studio.rangeMode = "all")}
        >all frames</button
      >
      <button
        aria-pressed={studio.rangeMode === "custom"}
        onclick={() => (studio.rangeMode = "custom")}>custom</button
      >
    </div>
    <div class="range mono">
      <input
        class="input"
        type="number"
        step="any"
        value={fmt(range[0], 4)}
        aria-label="Minimum"
        onchange={(e) => setCustom(0, e.currentTarget.value)}
      />
      <span>to</span>
      <input
        class="input"
        type="number"
        step="any"
        value={fmt(range[1], 4)}
        aria-label="Maximum"
        onchange={(e) => setCustom(1, e.currentTarget.value)}
      />
    </div>
  </section>

  <section class="block">
    <h4>Layers</h4>
    <div class="layers">
      <label><input type="checkbox" bind:checked={studio.layers.field} /> Field</label>
      <label><input type="checkbox" bind:checked={studio.layers.image} /> Camera frame</label>
      <label><input type="checkbox" bind:checked={studio.actualImage} /> Actual image</label>
      <label><input type="checkbox" bind:checked={studio.layers.boundary} /> Bubble boundary</label>
      <label><input type="checkbox" bind:checked={studio.layers.measured} /> Measured mask</label>
      <label
        ><input type="checkbox" bind:checked={studio.layers.walls} /> Walls and centreline</label
      >
      <label><input type="checkbox" bind:checked={studio.layers.probes} /> Probes</label>
      <label><input type="checkbox" bind:checked={studio.layers.grid} /> µm grid</label>
      {#if studio.layers.image}
        <label class="slider">
          Frame opacity
          <input type="range" min="0" max="1" step="0.05" bind:value={studio.imageOpacity} />
        </label>
      {/if}
    </div>
  </section>

  {#if studio.current}
    <section class="block">
      <QuantityCard quantity={studio.current} />
    </section>
  {/if}
</aside>

<style>
  .inspector {
    width: var(--inspector-width);
    height: 100%;
    display: grid;
    grid-template-rows: minmax(180px, 1fr) auto auto auto;
    gap: var(--s3);
    padding: var(--s3);
    border-left: 1px solid var(--line);
    background: var(--surface);
    overflow: auto;
    overscroll-behavior: contain;
  }
  .block {
    display: grid;
    gap: var(--s2);
    min-height: 0;
  }
  .picker {
    min-height: 0;
  }
  h4 {
    margin: var(--s1) 0 0;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--faint);
  }
  .maps {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 6px;
  }
  .map {
    display: grid;
    gap: 4px;
    padding: 5px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--line);
    font-size: 10.5px;
    color: var(--muted);
    text-align: left;
  }
  .map[aria-checked="true"] {
    border-color: color-mix(in srgb, var(--primary) 60%, var(--line));
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--primary) 35%, transparent);
    color: var(--text);
  }
  .swatch {
    display: block;
    height: 10px;
    border-radius: 3px;
  }
  .range {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
    align-items: center;
    gap: var(--s2);
    font-size: 12px;
    color: var(--muted);
  }
  .layers {
    display: grid;
    gap: 6px;
    font-size: 12px;
    color: var(--text-soft);
  }
  .layers label {
    display: flex;
    align-items: center;
    gap: var(--s2);
  }
  .layers input[type="checkbox"] {
    accent-color: var(--primary);
  }
  .slider {
    display: grid !important;
    gap: 4px;
  }
  .slider input {
    accent-color: var(--primary);
  }
</style>
