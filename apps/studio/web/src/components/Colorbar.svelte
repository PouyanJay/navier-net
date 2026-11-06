<script lang="ts">
  import { cssGradient, type ColormapName } from "../lib/colormaps";
  import { fmt } from "../lib/units";

  let {
    colormap,
    range,
    unit,
    label,
  }: { colormap: ColormapName; range: [number, number]; unit: string; label: string } = $props();

  const ticks = $derived.by(() => {
    const [lo, hi] = range;
    return [0, 0.25, 0.5, 0.75, 1].map((f) => ({ f, value: lo + (hi - lo) * f }));
  });
</script>

<div
  class="colorbar"
  role="img"
  aria-label="{label} from {fmt(range[0])} to {fmt(range[1])} {unit}"
>
  <div class="title">
    <span class="label">{label}</span>
    {#if unit}<span class="unit mono">{unit}</span>{/if}
  </div>
  <div class="bar" style:background={cssGradient(colormap, 32)}></div>
  <div class="ticks mono">
    {#each ticks as t (t.f)}
      <span style:left="{t.f * 100}%">{fmt(t.value, 3)}</span>
    {/each}
  </div>
</div>

<style>
  .colorbar {
    display: grid;
    gap: 4px;
    width: 260px;
    padding: var(--s2) var(--s3);
    border-radius: var(--radius-sm);
    background: color-mix(in srgb, var(--canvas) 72%, transparent);
    backdrop-filter: blur(6px);
    color: var(--canvas-ink);
  }
  .title {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    font-weight: 600;
  }
  .unit {
    color: var(--canvas-ink);
    opacity: 0.8;
  }
  .bar {
    height: 10px;
    border-radius: 3px;
  }
  .ticks {
    position: relative;
    height: 14px;
    font-size: 10px;
  }
  .ticks span {
    position: absolute;
    transform: translateX(-50%);
    white-space: nowrap;
  }
  .ticks span:first-child {
    transform: none;
  }
  .ticks span:last-child {
    transform: translateX(-100%);
  }
</style>
