<script lang="ts">
  import { fmt } from "../lib/units";
  import { studio } from "../stores/studio.svelte";

  const frames = $derived(studio.run?.frames ?? []);

  function onClick(event: MouseEvent, frame: number, role: string) {
    if (role === "excluded") return;
    if (event.shiftKey || event.metaKey || event.ctrlKey) studio.toggleFrame(frame);
    else studio.frame = frame;
  }
</script>

<div class="strip" role="toolbar" aria-label="Frames (shift-click to select several)">
  {#each frames as f (f.frame)}
    <button
      class="frame {f.role}"
      class:active={studio.frame === f.frame}
      class:picked={studio.frames.includes(f.frame)}
      disabled={f.role === "excluded"}
      aria-pressed={studio.frame === f.frame}
      title={f.t_ms === null
        ? `frame ${f.frame}: ${f.role}`
        : `frame ${f.frame} · ${fmt(f.t_ms, 2)} ms · ${f.role}`}
      onclick={(e) => onClick(e, f.frame, f.role)}
    >
      <span class="n mono">{f.frame}</span>
      {#if f.role === "holdout"}<span class="dot holdout" aria-hidden="true"></span>{/if}
      {#if f.role === "truncated"}<span class="dot truncated" aria-hidden="true"></span>{/if}
    </button>
  {/each}
</div>

<style>
  .strip {
    display: flex;
    gap: 3px;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .frame {
    position: relative;
    min-width: 34px;
    height: 30px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--line);
    color: var(--muted);
    display: grid;
    place-items: center;
    transition:
      border-color var(--duration) var(--ease),
      color var(--duration) var(--ease),
      background var(--duration) var(--ease);
  }
  .frame:hover:not(:disabled) {
    color: var(--text);
    background: var(--surface-hover);
  }
  .frame.active {
    color: var(--primary-ink);
    background: var(--primary);
    border-color: var(--primary);
  }
  .frame.picked:not(.active) {
    border-color: color-mix(in srgb, var(--primary) 70%, var(--line));
    box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--primary) 50%, transparent);
    color: var(--text);
  }
  .frame.truncated {
    opacity: 0.6;
  }
  .n {
    font-size: 12px;
    font-weight: 600;
  }
  .dot {
    position: absolute;
    top: 3px;
    right: 3px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }
  .dot.holdout {
    background: var(--holdout);
  }
  .dot.truncated {
    background: var(--warn);
  }
</style>
