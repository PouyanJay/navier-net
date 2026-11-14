<script lang="ts">
  import Select from "./Select.svelte";
  import { Icon, icons } from "../lib/icons";
  import { fmt } from "../lib/units";
  import { studio } from "../stores/studio.svelte";
  import FrameStrip from "./FrameStrip.svelte";

  const summary = $derived(studio.runs.find((r) => r.id === studio.runId) ?? null);
  const conditions = $derived(studio.run?.conditions ?? null);
</script>

<header class="topbar">
  <div class="run">
    <div class="field">
      <span class="field-label">Run</span>
      <Select
        label="Run"
        value={studio.runId ?? ""}
        options={studio.runs.map((r) => ({ value: r.id, label: r.id }))}
        onChange={(id) => studio.selectRun(id)}
        width="240px"
      />
    </div>
    {#if conditions}
      <div class="chips" aria-label="Run conditions">
        <span class="chip mono" title="Inlet flow rate"
          >{fmt(conditions.flow_rate_mL_hr, 3)} mL/h</span
        >
        {#if conditions.T_init_C !== null}
          <span class="chip mono" title="Liquid temperature">{fmt(conditions.T_init_C, 3)} °C</span>
        {/if}
        <span class="chip mono" title="Wall heat flux">{fmt(conditions.q_wall_W_cm2, 2)} W/cm²</span
        >
        {#if summary?.iou_mean != null}
          <span class="chip mono" title="Mean IoU over the event frames"
            >IoU {fmt(summary.iou_mean, 3)}</span
          >
        {/if}
      </div>
    {/if}
  </div>
  <div class="frames">
    <button class="icon-btn" aria-label="Previous frame" onclick={() => studio.stepFrame(-1)}>
      <Icon icon={icons.prev} size={16} />
    </button>
    <button
      class="icon-btn"
      class:active={studio.playing}
      aria-label={studio.playing ? "Pause" : "Play frames"}
      aria-pressed={studio.playing}
      onclick={() => (studio.playing = !studio.playing)}
    >
      <Icon icon={studio.playing ? icons.pause : icons.play} size={16} />
    </button>
    <button class="icon-btn" aria-label="Next frame" onclick={() => studio.stepFrame(1)}>
      <Icon icon={icons.next} size={16} />
    </button>
    <FrameStrip />
  </div>
</header>

<style>
  .topbar {
    height: var(--topbar-height);
    display: flex;
    align-items: center;
    gap: var(--s5);
    padding: 0 var(--s4);
    border-bottom: 1px solid var(--line);
    background: var(--surface);
  }
  .run {
    display: flex;
    align-items: center;
    gap: var(--s3);
    min-width: 0;
  }
  .field {
    display: flex;
    align-items: center;
    gap: var(--s2);
  }
  .field-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .chips {
    display: flex;
    gap: var(--s2);
  }
  .frames {
    display: flex;
    align-items: center;
    gap: var(--s1);
    flex: 1;
    min-width: 0;
  }
</style>
