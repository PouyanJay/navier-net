<script lang="ts">
  /**
   * A listbox that matches the studio's surfaces. The native select drew the
   * OS menu, which ignored the tokens; this one keeps the menu inside the
   * design: a raised surface, the primary tint on the chosen row, keyboard
   * support (arrows, Home, End, Enter, Escape) and one tab stop.
   */
  import { Icon, icons } from "../lib/icons";

  interface Option {
    value: string;
    label: string;
    hint?: string;
  }

  let {
    options,
    value,
    onChange,
    label,
    width = "100%",
  }: {
    options: Option[];
    value: string;
    onChange: (value: string) => void;
    label: string;
    width?: string;
  } = $props();

  let open = $state(false);
  let active = $state(0);
  let root = $state<HTMLDivElement | null>(null);
  const current = $derived(options.find((o) => o.value === value) ?? options[0] ?? null);

  function openMenu() {
    active = Math.max(
      0,
      options.findIndex((o) => o.value === value),
    );
    open = true;
  }

  function choose(index: number) {
    const option = options[index];
    if (option) onChange(option.value);
    open = false;
  }

  function onKey(event: KeyboardEvent) {
    if (!open) {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
        event.preventDefault();
        openMenu();
      }
      return;
    }
    if (event.key === "ArrowDown") active = Math.min(options.length - 1, active + 1);
    else if (event.key === "ArrowUp") active = Math.max(0, active - 1);
    else if (event.key === "Home") active = 0;
    else if (event.key === "End") active = options.length - 1;
    else if (event.key === "Enter" || event.key === " ") choose(active);
    else if (event.key === "Escape" || event.key === "Tab") open = false;
    else return;
    event.preventDefault();
  }

  function onWindowPointer(event: PointerEvent) {
    if (open && root && !root.contains(event.target as Node)) open = false;
  }
</script>

<svelte:window onpointerdown={onWindowPointer} />

<div class="listbox" bind:this={root} style:width>
  <button
    type="button"
    class="trigger"
    aria-haspopup="listbox"
    aria-expanded={open}
    aria-label={label}
    onclick={() => (open ? (open = false) : openMenu())}
    onkeydown={onKey}
  >
    <span class="value">{current?.label ?? "—"}</span>
    <span class="chevron" class:flipped={open} aria-hidden="true">
      <Icon icon={icons.chevron} size={14} />
    </span>
  </button>
  {#if open}
    <ul class="menu" role="listbox" aria-label={label} tabindex="-1" onkeydown={onKey}>
      {#each options as option, i (option.value)}
        <li
          role="option"
          aria-selected={option.value === value}
          class:active={i === active}
          class:chosen={option.value === value}
          onpointerenter={() => (active = i)}
          onpointerdown={(e) => {
            e.preventDefault();
            choose(i);
          }}
        >
          <span class="text">{option.label}</span>
          {#if option.hint}<span class="hint mono">{option.hint}</span>{/if}
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .listbox {
    position: relative;
    min-width: 0;
  }
  .trigger {
    width: 100%;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--s2);
    padding: 0 var(--s3);
    border-radius: var(--radius-sm);
    border: 1px solid var(--line);
    background: var(--surface);
    color: var(--text);
    font-size: 13px;
    text-align: left;
    transition:
      border-color var(--duration) var(--ease),
      box-shadow var(--duration) var(--ease);
  }
  .trigger:hover {
    border-color: var(--line-strong);
  }
  .trigger[aria-expanded="true"],
  .trigger:focus-visible {
    border-color: color-mix(in srgb, var(--primary) 55%, var(--line));
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--primary) 35%, transparent);
    outline: none;
  }
  .value {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-family: var(--font-mono);
    font-size: 12.5px;
  }
  .chevron {
    display: grid;
    color: var(--muted);
    transition: transform var(--duration) var(--ease);
  }
  .chevron.flipped {
    transform: rotate(180deg);
  }
  .menu {
    position: absolute;
    z-index: 30;
    top: calc(100% + 4px);
    left: 0;
    min-width: 100%;
    max-height: 320px;
    overflow: auto;
    margin: 0;
    padding: var(--s1);
    list-style: none;
    display: grid;
    gap: 2px;
    border-radius: var(--radius);
    border: 1px solid var(--line-strong);
    background: var(--surface-raised);
    box-shadow: var(--shadow);
    overscroll-behavior: contain;
  }
  li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--s3);
    padding: 6px var(--s2);
    border-radius: var(--radius-sm);
    color: var(--text-soft);
    font-size: 13px;
    white-space: nowrap;
    cursor: pointer;
  }
  li.active {
    background: var(--surface-hover);
    color: var(--text);
  }
  li.chosen {
    color: var(--primary);
    font-weight: 600;
  }
  .text {
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .hint {
    font-size: 11px;
    color: var(--muted);
  }
</style>
