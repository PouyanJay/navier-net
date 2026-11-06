<script lang="ts">
  import { SvelteMap } from "svelte/reactivity";
  import type { Quantity } from "../lib/api/types";
  import { Icon, icons } from "../lib/icons";

  let {
    quantities,
    selected,
    onSelect,
    multiple = false,
    onToggle,
  }: {
    quantities: Quantity[];
    selected: string[];
    onSelect?: (id: string) => void;
    multiple?: boolean;
    onToggle?: (id: string) => void;
  } = $props();

  let query = $state("");

  const groups = $derived.by(() => {
    const needle = query.trim().toLowerCase();
    const out = new SvelteMap<string, Quantity[]>();
    for (const q of quantities) {
      if (needle && !`${q.label} ${q.symbol} ${q.id} ${q.group}`.toLowerCase().includes(needle))
        continue;
      const list = out.get(q.group) ?? [];
      list.push(q);
      out.set(q.group, list);
    }
    return [...out.entries()];
  });

  function pick(q: Quantity) {
    if (q.available === false) return;
    if (multiple) onToggle?.(q.id);
    else onSelect?.(q.id);
  }
</script>

<div class="picker">
  <label class="search">
    <Icon icon={icons.search} size={14} />
    <input
      class="input"
      placeholder="Find a quantity"
      bind:value={query}
      aria-label="Find a quantity"
    />
  </label>
  <div class="groups scroll" role={multiple ? "group" : "listbox"} aria-label="Quantities">
    {#each groups as [group, list] (group)}
      <section>
        <h4>{group}</h4>
        <ul>
          {#each list as q (q.id)}
            <li>
              <button
                class="item"
                class:selected={selected.includes(q.id)}
                class:unavailable={q.available === false}
                role={multiple ? "checkbox" : "option"}
                aria-checked={multiple ? selected.includes(q.id) : undefined}
                aria-selected={multiple ? undefined : selected.includes(q.id)}
                aria-disabled={q.available === false}
                title={q.available === false ? "This run cannot produce this quantity" : q.plain}
                onclick={() => pick(q)}
              >
                <span class="symbol mono">{q.symbol || q.id}</span>
                <span class="label">{q.label}</span>
                <span class="unit mono">{q.unit}</span>
              </button>
            </li>
          {/each}
        </ul>
      </section>
    {/each}
  </div>
</div>

<style>
  .picker {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    gap: var(--s2);
    min-height: 0;
  }
  .search {
    display: flex;
    align-items: center;
    gap: var(--s2);
    padding: 0 var(--s2);
    color: var(--muted);
  }
  .search .input {
    border: 0;
    background: transparent;
    padding: 0;
    height: 30px;
  }
  .search .input:focus {
    outline: none;
  }
  .groups {
    min-height: 0;
    display: grid;
    gap: var(--s3);
    align-content: start;
  }
  h4 {
    margin: 0 0 var(--s1);
    padding: 0 var(--s2);
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--faint);
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 1px;
  }
  .item {
    width: 100%;
    display: grid;
    grid-template-columns: 44px minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--s2);
    padding: 6px var(--s2);
    border-radius: var(--radius-sm);
    text-align: left;
    color: var(--text-soft);
    transition: background var(--duration) var(--ease);
  }
  .item:hover {
    background: var(--surface-hover);
  }
  .item.selected {
    background: color-mix(in srgb, var(--primary) 16%, transparent);
    color: var(--text);
  }
  .item.unavailable {
    opacity: 0.38;
    cursor: not-allowed;
  }
  .symbol {
    font-size: 12px;
    color: var(--primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .label {
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .unit {
    font-size: 10.5px;
    color: var(--muted);
  }
</style>
