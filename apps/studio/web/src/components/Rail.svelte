<script lang="ts">
  import { Icon, icons } from "../lib/icons";
  import { studio, VIEWS, type Palette } from "../stores/studio.svelte";

  const palettes: { id: Palette; label: string }[] = [
    { id: "aurora", label: "Aurora" },
    { id: "ember", label: "Ember" },
    { id: "tide", label: "Tide" },
  ];
</script>

<nav class="rail" aria-label="Studio views">
  <div class="brand" title="naviernet studio">
    <span class="mark" aria-hidden="true"></span>
  </div>

  <ul class="views">
    {#each VIEWS as view (view.id)}
      <li>
        <button
          class="item"
          class:active={studio.view === view.id}
          aria-label={view.label}
          aria-current={studio.view === view.id ? "page" : undefined}
          title={view.label}
          onclick={() => (studio.view = view.id)}
        >
          <span class="pill"><Icon icon={icons[view.id]} size={19} /></span>
          <span class="label">{view.label}</span>
        </button>
      </li>
    {/each}
  </ul>

  <div class="footer">
    <div class="palettes" role="radiogroup" aria-label="Palette">
      {#each palettes as p (p.id)}
        <button
          class="swatch {p.id}"
          role="radio"
          aria-checked={studio.palette === p.id}
          aria-label="{p.label} palette"
          title="{p.label} palette"
          onclick={() => studio.setPalette(p.id)}
        ></button>
      {/each}
    </div>
    <button
      class="theme"
      aria-label={studio.theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      title={studio.theme === "dark" ? "Light mode" : "Dark mode"}
      onclick={() => studio.setTheme(studio.theme === "dark" ? "light" : "dark")}
    >
      <Icon icon={studio.theme === "dark" ? icons.light : icons.dark} size={17} />
    </button>
  </div>
</nav>

<style>
  .rail {
    width: var(--rail-width);
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    padding: var(--s3) var(--s2);
    background: var(--surface);
    border-right: 1px solid var(--line);
    min-height: 0;
  }
  .brand {
    display: grid;
    place-items: center;
    height: var(--topbar-height);
    margin: calc(var(--s3) * -1) calc(var(--s2) * -1) var(--s2);
    border-bottom: 1px solid var(--line);
  }
  .mark {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background:
      radial-gradient(circle at 35% 35%, var(--primary) 0 35%, transparent 36%),
      conic-gradient(from 90deg, var(--primary), var(--series-2), var(--primary));
    box-shadow: 0 0 18px color-mix(in srgb, var(--primary) 50%, transparent);
  }
  .views {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: var(--s2);
    align-content: start;
    flex: 1;
    min-height: 0;
    overflow: auto;
  }
  .item {
    width: 100%;
    display: grid;
    justify-items: center;
    gap: 4px;
    padding: 0;
    color: var(--muted);
    border-radius: var(--radius-sm);
  }
  .item:hover {
    color: var(--text);
  }
  .pill {
    display: grid;
    place-items: center;
    width: 44px;
    height: 32px;
    border-radius: 999px;
    transition:
      background var(--duration) var(--ease),
      color var(--duration) var(--ease);
  }
  .item:hover .pill {
    background: var(--surface-hover);
  }
  .item.active {
    color: var(--text);
  }
  .item.active .pill {
    background: color-mix(in srgb, var(--primary) 18%, transparent);
    color: var(--primary);
  }
  .label {
    font-size: 11px;
    font-weight: 500;
    line-height: 1.1;
    white-space: nowrap;
  }
  .item.active .label {
    color: var(--text);
    font-weight: 600;
  }
  .footer {
    display: grid;
    justify-items: center;
    gap: var(--s3);
    padding-top: var(--s3);
    border-top: 1px solid var(--line);
  }
  .palettes {
    display: flex;
    gap: 6px;
  }
  .swatch {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 2px solid transparent;
    transition: border-color var(--duration) var(--ease);
  }
  .swatch.aurora {
    background: linear-gradient(135deg, #5eead4, #c084fc);
  }
  .swatch.ember {
    background: linear-gradient(135deg, #fb923c, #f472b6);
  }
  .swatch.tide {
    background: linear-gradient(135deg, #60a5fa, #34d399);
  }
  .swatch[aria-checked="true"] {
    border-color: var(--text);
  }
  .theme {
    display: grid;
    place-items: center;
    width: 36px;
    height: 36px;
    border-radius: var(--radius-sm);
    color: var(--muted);
  }
  .theme:hover {
    background: var(--surface-hover);
    color: var(--text);
  }
</style>
