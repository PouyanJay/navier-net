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
    <span class="mark"></span>
  </div>
  <ul>
    {#each VIEWS as view (view.id)}
      <li>
        <button
          class="icon-btn"
          class:active={studio.view === view.id}
          aria-label={view.label}
          aria-current={studio.view === view.id ? "page" : undefined}
          title={view.label}
          onclick={() => (studio.view = view.id)}
        >
          <Icon icon={icons[view.id]} size={19} />
        </button>
        <span class="label">{view.label}</span>
      </li>
    {/each}
  </ul>
  <div class="bottom">
    <div class="swatches" role="radiogroup" aria-label="Palette">
      {#each palettes as p (p.id)}
        <button
          class="swatch {p.id}"
          role="radio"
          aria-checked={studio.palette === p.id}
          aria-label={p.label}
          title={p.label}
          onclick={() => studio.setPalette(p.id)}
        ></button>
      {/each}
    </div>
    <button
      class="icon-btn"
      aria-label={studio.theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      title={studio.theme === "dark" ? "Light mode" : "Dark mode"}
      onclick={() => studio.setTheme(studio.theme === "dark" ? "light" : "dark")}
    >
      <Icon icon={studio.theme === "dark" ? icons.light : icons.dark} size={18} />
    </button>
  </div>
</nav>

<style>
  .rail {
    width: var(--rail-width);
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: var(--s3) 0;
    gap: var(--s4);
    background: var(--surface);
    border-right: 1px solid var(--line);
  }
  .brand {
    display: grid;
    place-items: center;
    width: 36px;
    height: 36px;
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
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: var(--s1);
    flex: 1;
  }
  li {
    display: grid;
    justify-items: center;
    gap: 2px;
  }
  .label {
    font-size: 9.5px;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: var(--faint);
    text-transform: uppercase;
  }
  .bottom {
    display: grid;
    gap: var(--s3);
    justify-items: center;
  }
  .swatches {
    display: grid;
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
</style>
