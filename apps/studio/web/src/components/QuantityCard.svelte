<script lang="ts">
  import type { Quantity } from "../lib/api/types";
  import { tex } from "../lib/katex";
  import { fmt } from "../lib/units";

  let { quantity, compact = false }: { quantity: Quantity; compact?: boolean } = $props();
</script>

<article class="card" class:compact aria-label="About {quantity.label}">
  <header>
    <div>
      <h3>{quantity.label}</h3>
      <div class="meta mono">
        {#if quantity.symbol}<span>{quantity.symbol}</span>{/if}
        {#if quantity.unit}<span class="unit">{quantity.unit}</span>{:else}<span class="unit"
            >dimensionless</span
          >{/if}
      </div>
    </div>
    <span class="kind">{quantity.kind}</span>
  </header>
  {#if quantity.tex}
    <div class="formula">{@html tex(quantity.tex, true)}</div>
  {/if}
  <p class="plain">{quantity.plain}</p>
  {#if !compact}
    <dl>
      <dt>How it is computed</dt>
      <dd>{quantity.how}</dd>
      {#if quantity.factor !== undefined && quantity.scale}
        <dt>Conversion</dt>
        <dd class="mono">model units × {fmt(quantity.factor, 5)} = {quantity.unit}</dd>
      {/if}
      {#if quantity.caveats.length > 0}
        <dt>Keep in mind</dt>
        {#each quantity.caveats as caveat (caveat)}
          <dd class="caveat">{caveat}</dd>
        {/each}
      {/if}
    </dl>
  {/if}
</article>

<style>
  .card {
    display: grid;
    gap: var(--s3);
    padding: var(--s4);
    border-radius: var(--radius);
    background: var(--surface-raised);
    border: 1px solid var(--line);
  }
  .card.compact {
    padding: var(--s3);
    gap: var(--s2);
  }
  header {
    display: flex;
    justify-content: space-between;
    gap: var(--s2);
    align-items: flex-start;
  }
  h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }
  .meta {
    display: flex;
    gap: var(--s2);
    font-size: 12px;
    color: var(--muted);
    margin-top: 2px;
  }
  .unit {
    color: var(--primary);
  }
  .kind {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--faint);
  }
  .formula {
    padding: var(--s2) var(--s3);
    border-radius: var(--radius-sm);
    background: var(--surface);
    overflow-x: auto;
    font-size: 14px;
  }
  .plain {
    margin: 0;
    color: var(--text-soft);
  }
  dl {
    margin: 0;
    display: grid;
    gap: var(--s1);
  }
  dt {
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: var(--s2);
  }
  dd {
    margin: 0;
    color: var(--text-soft);
    font-size: 12px;
  }
  .caveat::before {
    content: "·";
    color: var(--warn);
    margin-right: 6px;
    font-weight: 700;
  }
</style>
