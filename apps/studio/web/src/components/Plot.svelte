<script lang="ts">
  import { SvelteMap } from "svelte/reactivity";
  import * as d3 from "d3";

  import { downloadCsv, downloadJson } from "../lib/export";
  import { Icon, icons } from "../lib/icons";
  import { fmt } from "../lib/units";

  export interface Series {
    name: string;
    color: string;
    dash?: string;
    x: number[];
    y: (number | null)[];
  }

  let {
    series,
    xLabel,
    yLabel,
    unit = "",
    title = "",
    height = 240,
    zeroLine = true,
    filename = "plot",
    meta = {},
  }: {
    series: Series[];
    xLabel: string;
    yLabel: string;
    unit?: string;
    title?: string;
    height?: number;
    zeroLine?: boolean;
    filename?: string;
    meta?: Record<string, unknown>;
  } = $props();

  let width = $state(600);
  let hover = $state<{
    x: number;
    values: { name: string; color: string; y: number | null }[];
  } | null>(null);

  const margin = { top: 12, right: 16, bottom: 34, left: 56 };

  const xs = $derived(series.flatMap((s) => s.x));
  const ys = $derived(
    series.flatMap((s) => s.y.filter((v): v is number => v !== null && Number.isFinite(v))),
  );
  const xScale = $derived(
    d3
      .scaleLinear()
      .domain(xs.length ? (d3.extent(xs) as [number, number]) : [0, 1])
      .range([margin.left, width - margin.right]),
  );
  const yScale = $derived.by(() => {
    let [lo, hi] = ys.length ? (d3.extent(ys) as [number, number]) : [0, 1];
    if (lo === hi) {
      lo -= 1;
      hi += 1;
    }
    const pad = (hi - lo) * 0.06;
    return d3
      .scaleLinear()
      .domain([lo - pad, hi + pad])
      .range([height - margin.bottom, margin.top]);
  });
  const xTicks = $derived(xScale.ticks(6));
  const yTicks = $derived(yScale.ticks(5));

  const line = $derived(
    d3
      .line<[number, number | null]>()
      .defined((d) => d[1] !== null && Number.isFinite(d[1]))
      .x((d) => xScale(d[0]))
      .y((d) => yScale(d[1] as number)),
  );

  function paths(s: Series): string {
    return line(s.x.map((x, i) => [x, s.y[i]] as [number, number | null])) ?? "";
  }

  function onMove(event: PointerEvent) {
    const rect = (event.currentTarget as SVGElement).getBoundingClientRect();
    const x = xScale.invert(event.clientX - rect.left);
    hover = {
      x,
      values: series.map((s) => {
        const i = d3.bisectCenter(s.x, x);
        return { name: s.name, color: s.color, y: s.y[i] ?? null };
      }),
    };
  }

  function exportCsv() {
    const columns = [xLabel, ...series.map((s) => s.name)];
    const grid = new SvelteMap<number, (number | null)[]>();
    series.forEach((s, k) => {
      s.x.forEach((x, i) => {
        const row = grid.get(x) ?? new Array<number | null>(series.length).fill(null);
        row[k] = s.y[i];
        grid.set(x, row);
      });
    });
    const rows = [...grid.entries()].sort((a, b) => a[0] - b[0]).map(([x, row]) => [x, ...row]);
    downloadCsv(`${filename}.csv`, columns, rows, { ...meta, y: yLabel, unit });
  }

  function exportJson() {
    downloadJson(`${filename}.json`, { title, xLabel, yLabel, unit, meta, series });
  }
</script>

<figure class="plot" bind:clientWidth={width}>
  <figcaption>
    <span class="title">{title}</span>
    <span class="actions">
      <button class="icon-btn small" aria-label="Download CSV" title="CSV" onclick={exportCsv}>
        <Icon icon={icons.download} size={14} />
      </button>
      <button class="link" onclick={exportJson}>JSON</button>
    </span>
  </figcaption>
  <svg
    {width}
    {height}
    role="img"
    aria-label="{title}: {yLabel} against {xLabel}"
    onpointermove={onMove}
    onpointerleave={() => (hover = null)}
  >
    <g class="grid">
      {#each yTicks as t (t)}
        <line x1={margin.left} x2={width - margin.right} y1={yScale(t)} y2={yScale(t)} />
      {/each}
    </g>
    {#if zeroLine && yScale.domain()[0] < 0 && yScale.domain()[1] > 0}
      <line class="zero" x1={margin.left} x2={width - margin.right} y1={yScale(0)} y2={yScale(0)} />
    {/if}
    <g class="axis">
      {#each xTicks as t (t)}
        <text x={xScale(t)} y={height - margin.bottom + 16} text-anchor="middle">{fmt(t, 4)}</text>
      {/each}
      {#each yTicks as t (t)}
        <text x={margin.left - 8} y={yScale(t) + 3.5} text-anchor="end">{fmt(t, 3)}</text>
      {/each}
      <text
        class="label"
        x={(margin.left + width - margin.right) / 2}
        y={height - 4}
        text-anchor="middle"
      >
        {xLabel}
      </text>
      <text
        class="label"
        transform="translate(12 {(margin.top + height - margin.bottom) / 2}) rotate(-90)"
        text-anchor="middle"
      >
        {yLabel}{unit ? ` (${unit})` : ""}
      </text>
    </g>
    {#each series as s (s.name)}
      <path class="series" d={paths(s)} stroke={s.color} stroke-dasharray={s.dash ?? ""} />
    {/each}
    {#if hover}
      <line
        class="cursor"
        x1={xScale(hover.x)}
        x2={xScale(hover.x)}
        y1={margin.top}
        y2={height - margin.bottom}
      />
    {/if}
  </svg>
  <div class="legend">
    {#each series as s (s.name)}
      <span class="entry">
        <i style:background={s.color} style:border-style={s.dash ? "dashed" : "solid"}></i>
        <span>{s.name}</span>
        {#if hover}
          <span class="mono value"
            >{fmt(hover.values.find((v) => v.name === s.name)?.y ?? null, 4)}</span
          >
        {/if}
      </span>
    {/each}
    {#if hover}<span class="mono at">{xLabel} {fmt(hover.x, 4)}</span>{/if}
  </div>
</figure>

<style>
  .plot {
    margin: 0;
    display: grid;
    gap: var(--s1);
    min-width: 0;
  }
  figcaption {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-soft);
  }
  .actions {
    display: inline-flex;
    gap: var(--s2);
    align-items: center;
  }
  .icon-btn.small {
    width: 28px;
    height: 28px;
  }
  .link {
    font-size: 11px;
    color: var(--muted);
  }
  .link:hover {
    color: var(--text);
  }
  svg {
    display: block;
    width: 100%;
    background: var(--canvas);
    border-radius: var(--radius-sm);
    touch-action: none;
  }
  .grid line {
    stroke: var(--canvas-grid);
  }
  .zero {
    stroke: var(--canvas-wall);
    stroke-dasharray: 3 3;
  }
  .axis text {
    fill: var(--canvas-ink);
    font-family: var(--font-mono);
    font-size: 10px;
  }
  .axis .label {
    font-family: var(--font-ui);
    font-size: 11px;
  }
  .series {
    fill: none;
    stroke-width: 1.6;
  }
  .cursor {
    stroke: var(--canvas-ink);
    stroke-opacity: 0.5;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s3);
    font-size: 11px;
    color: var(--text-soft);
    min-height: 16px;
  }
  .entry {
    display: inline-flex;
    gap: 6px;
    align-items: center;
  }
  .entry i {
    width: 14px;
    height: 0;
    border-top: 2px solid;
    border-color: inherit;
  }
  .value {
    color: var(--text);
  }
  .at {
    color: var(--muted);
  }
</style>
