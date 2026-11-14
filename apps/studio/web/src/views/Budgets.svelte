<script lang="ts">
  import { api } from "../lib/api/client";
  import type { FluxBudget, GapBudget, HeatBudget } from "../lib/api/types";
  import Plot, { type Series } from "../components/Plot.svelte";
  import { Icon, icons } from "../lib/icons";
  import { seriesColor } from "../lib/runData.svelte";
  import { fmt } from "../lib/units";
  import { studio } from "../stores/studio.svelte";

  let side = $state<"upper" | "lower">("upper");
  let gap = $state<Map<number, GapBudget>>(new Map());
  let flux = $state<Map<number, FluxBudget>>(new Map());
  let heat = $state<HeatBudget[]>([]);
  let error = $state<string | null>(null);
  let busy = $state(false);

  const frames = $derived(studio.selectedFrames);
  const hasPressure = $derived(studio.run?.fields.includes("p") ?? false);

  $effect(() => {
    const run = studio.runId;
    const list = frames;
    const s = side;
    if (!run || list.length === 0) return;
    let cancelled = false;
    busy = true;
    const gapJobs = hasPressure
      ? list.map((f) => api.getGapBudget(run, f, s).then((r) => [f, r] as const))
      : [];
    Promise.all([
      Promise.all(gapJobs),
      Promise.all(list.map((f) => api.getFluxBudget(run, f).then((r) => [f, r] as const))),
    ])
      .then(([g, q]) => {
        if (cancelled) return;
        gap = new Map(g);
        flux = new Map(q);
        error = null;
      })
      .catch((e: Error) => {
        if (!cancelled) error = e.message;
      })
      .finally(() => {
        if (!cancelled) busy = false;
      });
    return () => {
      cancelled = true;
    };
  });

  $effect(() => {
    const run = studio.runId;
    const event = studio.selectedFrames;
    if (!run || event.length === 0) return;
    let cancelled = false;
    Promise.all(event.map((f) => api.getHeatBudget(run, f))).then((list) => {
      if (!cancelled) heat = list;
    });
    return () => {
      cancelled = true;
    };
  });

  const TERMS = [
    { key: "mom_dpdx_Pa_per_mm", name: "pressure gradient" },
    { key: "mom_friction_Pa_per_mm", name: "wall friction" },
    { key: "mom_viscous_Pa_per_mm", name: "in-plane viscous" },
    { key: "mom_unsteady_Pa_per_mm", name: "unsteady inertia" },
    { key: "mom_convective_Pa_per_mm", name: "convective inertia" },
  ];

  const gapSeries = $derived<Series[]>(
    frames.flatMap((f, fi) => {
      const g = gap.get(f);
      if (!g) return [];
      return TERMS.map((t, ti) => ({
        name: `frame ${f} · ${t.name}`,
        color: seriesColor(ti),
        dash: fi === 0 ? undefined : `${3 + 2 * fi} 3`,
        x: g.stations.behind_nose_um,
        y: g.stations[t.key],
      }));
    }),
  );
  const gapSpeed = $derived<Series[]>(
    frames.flatMap((f, fi) => {
      const g = gap.get(f);
      return g
        ? [
            {
              name: `frame ${f}`,
              color: seriesColor(fi),
              x: g.stations.behind_nose_um,
              y: g.stations.u_mm_s,
            },
          ]
        : [];
    }),
  );
  const fluxSeries = $derived<Series[]>(
    frames.flatMap((f, fi) => {
      const q = flux.get(f);
      if (!q) return [];
      return [
        { name: `frame ${f} · liquid flux`, color: seriesColor(fi), x: q.x_um, y: q.q_liquid_mL_h },
        {
          name: `frame ${f} · required`,
          color: seriesColor(fi),
          dash: "5 4",
          x: q.x_um,
          y: q.q_required_mL_h,
        },
      ];
    }),
  );
  const heatSeries = $derived<Series[]>(
    heat.length
      ? [
          {
            name: "source makes",
            color: seriesColor(0),
            x: heat.map((h) => Number(h.frame)),
            y: heat.map((h) => Number(h.latent_from_source_mW)),
          },
          {
            name: "growth needs",
            color: seriesColor(2),
            dash: "5 4",
            x: heat.map((h) => Number(h.frame)),
            y: heat.map((h) => Number(h.latent_for_growth_mW)),
          },
          {
            name: "edge flux part",
            color: seriesColor(1),
            x: heat.map((h) => Number(h.frame)),
            y: heat.map((h) => Number(h.latent_from_flux_mW ?? NaN)),
          },
          {
            name: "footprint part",
            color: seriesColor(3),
            x: heat.map((h) => Number(h.frame)),
            y: heat.map((h) => Number(h.latent_from_footprint_mW ?? NaN)),
          },
          {
            name: "wall into liquid",
            color: seriesColor(4),
            dash: "2 3",
            x: heat.map((h) => Number(h.frame)),
            y: heat.map((h) => Number(h.wall_into_liquid_mW)),
          },
        ]
      : [],
  );

  const currentGap = $derived(gap.get(studio.frame) ?? null);
  const binColumns = $derived(
    currentGap?.bins[0] ? Object.keys(currentGap.bins[0]).filter((k) => k.endsWith("_Pa")) : [],
  );
</script>

<div class="view scroll">
  <div class="toolbar">
    <strong>Integral budgets</strong>
    <span class="muted">frames {frames.join(", ")} (shift-click to compare)</span>
    <div class="segmented" role="group" aria-label="Gap side">
      <button aria-pressed={side === "upper"} onclick={() => (side = "upper")}>upper gap</button>
      <button aria-pressed={side === "lower"} onclick={() => (side = "lower")}>lower gap</button>
    </div>
    {#if busy}<span class="muted">evaluating…</span>{/if}
    {#if error}<span class="chip bad">{error}</span>{/if}
    {#if studio.runId}
      <span class="spacer"></span>
      <a class="btn" href={api.budgetCsvUrl(studio.runId, "gap", studio.frame)} download
        ><Icon icon={icons.download} size={14} /> gap CSV</a
      >
      <a class="btn" href={api.budgetCsvUrl(studio.runId, "flux", studio.frame)} download
        ><Icon icon={icons.download} size={14} /> flux CSV</a
      >
      <a class="btn" href={api.budgetCsvUrl(studio.runId, "heat", studio.frame)} download
        ><Icon icon={icons.download} size={14} /> heat CSV</a
      >
    {/if}
  </div>

  <section class="panel">
    <div class="panel-title">
      <span>x-momentum along the middle of the {side} gap beside the head</span>
      <span class="muted"
        >each term as the force per volume it exerts (Pa/mm); the law says they sum to zero</span
      >
    </div>
    <div class="body two">
      <Plot
        title="Terms of the trained Brinkman law along the gap"
        xLabel="distance behind the nose (µm)"
        yLabel="force per volume"
        unit="Pa/mm"
        series={gapSeries}
        height={260}
        filename="{studio.runId}_gap_terms"
      />
      <Plot
        title="Liquid speed along the gap (negative is backflow)"
        xLabel="distance behind the nose (µm)"
        yLabel="u"
        unit="mm/s"
        series={gapSpeed}
        height={260}
        filename="{studio.runId}_gap_speed"
      />
    </div>
    {#if currentGap}
      <table class="table mono">
        <caption
          >Frame {studio.frame}: the pressure change each term alone would need across the bin (Pa);
          leftover is the integrated residual.</caption
        >
        <thead>
          <tr
            ><th>bin (µm behind nose)</th>{#each binColumns as c (c)}<th
                >{c.replace(/_Pa$/, "").replace(/^mom_/, "")}</th
              >{/each}</tr
          >
        </thead>
        <tbody>
          {#each currentGap.bins as b (b.from_um)}
            <tr
              ><td>{b.from_um} to {b.to_um}</td>{#each binColumns as c (c)}<td>{fmt(b[c], 3)}</td
                >{/each}</tr
            >
          {/each}
        </tbody>
      </table>
    {:else if !hasPressure}
      <p class="muted pad">This run has no pressure field, so there is no momentum budget.</p>
    {/if}
  </section>

  <section class="panel">
    <div class="panel-title">
      <span>Liquid flux through every section against mass conservation</span>
      <span class="muted"
        >required = inlet flux + rate of vapour growth upstream of the section</span
      >
    </div>
    <div class="body">
      <Plot
        title="Liquid volume flux along the channel"
        xLabel="x (µm)"
        yLabel="Q"
        unit="mL/h"
        series={fluxSeries}
        height={260}
        filename="{studio.runId}_flux"
      />
    </div>
  </section>

  <section class="panel">
    <div class="panel-title">
      <span>Heat budget per frame</span>
      <span class="muted"
        >latent power the source implies against what the measured growth needs, in mW</span
      >
    </div>
    <div class="body">
      <Plot
        title="Latent power"
        xLabel="frame"
        yLabel="power"
        unit="mW"
        series={heatSeries}
        height={240}
        zeroLine={false}
        filename="{studio.runId}_heat_budget"
      />
    </div>
  </section>
</div>

<style>
  .view {
    display: grid;
    gap: var(--s3);
    padding: var(--s3);
    align-content: start;
    height: 100%;
  }
  .toolbar {
    display: flex;
    gap: var(--s3);
    align-items: center;
    flex-wrap: wrap;
  }
  .spacer {
    flex: 1;
  }
  .body {
    padding: var(--s3);
    display: grid;
    gap: var(--s3);
  }
  .body.two {
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  }
  caption {
    text-align: left;
    padding: var(--s2) var(--s3);
    color: var(--muted);
    font-family: var(--font-ui);
    font-size: 12px;
  }
  .pad {
    padding: var(--s3);
  }
</style>
