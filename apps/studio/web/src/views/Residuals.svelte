<script lang="ts">
  import Colorbar from "../components/Colorbar.svelte";
  import Inspector from "../components/Inspector.svelte";
  import Plot, { type Series } from "../components/Plot.svelte";
  import Scene from "../components/Scene.svelte";
  import { FieldState } from "../lib/fieldState.svelte";
  import { RunData, seriesColor } from "../lib/runData.svelte";
  import { fmt } from "../lib/units";
  import { studio } from "../stores/studio.svelte";

  const field = new FieldState();
  const data = new RunData();

  const quantities = $derived(studio.quantitiesOfKind("residual", "term"));
  const walls = $derived(studio.run ? studio.run.domain_um.y_walls : null);

  // Keep the canvas on a residual when the user arrives from the Fields view.
  $effect(() => {
    if (studio.current && !["residual", "term"].includes(studio.current.kind)) {
      const first = quantities.find((q) => q.available);
      if (first) studio.quantity = first.id;
    }
  });

  const audit = $derived(data.audit);
  const equations = $derived(audit ? Object.keys(audit.frames[0]?.residual_rms ?? {}) : []);
  const rmsSeries = $derived<Series[]>(
    equations.map((eq, i) => ({
      name: eq,
      color: seriesColor(i),
      x: audit!.frames.map((f) => f.frame),
      y: audit!.frames.map((f) => f.residual_rms[eq] ?? null),
    })),
  );
  const fluxSeries = $derived<Series[]>(
    audit
      ? [
          {
            name: "worst section defect",
            color: seriesColor(0),
            x: audit.frames.map((f) => f.frame),
            y: audit.frames.map((f) => f.flux.max_abs),
          },
        ]
      : [],
  );
  const heatSeries = $derived<Series[]>(
    audit
      ? [
          {
            name: "source",
            color: seriesColor(0),
            x: audit.frames.map((f) => f.frame),
            y: audit.frames.map((f) => Number(f.heat.latent_from_source_mW)),
          },
          {
            name: "growth needs",
            color: seriesColor(2),
            dash: "5 4",
            x: audit.frames.map((f) => f.frame),
            y: audit.frames.map((f) => Number(f.heat.latent_for_growth_mW)),
          },
          {
            name: "edge flux",
            color: seriesColor(1),
            x: audit.frames.map((f) => f.frame),
            y: audit.frames.map((f) => Number(f.heat.latent_from_flux_mW ?? NaN)),
          },
          {
            name: "footprint",
            color: seriesColor(3),
            x: audit.frames.map((f) => f.frame),
            y: audit.frames.map((f) => Number(f.heat.latent_from_footprint_mW ?? NaN)),
          },
        ]
      : [],
  );
  const ylSeries = $derived<Series[]>(
    audit && audit.frames[0]?.front
      ? [
          {
            name: "rms",
            color: seriesColor(0),
            x: audit.frames.map((f) => f.frame),
            y: audit.frames.map((f) => Number(f.front?.laplace_rms_Pa ?? NaN)),
          },
          {
            name: "scatter",
            color: seriesColor(1),
            dash: "5 4",
            x: audit.frames.map((f) => f.frame),
            y: audit.frames.map((f) => Number(f.front?.laplace_scatter_Pa ?? NaN)),
          },
        ]
      : [],
  );
  const verdicts = $derived(audit ? Object.entries(audit.verdicts) : []);
  const relative = $derived.by(() => {
    const frame = audit?.frames.find((f) => f.frame === studio.frame);
    const rel = frame?.relative_error;
    if (!rel) return [];
    return Object.entries(rel)
      .filter(([, v]) => typeof v === "object")
      .map(([eq, regions]) => ({ eq, regions: regions as Record<string, number> }));
  });
</script>

<div class="view">
  <div class="stage scroll">
    <div class="toolbar">
      <label class="toggle"
        ><input type="checkbox" bind:checked={studio.layers.measured} /> measured mask</label
      >
      <div class="title">
        <strong>{studio.current?.label ?? studio.quantity}</strong>
        <span class="muted mono">frame {studio.frame}</span>
        {#if field.busy}<span class="muted">evaluating…</span>{/if}
        {#if field.error}<span class="chip bad">{field.error}</span>{/if}
      </div>
      <div class="verdicts">
        {#each verdicts as [name, v] (name)}
          <span
            class="chip {v.pass ? 'ok' : 'bad'}"
            title="worst {fmt(v.worst, 3)} against {fmt(v.max ?? v.min ?? null, 3)}"
            >{name} {fmt(v.worst, 3)}</span
          >
        {/each}
      </div>
    </div>
    <div class="canvas">
      <Scene
        grid={field.grid}
        colormap={field.colormap}
        range={field.range}
        unit={studio.current?.unit ?? ""}
        label={studio.current?.label ?? ""}
        extent={data.extent}
        boundary={field.grid?.header.boundary_um ?? data.contour?.rings_um ?? []}
        measured={data.measured}
        {walls}
        centreline={studio.run?.domain_um.centreline ?? null}
        showField={studio.layers.field}
        showBoundary={studio.layers.boundary}
        showMeasured={studio.layers.measured}
        showWalls={studio.layers.walls}
        showGrid={studio.layers.grid}
        probes={studio.layers.probes ? studio.probes : []}
        height={360}
      />
      <div class="legend">
        <Colorbar
          colormap={field.colormap}
          range={field.range}
          unit={studio.current?.unit ?? ""}
          label={studio.current?.label ?? ""}
        />
      </div>
    </div>
    {#if audit}
      <div class="charts">
        <Plot
          title="Residual rms per equation (audit grid, model units)"
          xLabel="frame"
          yLabel="rms"
          series={rmsSeries}
          height={200}
          zeroLine={false}
          filename="{studio.runId}_residual_rms"
        />
        <Plot
          title="Section flux defect (fraction of the inlet flux)"
          xLabel="frame"
          yLabel="|defect|"
          series={fluxSeries}
          height={200}
          zeroLine={false}
          filename="{studio.runId}_flux_defect"
        />
        <Plot
          title="Latent power: what the source makes against what growth needs"
          xLabel="frame"
          yLabel="power"
          unit="mW"
          series={heatSeries}
          height={200}
          zeroLine={false}
          filename="{studio.runId}_heat"
        />
        {#if ylSeries.length}
          <Plot
            title="Young-Laplace mismatch around the edge"
            xLabel="frame"
            yLabel="mismatch"
            unit="Pa"
            series={ylSeries}
            height={200}
            zeroLine={false}
            filename="{studio.runId}_yl"
          />
        {/if}
      </div>
      {#if relative.length}
        <div class="panel">
          <div class="panel-title">
            Relative error by region at frame {studio.frame} (rms residual over rms of the terms)
          </div>
          <table class="table mono">
            <thead>
              <tr
                ><th>equation</th>{#each Object.keys(relative[0].regions) as r (r)}<th>{r}</th
                  >{/each}</tr
              >
            </thead>
            <tbody>
              {#each relative as row (row.eq)}
                <tr
                  ><td>{row.eq}</td>{#each Object.values(row.regions) as v, i (i)}<td
                      >{fmt(v, 3)}</td
                    >{/each}</tr
                >
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    {:else}
      <p class="muted">
        This run wrote no audit.json; the per-frame tables need the evaluate stage's audit.
      </p>
    {/if}
  </div>
  <Inspector {quantities} range={field.range} colormap={field.colormap} />
</div>

<style>
  .view {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    height: 100%;
    min-height: 0;
  }
  .stage {
    display: grid;
    gap: var(--s3);
    padding: var(--s3);
    min-width: 0;
    align-content: start;
  }
  .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--s3);
    flex-wrap: wrap;
  }
  .title {
    display: flex;
    align-items: center;
    gap: var(--s3);
  }
  .verdicts {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .canvas {
    position: relative;
  }
  .legend {
    position: absolute;
    top: var(--s3);
    right: var(--s3);
  }
  .charts {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: var(--s3);
  }
</style>
