<script lang="ts">
  import { api, postCsv, probeUrl } from "../lib/api/client";
  import { download, downloadJson } from "../lib/export";
  import { Icon, icons } from "../lib/icons";
  import { GRID_NX, GRID_NY, studio } from "../stores/studio.svelte";

  let quantities = $state<string[]>(["speed", "p_rel", "T"]);
  let includeGrids = $state(true);
  let includeEdge = $state(true);
  let includeProbes = $state(true);
  let includeBudgets = $state(true);
  let includeArtifacts = $state(true);
  let busy = $state(false);
  let log = $state<string[]>([]);

  const frames = $derived(
    studio.selectedFrames.length ? studio.selectedFrames : studio.eventFrames.map((f) => f.frame),
  );
  const edgeIds = [
    "kappa_par",
    "kappa_gap",
    "yl_jump",
    "shear",
    "normal_speed",
    "u_normal",
    "kinematic",
    "flux_g",
  ];

  function toggle(id: string) {
    quantities = quantities.includes(id) ? quantities.filter((q) => q !== id) : [...quantities, id];
  }

  async function exportAll() {
    if (!studio.runId) return;
    const run = studio.runId;
    busy = true;
    log = [];
    try {
      if (includeArtifacts) {
        for (const name of ["metrics", "audit", "trajectory", "front_velocity", "config"]) {
          try {
            downloadJson(`${run}_${name}.json`, await api.getArtifact(run, name));
            log = [...log, `${name}.json`];
          } catch {
            log = [...log, `${name}: not written by this run`];
          }
        }
      }
      if (includeGrids && quantities.length) {
        for (const f of frames) {
          const url = api.gridCsvUrl(run, quantities, f, GRID_NX, GRID_NY);
          const text = await (await fetch(url)).text();
          download(`${run}_grid_f${f}.csv`, text, "text/csv");
          log = [...log, `grid frame ${f} (${quantities.join(", ")})`];
        }
      }
      if (includeEdge && studio.run?.has_front) {
        for (const f of frames) {
          const text = await (await fetch(api.edgeCsvUrl(run, edgeIds, f))).text();
          download(`${run}_edge_f${f}.csv`, text, "text/csv");
          log = [...log, `edge frame ${f}`];
        }
      }
      if (includeProbes) {
        for (const p of studio.probes) {
          const text = await postCsv(probeUrl(run), {
            kind: p.kind,
            quantities: quantities.length ? quantities : ["speed"],
            frames,
            points_um: p.points,
            n: p.n,
            section: p.section,
            behind_nose_um: p.behindNoseUm,
            span_um: p.spanUm,
          });
          download(`${run}_probe_${p.name.replace(/\s+/g, "_")}.csv`, text, "text/csv");
          log = [...log, `probe ${p.name}`];
        }
      }
      if (includeBudgets) {
        for (const f of frames) {
          for (const kind of ["gap", "flux", "heat"] as const) {
            const response = await fetch(api.budgetCsvUrl(run, kind, f));
            if (response.ok)
              download(`${run}_${kind}_f${f}.csv`, await response.text(), "text/csv");
          }
          log = [...log, `budgets frame ${f}`];
        }
      }
    } finally {
      busy = false;
    }
  }

  function copyLink() {
    navigator.clipboard?.writeText(location.href);
    log = [...log, "link copied"];
  }
</script>

<div class="view scroll">
  <header>
    <strong>Export</strong>
    <p class="muted">
      Everything downloads as CSV (with provenance comment lines) or JSON. Frames: {frames.join(
        ", ",
      )}. Quantities come from the studio's catalogue; each CSV header repeats what every column is
      and how it was computed.
    </p>
  </header>
  <div class="grid">
    <section class="panel">
      <div class="panel-title">What to include</div>
      <div class="body">
        <label
          ><input type="checkbox" bind:checked={includeArtifacts} /> Pipeline artifacts (metrics, audit,
          trajectory, front velocity, config) as JSON</label
        >
        <label
          ><input type="checkbox" bind:checked={includeGrids} /> Field grids per frame ({GRID_NX} × {GRID_NY})
          as CSV</label
        >
        <label
          ><input type="checkbox" bind:checked={includeEdge} disabled={!studio.run?.has_front} /> Edge
          quantities per frame as CSV</label
        >
        <label
          ><input type="checkbox" bind:checked={includeProbes} /> Every probe ({studio.probes
            .length}) as CSV</label
        >
        <label
          ><input type="checkbox" bind:checked={includeBudgets} /> Budgets per frame (gap, flux, heat)
          as CSV</label
        >
        <div class="actions">
          <button class="btn primary" disabled={busy || !studio.runId} onclick={exportAll}>
            <Icon icon={icons.download} size={14} />
            {busy ? "exporting…" : "download everything"}
          </button>
          <button class="btn" onclick={copyLink}
            ><Icon icon={icons.link} size={14} /> copy link to this view</button
          >
        </div>
        {#if log.length}
          <ul class="log mono">
            {#each log as entry, i (i)}<li>{entry}</li>{/each}
          </ul>
        {/if}
      </div>
    </section>
    <section class="panel">
      <div class="panel-title">Grid and probe quantities</div>
      <div class="body chips">
        {#each studio.quantitiesOfKind("volume", "derived", "residual", "term") as q (q.id)}
          <button
            class="chip"
            class:ok={quantities.includes(q.id)}
            disabled={q.available === false}
            onclick={() => toggle(q.id)}
            title={q.plain}
          >
            <span class="mono">{q.symbol || q.id}</span>
            {q.label}
          </button>
        {/each}
      </div>
    </section>
  </div>
</div>

<style>
  .view {
    display: grid;
    gap: var(--s4);
    padding: var(--s4);
    align-content: start;
    height: 100%;
  }
  header p {
    margin: var(--s1) 0 0;
    max-width: 70ch;
  }
  .grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: var(--s3);
  }
  .body {
    padding: var(--s3);
    display: grid;
    gap: var(--s2);
    font-size: 12px;
    color: var(--text-soft);
  }
  .body label {
    display: flex;
    gap: var(--s2);
    align-items: center;
  }
  .body input {
    accent-color: var(--primary);
  }
  .actions {
    display: flex;
    gap: var(--s2);
    margin-top: var(--s2);
  }
  .log {
    margin: var(--s2) 0 0;
    padding-left: var(--s4);
    color: var(--muted);
    font-size: 11px;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .chips .chip {
    cursor: pointer;
  }
</style>
