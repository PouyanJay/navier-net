<script lang="ts">
  import { api, postCsv, probeUrl } from "../lib/api/client";
  import type { ProbeRequest, ProbeResponse, SectionKind } from "../lib/api/types";
  import Colorbar from "../components/Colorbar.svelte";
  import Plot, { type Series } from "../components/Plot.svelte";
  import QuantityCard from "../components/QuantityCard.svelte";
  import QuantityPicker from "../components/QuantityPicker.svelte";
  import Scene from "../components/Scene.svelte";
  import { download, downloadJson } from "../lib/export";
  import { FieldState } from "../lib/fieldState.svelte";
  import { Icon, icons } from "../lib/icons";
  import { RunData, seriesColor } from "../lib/runData.svelte";
  import { fmt } from "../lib/units";
  import { SECTION_LABELS, studio, type Probe } from "../stores/studio.svelte";

  const field = new FieldState();
  const data = new RunData();
  const pointQuantities = $derived(
    studio.quantitiesOfKind("volume", "derived", "residual", "term"),
  );

  let selected = $state<string[]>(["speed"]);
  let groupBy = $state<"frame" | "quantity">("frame");
  let results = $state<Map<string, ProbeResponse>>(new Map());
  let error = $state<string | null>(null);
  let busy = $state(false);
  let sectionKind = $state<SectionKind>("gap_top");
  let behind = $state(150);

  const walls = $derived(studio.run ? studio.run.domain_um.y_walls : null);
  const active = $derived(
    studio.probes.find((p) => p.id === studio.activeProbe) ?? studio.probes[0] ?? null,
  );
  const frames = $derived(studio.selectedFrames);
  const probeColors = $derived(
    Object.fromEntries(studio.probes.map((p, i) => [p.id, seriesColor(i)])),
  );

  function request(probe: Probe): ProbeRequest {
    return {
      kind: probe.kind,
      quantities: selected,
      frames,
      points_um: probe.points,
      n: probe.n,
      section: probe.section,
      behind_nose_um: probe.behindNoseUm,
      span_um: probe.spanUm,
    };
  }

  $effect(() => {
    const run = studio.runId;
    const probes = studio.probes.filter((p) => p.visible);
    const ids = selected;
    const list = frames;
    if (!run || probes.length === 0 || ids.length === 0 || list.length === 0) {
      results = new Map();
      return;
    }
    let cancelled = false;
    busy = true;
    Promise.all(probes.map((p) => api.probe(run, request(p)).then((r) => [p.id, r] as const)))
      .then((pairs) => {
        if (cancelled) return;
        results = new Map(pairs);
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

  function onDraw(kind: "line" | "point", points: [number, number][]) {
    const count = studio.probes.length + 1;
    if (kind === "point") {
      if (active?.kind === "points") {
        studio.updateProbe(active.id, { points: [...(active.points ?? []), points[0]] });
      } else {
        studio.addProbe({ name: `P${count}`, kind: "points", points, n: 1 });
      }
      return;
    }
    studio.addProbe({ name: `L${count}`, kind: "line", points, n: 200 });
    studio.tool = "pan";
  }

  function addSection() {
    const count = studio.probes.length + 1;
    studio.addProbe({
      name: `S${count} ${SECTION_LABELS[sectionKind]}`,
      kind: "section",
      section: sectionKind,
      behindNoseUm: behind,
      spanUm: [450, 40],
      n: 200,
    });
  }

  function toggleQuantity(id: string) {
    selected = selected.includes(id) ? selected.filter((q) => q !== id) : [...selected, id];
  }

  /** One plot per probe: series are frame x quantity. */
  function seriesFor(probe: Probe): Series[] {
    const r = results.get(probe.id);
    if (!r) return [];
    const out: Series[] = [];
    r.frames.forEach((f, fi) => {
      selected.forEach((q, qi) => {
        const index = groupBy === "frame" ? fi : qi;
        const dash =
          groupBy === "frame"
            ? qi === 0
              ? undefined
              : `${3 + 2 * qi} 3`
            : fi === 0
              ? undefined
              : `${3 + 2 * fi} 3`;
        out.push({
          name: `${f.frame !== null ? `frame ${f.frame}` : `${fmt(f.t_ms, 3)} ms`} · ${q}`,
          color: seriesColor(index),
          dash,
          x: probe.kind === "points" ? r.frames.map((g) => g.t_ms) : f.stations.s_um,
          y: probe.kind === "points" ? r.frames.map((g) => g.values[q][0] ?? null) : f.values[q],
        });
      });
    });
    if (probe.kind === "points") {
      // Points plot against time: one series per point and quantity.
      const first = r.frames[0];
      const n = first?.stations.s_um.length ?? 0;
      const byPoint: Series[] = [];
      for (let k = 0; k < n; k++) {
        selected.forEach((q, qi) => {
          byPoint.push({
            name: `${probe.name}${n > 1 ? ` ${k + 1}` : ""} · ${q}`,
            color: seriesColor(k),
            dash: qi === 0 ? undefined : `${3 + 2 * qi} 3`,
            x: r.frames.map((g) => g.t_ms),
            y: r.frames.map((g) => g.values[q][k] ?? null),
          });
        });
      }
      return byPoint;
    }
    return out;
  }

  function xLabelFor(probe: Probe): string {
    if (probe.kind === "points") return "t (ms)";
    const r = results.get(probe.id);
    return `${r?.frames[0]?.stations.label ?? "s"} (µm)`;
  }

  const markers = $derived.by(() => {
    const r = active ? results.get(active.id) : null;
    const f = r?.frames.find((g) => g.frame === studio.frame) ?? r?.frames[0];
    if (!f || active?.kind !== "section") return [];
    return f.stations.x_um.map((x, i) => ({
      x,
      y: f.stations.y_um[i],
      color: probeColors[active.id],
    }));
  });

  async function exportCsv(probe: Probe) {
    if (!studio.runId) return;
    const text = await postCsv(probeUrl(studio.runId), request(probe));
    download(`${studio.runId}_${probe.name.replace(/\s+/g, "_")}.csv`, text, "text/csv");
  }

  function exportJson(probe: Probe) {
    const r = results.get(probe.id);
    if (r)
      downloadJson(`${studio.runId}_${probe.name.replace(/\s+/g, "_")}.json`, {
        probe,
        response: r,
      });
  }
</script>

<div class="view">
  <div class="stage scroll">
    <div class="toolbar">
      <div class="tools segmented" role="group" aria-label="Tool">
        <button
          aria-pressed={studio.tool === "pan"}
          onclick={() => (studio.tool = "pan")}
          title="Pan and zoom"><Icon icon={icons.pan} size={14} /> pan</button
        >
        <button
          aria-pressed={studio.tool === "line"}
          onclick={() => (studio.tool = "line")}
          title="Draw a line: click vertices, double-click to finish"
          ><Icon icon={icons.line} size={14} /> line</button
        >
        <button
          aria-pressed={studio.tool === "point"}
          onclick={() => (studio.tool = "point")}
          title="Place a point (adds to the active point probe)"
          ><Icon icon={icons.point} size={14} /> point</button
        >
      </div>
      <div class="section">
        <select class="select" bind:value={sectionKind} aria-label="Preset section">
          {#each Object.entries(SECTION_LABELS) as [id, label] (id)}
            <option value={id}>{label}</option>
          {/each}
        </select>
        {#if sectionKind.startsWith("gap_")}
          <label class="inline mono"
            >behind nose <input
              class="input small"
              type="number"
              min="10"
              step="10"
              bind:value={behind}
            /> µm</label
          >
        {/if}
        <button class="btn" aria-label="Add the chosen section as a probe" onclick={addSection}
          ><Icon icon={icons.add} size={14} /> add section</button
        >
      </div>
      <div class="status">
        {#if busy}<span class="muted">evaluating…</span>{/if}
        {#if error}<span class="chip bad">{error}</span>{/if}
        <span class="muted">frames {frames.join(", ")}</span>
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
        probes={studio.probes}
        {probeColors}
        tool={studio.tool}
        {onDraw}
        {markers}
        height={340}
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
    <div class="plots">
      <div class="plots-head">
        <span class="muted">Group series by</span>
        <div class="segmented" role="group" aria-label="Group series">
          <button aria-pressed={groupBy === "frame"} onclick={() => (groupBy = "frame")}
            >frame</button
          >
          <button aria-pressed={groupBy === "quantity"} onclick={() => (groupBy = "quantity")}
            >quantity</button
          >
        </div>
      </div>
      {#each studio.probes.filter((p) => p.visible) as probe (probe.id)}
        <div class="probe-plot" class:active={probe.id === studio.activeProbe}>
          <Plot
            title="{probe.name}{probe.kind === 'section' && probe.section
              ? ` · ${SECTION_LABELS[probe.section]}`
              : ''}"
            xLabel={xLabelFor(probe)}
            yLabel={selected.length === 1
              ? (studio.byId.get(selected[0])?.symbol ?? selected[0])
              : "value"}
            unit={selected.length === 1 ? (studio.byId.get(selected[0])?.unit ?? "") : ""}
            series={seriesFor(probe)}
            height={240}
            filename="{studio.runId}_{probe.name}"
            meta={{ run: studio.runId, probe: probe.name, frames: frames.join(" ") }}
          />
          <div class="probe-actions">
            <button class="btn" onclick={() => exportCsv(probe)}
              ><Icon icon={icons.download} size={14} /> CSV (server)</button
            >
            <button class="btn" onclick={() => exportJson(probe)}>JSON</button>
          </div>
        </div>
      {/each}
      {#if studio.probes.length === 0}
        <p class="empty">
          No probes yet. Draw a line or place a point on the canvas, or add a preset section (the
          gap cross-section follows the nose from frame to frame).
        </p>
      {/if}
    </div>
  </div>
  <aside class="side">
    <section class="block">
      <h4>Probes</h4>
      <ul class="probe-list">
        {#each studio.probes as probe, i (probe.id)}
          <li class:active={probe.id === studio.activeProbe}>
            <div class="pick">
              <button
                class="swatch-btn"
                aria-label="Make {probe.name} the active probe"
                onclick={() => (studio.activeProbe = probe.id)}
              >
                <i style:background={seriesColor(i)}></i>
              </button>
              <input
                class="name"
                value={probe.name}
                aria-label="Probe name"
                onchange={(e) => studio.updateProbe(probe.id, { name: e.currentTarget.value })}
              />
            </div>
            <label class="vis"
              ><input
                type="checkbox"
                checked={probe.visible}
                onchange={(e) => studio.updateProbe(probe.id, { visible: e.currentTarget.checked })}
              /></label
            >
            {#if probe.kind === "section" && probe.section?.startsWith("gap_")}
              <input
                class="input small mono"
                type="number"
                step="10"
                value={probe.behindNoseUm}
                aria-label="Distance behind the nose"
                onchange={(e) =>
                  studio.updateProbe(probe.id, { behindNoseUm: Number(e.currentTarget.value) })}
              />
            {/if}
            <button
              class="icon-btn"
              aria-label="Remove probe"
              onclick={() => studio.removeProbe(probe.id)}
              ><Icon icon={icons.remove} size={14} /></button
            >
          </li>
        {/each}
      </ul>
    </section>
    <section class="block picker">
      <h4>Quantities</h4>
      <QuantityPicker quantities={pointQuantities} {selected} multiple onToggle={toggleQuantity} />
    </section>
    {#if selected.length === 1 && studio.byId.get(selected[0])}
      <QuantityCard quantity={studio.byId.get(selected[0])!} compact />
    {/if}
  </aside>
</div>

<style>
  .view {
    display: grid;
    grid-template-columns: minmax(0, 1fr) var(--inspector-width);
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
    gap: var(--s4);
    align-items: center;
    flex-wrap: wrap;
  }
  .tools button {
    display: inline-flex;
    gap: 6px;
    align-items: center;
  }
  .section {
    display: flex;
    gap: var(--s2);
    align-items: center;
  }
  .section .select {
    width: 220px;
  }
  .inline {
    display: inline-flex;
    gap: 6px;
    align-items: center;
    font-size: 12px;
    color: var(--muted);
  }
  .input.small {
    width: 72px;
    height: 28px;
    padding: 0 6px;
  }
  .status {
    margin-left: auto;
    display: flex;
    gap: var(--s2);
    align-items: center;
  }
  .canvas {
    position: relative;
  }
  .legend {
    position: absolute;
    top: var(--s3);
    right: var(--s3);
  }
  .plots {
    display: grid;
    gap: var(--s4);
  }
  .plots-head {
    display: flex;
    gap: var(--s2);
    align-items: center;
    font-size: 12px;
  }
  .probe-plot {
    display: grid;
    gap: var(--s2);
    padding: var(--s3);
    border-radius: var(--radius);
    border: 1px solid var(--line);
  }
  .probe-plot.active {
    border-color: color-mix(in srgb, var(--primary) 50%, var(--line));
  }
  .probe-actions {
    display: flex;
    gap: var(--s2);
  }
  .empty {
    color: var(--muted);
    padding: var(--s4);
    border: 1px dashed var(--line);
    border-radius: var(--radius);
  }
  .side {
    display: grid;
    grid-template-rows: auto minmax(160px, 1fr) auto;
    gap: var(--s3);
    padding: var(--s3);
    border-left: 1px solid var(--line);
    background: var(--surface);
    overflow: auto;
    min-height: 0;
  }
  .block {
    display: grid;
    gap: var(--s2);
    min-height: 0;
  }
  h4 {
    margin: 0;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--faint);
  }
  .probe-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 4px;
  }
  .probe-list li {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto auto;
    gap: 6px;
    align-items: center;
    padding: 4px 6px;
    border-radius: var(--radius-sm);
    border: 1px solid transparent;
  }
  .probe-list li.active {
    border-color: color-mix(in srgb, var(--primary) 50%, var(--line));
  }
  .pick {
    display: flex;
    gap: 8px;
    align-items: center;
    min-width: 0;
  }
  .swatch-btn {
    display: grid;
    place-items: center;
    width: 20px;
    height: 20px;
    flex: none;
  }
  .pick i {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex: none;
  }
  .name {
    background: transparent;
    border: 0;
    color: var(--text);
    min-width: 0;
    width: 100%;
    font-size: 12px;
  }
  .vis input {
    accent-color: var(--primary);
  }
</style>
