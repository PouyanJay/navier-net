<script lang="ts">
  import { api } from "../lib/api/client";
  import type { Agreement } from "../lib/api/types";
  import Plot from "../components/Plot.svelte";
  import Scene from "../components/Scene.svelte";
  import { download, downloadCsv } from "../lib/export";
  import { Icon, icons } from "../lib/icons";
  import { RunData, seriesColor } from "../lib/runData.svelte";
  import { fmt } from "../lib/units";
  import { IMAGE_MODES, studio } from "../stores/studio.svelte";

  const data = new RunData();
  let scene = $state<Scene | null>(null);
  let agreement = $state<Agreement | null>(null);
  let all = $state<Agreement[]>([]);
  let showImage = $state(true);
  let showMeasured = $state(true);
  let showModel = $state(true);
  let opacity = $state(1);

  $effect(() => {
    const run = studio.runId;
    const frame = studio.frame;
    if (!run) return;
    let cancelled = false;
    api.getAgreement(run, frame).then((a) => {
      if (!cancelled) agreement = a;
    });
    return () => {
      cancelled = true;
    };
  });

  $effect(() => {
    const run = studio.runId;
    const frames = studio.eventFrames.map((f) => f.frame);
    if (!run || frames.length === 0) return;
    let cancelled = false;
    Promise.all(frames.map((f) => api.getAgreement(run, f))).then((list) => {
      if (!cancelled) all = list;
    });
    return () => {
      cancelled = true;
    };
  });

  const imageUrl = $derived(
    showImage && studio.runId ? api.frameUrl(studio.runId, studio.frame, studio.imageMode) : null,
  );
  const walls = $derived(studio.run ? studio.run.domain_um.y_walls : null);
  const rows = $derived(
    all.map((a) => [
      a.frame,
      a.t_ms,
      a.role,
      a.iou,
      a.nose_model_um,
      a.nose_measured_um,
      a.nose_error_um,
      a.area_model_mm2,
      a.area_measured_mm2,
      a.only_model_mm2,
      a.only_measured_mm2,
    ]),
  );

  async function exportPng() {
    if (!scene) return;
    download(`${studio.runId}_geometry_f${studio.frame}.png`, await scene.toPng());
  }

  function exportTable() {
    downloadCsv(
      `${studio.runId}_geometry.csv`,
      [
        "frame",
        "t_ms",
        "role",
        "iou",
        "nose_model_um",
        "nose_measured_um",
        "nose_error_um",
        "area_model_mm2",
        "area_measured_mm2",
        "only_model_mm2",
        "only_measured_mm2",
      ],
      rows,
      { run: studio.runId },
    );
  }
</script>

<div class="view">
  <div class="stage">
    <div class="toolbar">
      <div class="title">
        <strong>Geometry against the camera</strong>
        <span class="muted mono"
          >frame {studio.frame}{studio.frameInfo?.t_ms != null
            ? ` · ${fmt(studio.frameInfo.t_ms, 3)} ms`
            : ""}</span
        >
        {#if agreement}
          <span
            class="chip mono"
            title="Intersection over union of the model's outline and the measured mask"
            >IoU {fmt(agreement.iou, 4)}</span
          >
          <span class="chip mono" title="Model nose minus measured nose"
            >nose {fmt(agreement.nose_error_um, 3)} µm</span
          >
          {#if agreement.role === "holdout"}<span class="chip holdout">held out</span>{/if}
        {/if}
      </div>
      <div class="actions">
        <label class="toggle"><input type="checkbox" bind:checked={showImage} /> frame</label>
        <div class="segmented" role="group" aria-label="Image">
          {#each IMAGE_MODES as m (m.id)}
            <button
              aria-pressed={studio.imageMode === m.id}
              onclick={() => (studio.imageMode = m.id)}>{m.label}</button
            >
          {/each}
        </div>
        <label class="toggle"><input type="checkbox" bind:checked={showModel} /> model</label>
        <label class="toggle slider"
          >opacity <input type="range" min="0.2" max="1" step="0.05" bind:value={opacity} /></label
        >
        <button class="btn" onclick={exportTable}
          ><Icon icon={icons.download} size={14} /> table CSV</button
        >
        <button class="btn" onclick={exportPng}><Icon icon={icons.download} size={14} /> PNG</button
        >
      </div>
    </div>
    <div class="canvas">
      <Scene
        bind:this={scene}
        grid={null}
        colormap="grey"
        range={[0, 1]}
        label="camera frame with outlines"
        extent={data.extent}
        boundary={agreement?.model_rings_um ?? data.contour?.rings_um ?? []}
        measured={agreement?.measured_rings_um ?? []}
        {walls}
        centreline={studio.run?.domain_um.centreline ?? null}
        {imageUrl}
        transform={data.transform}
        imageOpacity={opacity}
        showField={false}
        showBoundary={showModel}
        {showMeasured}
        showWalls={studio.layers.walls}
        showGrid={studio.layers.grid}
        probes={[]}
      />
      <div class="key">
        <span><i class="model"></i> model outline</span>
        <span><i class="measured"></i> measured mask</span>
      </div>
    </div>
    <div class="charts">
      <Plot
        title="Agreement per frame"
        xLabel="frame"
        yLabel="IoU"
        series={[
          {
            name: "IoU",
            color: seriesColor(0),
            x: all.map((a) => a.frame),
            y: all.map((a) => a.iou),
          },
        ]}
        height={180}
        zeroLine={false}
        filename="{studio.runId}_iou"
      />
      <Plot
        title="Nose position"
        xLabel="frame"
        yLabel="x"
        unit="µm"
        series={[
          {
            name: "model",
            color: seriesColor(0),
            x: all.map((a) => a.frame),
            y: all.map((a) => a.nose_model_um),
          },
          {
            name: "measured",
            color: seriesColor(2),
            dash: "5 4",
            x: all.map((a) => a.frame),
            y: all.map((a) => a.nose_measured_um),
          },
        ]}
        height={180}
        zeroLine={false}
        filename="{studio.runId}_nose"
      />
      <Plot
        title="Projected area"
        xLabel="frame"
        yLabel="area"
        unit="mm²"
        series={[
          {
            name: "model",
            color: seriesColor(0),
            x: all.map((a) => a.frame),
            y: all.map((a) => a.area_model_mm2),
          },
          {
            name: "measured",
            color: seriesColor(2),
            dash: "5 4",
            x: all.map((a) => a.frame),
            y: all.map((a) => a.area_measured_mm2),
          },
        ]}
        height={180}
        zeroLine={false}
        filename="{studio.runId}_area"
      />
    </div>
  </div>
  <aside class="side scroll">
    <h4>Per frame</h4>
    <table class="table mono">
      <thead>
        <tr><th>frame</th><th>IoU</th><th>nose µm</th><th>Δarea</th></tr>
      </thead>
      <tbody>
        {#each all as a (a.frame)}
          <tr class:active={a.frame === studio.frame} onclick={() => (studio.frame = a.frame)}>
            <td>{a.frame}{a.role === "holdout" ? " ·" : ""}</td>
            <td>{fmt(a.iou, 4)}</td>
            <td>{fmt(a.nose_error_um, 3)}</td>
            <td>{fmt(a.area_model_mm2 - a.area_measured_mm2, 3)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
    <p class="note">
      The model's interface is the measured outline when the run used one, so this view is the
      outline against the raw image and the mask: the rim band is where the geometry can still be
      wrong. A dot marks the held-out frame.
    </p>
  </aside>
</div>

<style>
  .view {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 300px;
    height: 100%;
    min-height: 0;
  }
  .stage {
    display: grid;
    grid-template-rows: auto minmax(220px, 1fr) auto;
    gap: var(--s3);
    padding: var(--s3);
    min-width: 0;
    overflow: auto;
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
  .actions {
    display: flex;
    gap: var(--s3);
    align-items: center;
  }
  .toggle {
    display: inline-flex;
    gap: 6px;
    align-items: center;
    font-size: 12px;
    color: var(--text-soft);
  }
  .toggle input {
    accent-color: var(--primary);
  }
  .canvas {
    position: relative;
    min-height: 0;
  }
  .key {
    position: absolute;
    top: var(--s3);
    right: var(--s3);
    display: flex;
    gap: var(--s3);
    padding: 6px 10px;
    border-radius: var(--radius-sm);
    background: color-mix(in srgb, var(--canvas) 70%, transparent);
    color: var(--canvas-ink);
    font-size: 11px;
  }
  .key i {
    display: inline-block;
    width: 16px;
    border-top: 2px solid;
    margin-right: 6px;
    vertical-align: middle;
  }
  .key .model {
    border-color: var(--boundary-light);
  }
  .key .measured {
    border-color: var(--measured);
    border-top-style: dashed;
  }
  .charts {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: var(--s3);
  }
  .side {
    border-left: 1px solid var(--line);
    background: var(--surface);
    padding: var(--s3);
    display: grid;
    gap: var(--s3);
    align-content: start;
  }
  h4 {
    margin: 0;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--faint);
  }
  .side table th,
  .side table td {
    padding: 4px 6px;
    font-size: 11px;
  }
  .side {
    overflow-x: hidden;
  }
  tr {
    cursor: pointer;
  }
  tr.active td {
    color: var(--primary);
  }
  .note {
    font-size: 12px;
    color: var(--muted);
  }
</style>
