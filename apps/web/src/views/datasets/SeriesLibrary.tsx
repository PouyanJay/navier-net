import { useState } from "react";

import { Chip, DL, type KV, Panel } from "../../components";
import type {
  DatasetDetail,
  DatasetSummary,
  ProjectSummary,
} from "../../lib/api";
import { NewSeriesModal } from "./NewSeriesModal";

interface SeriesLibraryProps {
  project: ProjectSummary;
  series: DatasetSummary[];
  trainedIds: Set<string>;
  selected: string | null;
  /** The selected series' detail, for the read-only conditions summary. */
  detail: DatasetDetail | null;
  onSelect: (id: string) => void;
  /** Called with the updated project after a new series is uploaded+attached. */
  onProjectChanged: (project: ProjectSummary) => void;
}

/** The inputs set for a series in the upload modal, as read-only rows. */
function conditionItems(detail: DatasetDetail): KV[] {
  const c = detail.conditions;
  return [
    { label: "Working fluid", value: c.fluid },
    { label: "Frame interval", value: c.dt_frame_ms, hint: "ms" },
    { label: "Channel width", value: c.channel_width_um, hint: "µm" },
    { label: "Channel height", value: c.channel_height_um, hint: "µm" },
    { label: "Saturation temp", value: c.T_sat_C, hint: "°C" },
    { label: "Wall heat flux", value: c.q_wall_W_cm2, hint: "W·cm⁻²" },
    { label: "Flow rate", value: c.flow_rate_mL_hr, hint: "mL·hr⁻¹" },
    { label: "Reference velocity", value: c.U_ref_m_s ?? "—", hint: "m·s⁻¹" },
  ];
}

function seriesChip(summary: DatasetSummary, trained: boolean) {
  if (trained) return <Chip tone="green">trained</Chip>;
  if (!summary.conditions_set)
    return <Chip tone="amber">needs conditions</Chip>;
  if (summary.processed) return <Chip tone="green">tensors ready</Chip>;
  return <Chip>uploaded</Chip>;
}

function seriesMeta(summary: DatasetSummary): string {
  const parts = [`${summary.n_frames}`];
  if (summary.frame_px) {
    const [width, height] = summary.frame_px;
    parts.push(`× ${width === height ? `${width}²` : `${width}×${height}`}`);
  } else {
    parts.push("frames");
  }
  if (summary.dt_frame_ms != null) parts.push(`· Δt ${summary.dt_frame_ms} ms`);
  return parts.join(" ");
}

/** The project's uploaded series; select one to edit it, or add another. */
export function SeriesLibrary({
  project,
  series,
  trainedIds,
  selected,
  detail,
  onSelect,
  onProjectChanged,
}: SeriesLibraryProps) {
  const [adding, setAdding] = useState(false);

  return (
    <Panel title="Series library" subtitle="per-series conditions">
      {series.length === 0 && (
        <div className="dsempty">
          <b>No series yet</b>
          Upload the first high-speed sequence for this project to begin
          calibration and segmentation.
        </div>
      )}
      <div className="dsrows">
        {series.map((summary) => (
          <button
            key={summary.id}
            type="button"
            className={summary.id === selected ? "dsrow sel" : "dsrow"}
            aria-current={summary.id === selected || undefined}
            onClick={() => onSelect(summary.id)}
          >
            <span className="ic mono" aria-hidden="true">
              TIF
            </span>
            <span className="m">
              <b>{summary.id}</b>
              <span className="mono">{seriesMeta(summary)}</span>
            </span>
            <span className="st">
              {seriesChip(summary, trainedIds.has(summary.id))}
            </span>
          </button>
        ))}
      </div>
      <button type="button" className="addds" onClick={() => setAdding(true)}>
        + Upload new series · TIFF frames
      </button>
      {adding && (
        <NewSeriesModal
          project={project}
          onClose={() => setAdding(false)}
          onAttached={(updated, seriesId) => {
            onProjectChanged(updated);
            onSelect(seriesId);
          }}
        />
      )}
      {detail && detail.id === selected && (
        <section
          className="ds-conditions"
          aria-label={`${detail.id} conditions`}
        >
          <div className="ds-conditions-hd">
            <h3 className="mono">{detail.id}</h3>
            <span className="sub">inputs</span>
          </div>
          <DL items={conditionItems(detail)} />
        </section>
      )}
      <p className="note">
        <b>Transfer learning:</b> once two or more series are configured, Stage
        B can train jointly across heat-flux conditions.
      </p>
    </Panel>
  );
}
