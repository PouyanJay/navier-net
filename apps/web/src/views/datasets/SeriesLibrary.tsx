import { useState } from "react";

import { Button, Callout, Chip, DL, type KV, Panel } from "../../components";
import type {
  ConditionsUpdate,
  DatasetDetail,
  DatasetSummary,
  ProjectSummary,
} from "../../lib/api";
import { EditConditionsModal } from "./EditConditionsModal";
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
  /** Persist an edit to the selected series' operating conditions. */
  onSaveConditions: (updates: ConditionsUpdate) => Promise<void>;
  /** Re-run preprocessing for the selected series (after a baked-field edit). */
  onPreprocess: () => void;
  /** Whether a preprocessing job is already running for the selected series. */
  preprocessing: boolean;
}

/** The series' conditions as read-only summary rows (edited via the
 * Edit-conditions modal). The unit is carried in the label (e.g. "Frame interval
 * (ms)") so the values stay a clean column of numbers. */
function conditionItems(detail: DatasetDetail): KV[] {
  const c = detail.conditions;
  return [
    { label: "Working fluid", value: c.fluid },
    { label: "Frame interval (ms)", value: c.dt_frame_ms },
    { label: "Channel width (µm)", value: c.channel_width_um },
    { label: "Channel height (µm)", value: c.channel_height_um },
    { label: "Saturation temp (°C)", value: c.T_sat_C },
    { label: "Wall heat flux (W·cm⁻²)", value: c.q_wall_W_cm2 },
    { label: "Flow rate (mL·hr⁻¹)", value: c.flow_rate_mL_hr },
    { label: "Reference velocity (m·s⁻¹)", value: c.U_ref_m_s ?? "—" },
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
  onSaveConditions,
  onPreprocess,
  preprocessing,
}: SeriesLibraryProps) {
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState(false);
  const trained = detail != null && trainedIds.has(detail.id);

  return (
    <Panel
      title="Series library"
      subtitle="per-series conditions"
      className="lib-card"
    >
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
            <Button
              className="ds-conditions-edit"
              onClick={() => setEditing(true)}
            >
              Edit conditions
            </Button>
          </div>
          <DL items={conditionItems(detail)} />
          {detail.processed && !detail.conditions_applied && (
            <div className="ds-reprocess">
              <Callout tone="caution" title="Re-preprocess required">
                A tensor-baked condition (frame interval, channel width, or
                reference velocity) changed since these tensors were built.
                {trained
                  ? " Re-preprocessing rebuilds them and marks the trained run stale."
                  : " Re-preprocess to rebuild them from the new values."}
              </Callout>
              <Button
                variant="primary"
                onClick={onPreprocess}
                disabled={preprocessing}
              >
                {preprocessing ? "Re-preprocessing…" : "Re-preprocess"}
              </Button>
            </div>
          )}
        </section>
      )}
      {editing && detail && detail.id === selected && (
        <EditConditionsModal
          detail={detail}
          onClose={() => setEditing(false)}
          onSave={onSaveConditions}
        />
      )}
      <p className="note">
        <b>Transfer learning:</b> once two or more series are preprocessed,
        select them together in the Solver to train one model jointly across
        their operating conditions.
      </p>
    </Panel>
  );
}
