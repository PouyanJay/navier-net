import { useState } from "react";

import { Button, Callout, Chip, DL, type KV } from "../../components";
import { ConditionsIcon, HugeiconsIcon } from "../../components/icons";
import type {
  ConditionsUpdate,
  DatasetDetail,
  DatasetSummary,
  ProjectSummary,
} from "../../lib/api";
import { seriesName } from "../../lib/series";
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
  /** Set (or clear) the selected series' editable display name. */
  onSaveLabel: (label: string) => Promise<void>;
  /** Re-run preprocessing for the selected series (after a baked-field edit). */
  onPreprocess: () => void;
  /** Whether a preprocessing job is already running for the selected series. */
  preprocessing: boolean;
}

/**
 * The series' conditions as read-only summary rows (edited via the
 * Edit-conditions modal). The unit is carried in the label (e.g. "Frame interval
 * (ms)") so the values stay a clean column of numbers.
 *
 * Banded by what each condition FEEDS rather than listed flat: geometry drives
 * Bond and the Hele-Shaw number, thermal drives Jakob and Péclet, flow and
 * capture set the reference velocity every other group is scaled by. Grouping
 * the inputs the way the dimensionless outputs derive from them makes the panel
 * explain the model instead of just collecting values.
 */
function conditionItems(detail: DatasetDetail): KV[] {
  const c = detail.conditions;
  return [
    { group: "Fluid", label: "Working fluid", value: c.fluid },
    { group: "Fluid", label: "Saturation temp (°C)", value: c.T_sat_C },
    {
      group: "Geometry",
      label: "Channel width (µm)",
      value: c.channel_width_um,
    },
    {
      group: "Geometry",
      label: "Channel height (µm)",
      value: c.channel_height_um,
    },
    {
      group: "Thermal",
      label: "Wall heat flux (W·cm⁻²)",
      value: c.q_wall_W_cm2,
    },
    {
      group: "Flow & capture",
      label: "Flow rate (mL·hr⁻¹)",
      value: c.flow_rate_mL_hr,
    },
    {
      group: "Flow & capture",
      label: "Reference velocity (m·s⁻¹)",
      value: c.U_ref_m_s ?? "—",
    },
    {
      group: "Flow & capture",
      label: "Frame interval (ms)",
      value: c.dt_frame_ms,
    },
  ];
}

/**
 * What the rail can honestly say about the saved conditions.
 *
 * The mockup reads "saved · 2 min ago", but nothing in the API records when a
 * series was last written, so a relative time here would be invented. What is
 * real, and more useful at this spot, is whether the values on screen are the
 * ones the tensors were actually built from.
 */
function savedState(detail: DatasetDetail): string {
  if (!detail.conditions_set) return "not set";
  if (detail.processed && !detail.conditions_applied)
    return "saved · not built";
  return "saved";
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

/** The project's uploaded series; select one to edit it, or add another.
 *
 * Rendered as the datasets stage's secondary rail (see `StageAside`), so it
 * brings no frame of its own -- the shell owns the heading, the collapse control
 * and the scrolling. */
export function SeriesLibrary({
  project,
  series,
  trainedIds,
  selected,
  detail,
  onSelect,
  onProjectChanged,
  onSaveConditions,
  onSaveLabel,
  onPreprocess,
  preprocessing,
}: SeriesLibraryProps) {
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState(false);
  const trained = detail != null && trainedIds.has(detail.id);

  return (
    <>
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
            <span
              className="m"
              title={summary.label ? `series id: ${summary.id}` : undefined}
            >
              <b>{seriesName(summary)}</b>
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
            <h3
              className={detail.label ? undefined : "mono"}
              title={detail.label ? `series id: ${detail.id}` : undefined}
            >
              {seriesName(detail)}
            </h3>
            <span className="sub">inputs</span>
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
          onSaveLabel={onSaveLabel}
        />
      )}
      <p className="note lib-note">
        <b>Transfer learning:</b> once two or more series are preprocessed,
        select them together in the Solver to train one model jointly across
        their operating conditions.
      </p>

      {/* The shell's own action bar (see .aside-actions): sticky, frosted, and
          ruled across the rail, so it stays put however long the list grows and
          the fields sliding under it do not read through. The state beside the
          button says whether what is saved is what the tensors were built
          from. */}
      {detail && detail.id === selected && (
        <div className="aside-actions ds-ft">
          <span className="ds-ft-state mono">{savedState(detail)}</span>
          <Button variant="primary" onClick={() => setEditing(true)}>
            <HugeiconsIcon icon={ConditionsIcon} size={14} aria-hidden="true" />
            Edit conditions
          </Button>
        </div>
      )}
    </>
  );
}
