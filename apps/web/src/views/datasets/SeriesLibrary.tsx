import { useState, type ReactNode } from "react";

import { Button, Callout, Chip } from "../../components";
import {
  ConditionsIcon,
  FluidIcon,
  FlowIcon,
  GeometryIcon,
  HugeiconsIcon,
  MenuOpenIcon,
  SeriesIcon,
  ThermalIcon,
} from "../../components/icons";
import type {
  ConditionsUpdate,
  DatasetDetail,
  DatasetSummary,
  OperatingConditions,
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
  /** The selected series' detail, for the conditions the cards summarise. */
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

/** One value with its quiet unit suffix, for a card's rows. */
interface Row {
  label: string;
  value: ReactNode;
  unit?: string;
  /** Computed from the rows above it, so it is read, never edited. */
  derived?: boolean;
}

interface Domain {
  id: string;
  title: string;
  icon: typeof FluidIcon;
  /** One line honest enough to stand in for the rows when the card is shut. */
  summary: (c: OperatingConditions) => string;
  rows: (c: OperatingConditions) => Row[];
}

/** 4·area over wetted perimeter of the rectangular channel, in µm. */
function hydraulicDiameter(c: OperatingConditions): number {
  return Math.round(
    (2 * c.channel_width_um * c.channel_height_um) /
      (c.channel_width_um + c.channel_height_um),
  );
}

/**
 * The conditions banded by what each one FEEDS rather than listed flat:
 * geometry drives Bond and the Hele-Shaw number, thermal drives Jakob and
 * Péclet, flow and capture set the reference velocity every other group is
 * scaled by. Each band is one card, its summary carrying the values so the
 * closed rail still reads.
 */
const DOMAINS: Domain[] = [
  {
    id: "fluid",
    title: "Fluid",
    icon: FluidIcon,
    summary: (c) => `${c.fluid} · T_sat ${c.T_sat_C} °C`,
    rows: (c) => [
      { label: "Working fluid", value: c.fluid },
      { label: "Saturation temp", value: c.T_sat_C, unit: "°C" },
    ],
  },
  {
    id: "geometry",
    title: "Geometry",
    icon: GeometryIcon,
    summary: (c) =>
      `${c.channel_width_um} × ${c.channel_height_um} µm · D_h ${hydraulicDiameter(c)} µm`,
    rows: (c) => [
      { label: "Channel width", value: c.channel_width_um, unit: "µm" },
      { label: "Channel height", value: c.channel_height_um, unit: "µm" },
      {
        label: "Hydraulic diameter",
        value: hydraulicDiameter(c),
        unit: "µm",
        derived: true,
      },
    ],
  },
  {
    id: "thermal",
    title: "Thermal",
    icon: ThermalIcon,
    summary: (c) => `q_wall ${c.q_wall_W_cm2} W·cm⁻²`,
    rows: (c) => [
      { label: "Wall heat flux", value: c.q_wall_W_cm2, unit: "W·cm⁻²" },
    ],
  },
  {
    id: "flow",
    title: "Flow & capture",
    icon: FlowIcon,
    summary: (c) => `${c.flow_rate_mL_hr} mL·hr⁻¹ · Δt ${c.dt_frame_ms} ms`,
    rows: (c) => [
      { label: "Flow rate", value: c.flow_rate_mL_hr, unit: "mL·hr⁻¹" },
      {
        label: "Reference velocity",
        value: c.U_ref_m_s ?? "—",
        unit: c.U_ref_m_s != null ? "m·s⁻¹" : undefined,
      },
      { label: "Frame interval", value: c.dt_frame_ms, unit: "ms" },
    ],
  },
];

/* Fluid and Geometry open by default: they are the identity of the rig, the
   first things checked against a lab notebook, and together they still leave
   the other two cards' summaries on screen. */
const OPEN_BY_DEFAULT = ["fluid", "geometry"];

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

/** One condition domain: closed it is a summary, open it is the rows. */
function DomainCard({
  domain,
  conditions,
  unset,
  open,
  onToggle,
}: {
  domain: Domain;
  conditions: OperatingConditions;
  unset: boolean;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <section
      className={open ? "dcard open" : "dcard"}
      aria-label={`${domain.title} conditions`}
    >
      <button
        type="button"
        className="dcard-hd"
        aria-expanded={open}
        onClick={onToggle}
      >
        <span className="dcard-ic" aria-hidden="true">
          <HugeiconsIcon icon={domain.icon} size={15} />
        </span>
        <span className="dcard-m">
          <b>{domain.title}</b>
          {/* Closed, the summary stands in for the rows, so it goes amber and
              says so when there is nothing behind it to stand in for. */}
          <span className={unset ? "mono miss" : "mono"}>
            {unset ? "not set" : domain.summary(conditions)}
          </span>
        </span>
        <span className="dcard-chev" aria-hidden="true">
          <HugeiconsIcon icon={MenuOpenIcon} size={13} />
        </span>
      </button>
      {open && !unset && (
        <dl className="dcard-rows">
          {domain.rows(conditions).map((row) => (
            <div
              key={row.label}
              className={row.derived ? "drow derived" : "drow"}
            >
              <dt>{row.label}</dt>
              <dd className="mono">
                {row.value}
                {row.unit && <em>{row.unit}</em>}
              </dd>
            </div>
          ))}
        </dl>
      )}
      {open && unset && (
        <p className="dcard-none">
          Set in the conditions form; nothing is recorded for this series yet.
        </p>
      )}
    </section>
  );
}

/** The project's uploaded series; select one to read and edit its conditions,
 * or add another.
 *
 * Rendered as the datasets stage's secondary rail (see `StageAside`), so it
 * brings no frame of its own — the shell owns the heading, the collapse
 * control and the scrolling. Below the list, the selected series' conditions
 * read as one card per domain; the cards expand independently, so two bands
 * can be compared side by side. */
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
  const [openCards, setOpenCards] = useState<ReadonlySet<string>>(
    () => new Set(OPEN_BY_DEFAULT),
  );
  const trained = detail != null && trainedIds.has(detail.id);
  const unset = detail != null && !detail.conditions_set;

  const toggleCard = (id: string) =>
    setOpenCards((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

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
            {/* A film strip, bare: the tile behind the old "TIF" text badge
                boxed a glyph that already reads on its own. */}
            <span className="ic" aria-hidden="true">
              <HugeiconsIcon icon={SeriesIcon} size={17} />
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
      {detail && detail.id === selected && (
        <div className="dcards" aria-label={`${detail.id} conditions`}>
          {DOMAINS.map((domain) => (
            <DomainCard
              key={domain.id}
              domain={domain}
              conditions={detail.conditions}
              unset={unset}
              open={openCards.has(domain.id)}
              onToggle={() => toggleCard(domain.id)}
            />
          ))}

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
        </div>
      )}

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
      {editing && detail && detail.id === selected && (
        <EditConditionsModal
          detail={detail}
          onClose={() => setEditing(false)}
          onSave={onSaveConditions}
          onSaveLabel={onSaveLabel}
        />
      )}

      {/* Upload sinks to the foot of the rail, just above the action bar:
          growing the library is the last thing in reading order, after the
          series that exist and the conditions of the one selected. */}
      <button
        type="button"
        className="addds addds-foot"
        onClick={() => setAdding(true)}
      >
        + Upload new series · TIFF frames
      </button>

      {/* The shell's own action bar (see .aside-actions): the way out is in
          the same place whatever series is selected and whatever cards are
          open. Missing conditions rename the edit to what it actually is. */}
      {detail && detail.id === selected && (
        <div className="aside-actions ds-ft">
          <span className="ds-ft-state mono">{savedState(detail)}</span>
          <Button variant="primary" onClick={() => setEditing(true)}>
            <HugeiconsIcon icon={ConditionsIcon} size={14} aria-hidden="true" />
            {unset ? "Set conditions" : "Edit conditions"}
          </Button>
        </div>
      )}
    </>
  );
}
