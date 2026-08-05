import { useState, type ReactNode } from "react";

import { Button, Callout, ConfirmDeleteDialog } from "../../components";
import {
  DeleteIcon,
  EditIcon,
  FluidIcon,
  FlowIcon,
  GeometryIcon,
  HugeiconsIcon,
  SeriesIcon,
  ThermalIcon,
  UploadIcon,
} from "../../components/icons";
import { api } from "../../lib/api";
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
    rows: (c) => [
      { label: "Working fluid", value: c.fluid },
      { label: "Saturation temp", value: c.T_sat_C, unit: "°C" },
    ],
  },
  {
    id: "geometry",
    title: "Geometry",
    icon: GeometryIcon,
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
    rows: (c) => [
      { label: "Wall heat flux", value: c.q_wall_W_cm2, unit: "W·cm⁻²" },
    ],
  },
  {
    id: "flow",
    title: "Flow & capture",
    icon: FlowIcon,
    rows: (c) => [
      { label: "Flow rate", value: c.flow_rate_mL_hr, unit: "mL·hr⁻¹" },
      { label: "Reference velocity", value: c.U_ref_m_s ?? "—", unit: "m·s⁻¹" },
      { label: "Frame interval", value: c.dt_frame_ms, unit: "ms" },
    ],
  },
];

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

/** One condition domain: a bare icon-and-label separator over its rows. No
 * box and no fold — inside the series card, cards-within-a-card read as
 * chrome, and a summary standing in for two visible rows saved nothing. */
function DomainBand({
  domain,
  conditions,
}: {
  domain: Domain;
  conditions: OperatingConditions;
}) {
  return (
    <section className="dband" aria-label={`${domain.title} conditions`}>
      <h3 className="dband-hd">
        <HugeiconsIcon icon={domain.icon} size={13} aria-hidden="true" />
        {domain.title}
      </h3>
      <dl className="dband-rows">
        {domain.rows(conditions).map((row) => (
          <div
            key={row.label}
            className={row.derived ? "drow derived" : "drow"}
          >
            {/* The unit rides the variable's name in brackets, so the value
                column stays a clean run of numbers. */}
            <dt>
              {row.label}
              {row.unit && ` (${row.unit})`}
            </dt>
            <dd className="mono">{row.value}</dd>
          </div>
        ))}
      </dl>
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
  const [conditionsShown, setConditionsShown] = useState(true);
  const [removing, setRemoving] = useState<DatasetSummary | null>(null);
  const trained = detail != null && trainedIds.has(detail.id);
  const unset = detail != null && !detail.conditions_set;

  return (
    <>
      {series.length === 0 && (
        <div className="dsempty">
          <b>No series yet</b>
          Upload the first high-speed sequence for this project to begin
          calibration and segmentation.
        </div>
      )}
      {/* Each series is a card in the same idiom as the condition cards, and
          the selected one holds them INSIDE it — the containment, not a
          heading, is what says the conditions belong to this series. */}
      <div className="scards">
        {series.map((summary) => {
          const isSelected = summary.id === selected;
          const showsConditions = isSelected && conditionsShown;
          return (
            <section
              key={summary.id}
              className={isSelected ? "scard sel" : "scard"}
              aria-label={`${seriesName(summary)} series`}
            >
              <div className="scard-hd">
                <button
                  type="button"
                  className="scard-sel"
                  aria-current={isSelected || undefined}
                  aria-expanded={showsConditions}
                  title={summary.label ? `series id: ${summary.id}` : undefined}
                  onClick={() => {
                    // Selecting a series opens it; re-clicking the selected
                    // one folds its conditions without deselecting — the
                    // canvas always has exactly one series in view.
                    if (isSelected) setConditionsShown(!conditionsShown);
                    else {
                      setConditionsShown(true);
                      onSelect(summary.id);
                    }
                  }}
                >
                  <span className="scard-ic" aria-hidden="true">
                    <HugeiconsIcon icon={SeriesIcon} size={18} />
                  </span>
                  <span className="scard-m">
                    <b>{seriesName(summary)}</b>
                    <span className="mono">{seriesMeta(summary)}</span>
                  </span>
                </button>
                {/* Per-series actions, not a state chip: trained/ready already
                    reads on the canvas header, so the card carries what you
                    can DO with the series instead of repeating it. */}
                <button
                  type="button"
                  className="scard-act"
                  aria-label={`Edit ${seriesName(summary)} conditions`}
                  title={`Edit ${seriesName(summary)} conditions`}
                  onClick={() => {
                    if (!isSelected) onSelect(summary.id);
                    setEditing(true);
                  }}
                >
                  <HugeiconsIcon icon={EditIcon} size={14} aria-hidden="true" />
                </button>
                <button
                  type="button"
                  className="scard-act danger"
                  aria-label={`Delete ${seriesName(summary)}`}
                  title={`Delete ${seriesName(summary)}`}
                  onClick={() => setRemoving(summary)}
                >
                  <HugeiconsIcon
                    icon={DeleteIcon}
                    size={14}
                    aria-hidden="true"
                  />
                </button>
              </div>

              {showsConditions && detail && detail.id === summary.id && (
                <div className="scard-body">
                  <div
                    className="dbands"
                    aria-label={`${detail.id} conditions`}
                  >
                    {unset ? (
                      <p className="dband-none">
                        No conditions recorded for this series yet — set them
                        below.
                      </p>
                    ) : (
                      DOMAINS.map((domain) => (
                        <DomainBand
                          key={domain.id}
                          domain={domain}
                          conditions={detail.conditions}
                        />
                      ))
                    )}

                    {detail.processed && !detail.conditions_applied && (
                      <div className="ds-reprocess">
                        <Callout tone="caution" title="Re-preprocess required">
                          A tensor-baked condition (frame interval, channel
                          width, or reference velocity) changed since these
                          tensors were built.
                          {trained
                            ? " Re-preprocessing rebuilds them and marks the trained run stale."
                            : " Re-preprocess to rebuild them from the new values."}
                        </Callout>
                        <Button
                          variant="primary"
                          onClick={onPreprocess}
                          disabled={preprocessing}
                        >
                          {preprocessing
                            ? "Re-preprocessing…"
                            : "Re-preprocess"}
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </section>
          );
        })}
      </div>

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
      {removing && (
        <ConfirmDeleteDialog
          title="Delete series"
          confirmLabel="Delete series"
          onClose={() => setRemoving(null)}
          onConfirm={async () => {
            const updated = await api.removeSeries(project.id, removing.id);
            onProjectChanged(updated);
            if (removing.id === selected) {
              const rest = series.filter((s) => s.id !== removing.id);
              if (rest[0]) onSelect(rest[0].id);
            }
            setRemoving(null);
          }}
        >
          Delete <b>{seriesName(removing)}</b> from this project? If no other
          project uses it, its frames, tensors and runs are deleted with it.
        </ConfirmDeleteDialog>
      )}
      {editing && detail && detail.id === selected && (
        <EditConditionsModal
          detail={detail}
          onClose={() => setEditing(false)}
          onSave={onSaveConditions}
          onSaveLabel={onSaveLabel}
        />
      )}

      {/* The shell's own action bar (see .aside-actions): one primary action,
          in a place that never moves. Editing lives on each series card's own
          pencil, so the bar carries the action that grows the library — and
          the state beside it counts what that action changes. The old "saved"
          readout described the edit button this bar no longer holds; its
          useful states moved next to their objects (the re-preprocess callout,
          the amber "not set" bands). */}
      <div className="aside-actions ds-ft">
        <span className="ds-ft-state mono">
          {series.length === 0
            ? "no series yet"
            : series.length === 1
              ? "1 series"
              : `${series.length} series`}
        </span>
        <Button variant="primary" onClick={() => setAdding(true)}>
          <HugeiconsIcon icon={UploadIcon} size={14} aria-hidden="true" />
          Upload new series
        </Button>
      </div>
    </>
  );
}
