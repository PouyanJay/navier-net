import { useEffect, useId, useRef, useState } from "react";

import { Button, Callout } from "../../components";
import { useToast } from "../../components/Toast";
import { api, type ProjectSummary } from "../../lib/api";
import { errorMessage } from "../../lib/errors";

const POLL_INTERVAL_MS = 1000;
const SERIES_ID_RE = /^[A-Za-z0-9._-]+$/;

/**
 * The two conditions preprocessing cannot be run without, with the server-side
 * bounds they are validated against (`CONDITION_FIELDS` in the datasets
 * service). Everything else about a series is editable afterwards in the
 * conditions panel; these two are baked into the tensors, so asking later would
 * mean asking for a re-run.
 */
const REQUIRED_CONDITIONS = [
  {
    key: "dt_frame_ms",
    label: "Frame interval",
    hint: "Δt between frames",
    unit: "ms",
    step: 0.1,
    placeholder: "0.5",
    min: 1e-6,
    max: 1e4,
    why: "sets the time axis of the tensors",
  },
  {
    key: "channel_width_um",
    label: "Channel width",
    hint: "imaged direction",
    unit: "µm",
    step: 10,
    placeholder: "300",
    min: 1,
    max: 1e6,
    why: "calibrates µm/px from the detected walls",
  },
] as const;

type ConditionKey = (typeof REQUIRED_CONDITIONS)[number]["key"];
type ConditionInputs = Record<ConditionKey, string>;

const EMPTY_CONDITIONS: ConditionInputs = {
  dt_frame_ms: "",
  channel_width_um: "",
};

type Phase = "form" | "uploading" | "preprocessing";

interface NewSeriesModalProps {
  project: ProjectSummary;
  onClose: () => void;
  /** Fired as soon as the series is attached (even if preprocessing is still
   * running or failed) so the library reflects reality. */
  onAttached: (project: ProjectSummary, seriesId: string) => void;
}

/** Parse a required measurement, or null when it is missing or out of range. */
function parseCondition(raw: string, min: number, max: number): number | null {
  const value = Number(raw);
  if (raw.trim() === "" || !Number.isFinite(value)) return null;
  return value >= min && value <= max ? value : null;
}

/** Upload a new series and run its preprocessing in one guided flow:
 * name → frames → the conditions preprocessing needs → upload → pipeline runs. */
export function NewSeriesModal({
  project,
  onClose,
  onAttached,
}: NewSeriesModalProps) {
  const [name, setName] = useState("");
  const [files, setFiles] = useState<FileList | null>(null);
  const [conditions, setConditions] =
    useState<ConditionInputs>(EMPTY_CONDITIONS);
  const [phase, setPhase] = useState<Phase>("form");
  const [error, setError] = useState<string | null>(null);
  const attached = useRef(false);
  const toast = useToast();

  const validName = SERIES_ID_RE.test(name) && !project.datasets.includes(name);
  const parsed = REQUIRED_CONDITIONS.map((spec) => ({
    spec,
    value: parseCondition(conditions[spec.key], spec.min, spec.max),
  }));
  const conditionsValid = parsed.every((entry) => entry.value !== null);

  // The pipeline reports states, not percentages; poll until it settles.
  useEffect(() => {
    if (phase !== "preprocessing") return;
    const id = window.setInterval(async () => {
      try {
        const status = await api.getPreprocessStatus(name);
        if (status.state === "done") {
          window.clearInterval(id);
          toast("Series ready", `${name}: tensors and QC available`, "ok");
          onClose();
        }
        if (status.state === "error") {
          window.clearInterval(id);
          setError(
            `Preprocessing failed: ${status.message ?? "see the API log"}. ` +
              "The frames are uploaded; you can rerun preprocessing from the sequence panel.",
          );
          setPhase("form");
        }
      } catch (err) {
        window.clearInterval(id);
        setError(`Lost track of the preprocessing job: ${errorMessage(err)}`);
        setPhase("form");
      }
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [phase, name, onClose, toast]);

  async function start() {
    if (!validName || !files?.length || !conditionsValid) return;
    setError(null);
    setPhase("uploading");
    try {
      await api.uploadFrames(name, files);
    } catch (err) {
      setError(`Upload failed: ${errorMessage(err)}`);
      setPhase("form");
      return;
    }
    try {
      // Before preprocessing, not after: these values are baked into the
      // calibration and the time axis, so running first would produce tensors
      // built from another series' conditions.
      await api.updateConditions(name, {
        dt_frame_ms: parsed[0].value ?? undefined,
        channel_width_um: parsed[1].value ?? undefined,
      });
    } catch (err) {
      setError(
        `The frames are uploaded, but the operating conditions could not be saved: ` +
          `${errorMessage(err)}. Preprocessing was not started, since it would have ` +
          `used another series' values.`,
      );
      setPhase("form");
      return;
    }
    try {
      const updated = await api.updateProject(project.id, {
        datasets: [...project.datasets, name],
      });
      attached.current = true;
      onAttached(updated, name);
    } catch (err) {
      setError(`Uploaded, but linking the series failed: ${errorMessage(err)}`);
      setPhase("form");
      return;
    }
    try {
      await api.startPreprocess(name);
      setPhase("preprocessing");
    } catch (err) {
      setError(
        `The series is uploaded, but preprocessing could not start: ${errorMessage(err)}`,
      );
      setPhase("form");
    }
  }

  const busy = phase !== "form";

  return (
    <div
      className="modal-ov"
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && phase === "form") onClose();
      }}
    >
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label="Upload new series"
      >
        <div className="hd">
          <h2>Upload new series</h2>
          <span className="sub">TIFF frames · preprocesses on upload</span>
        </div>
        <div className="body">
          <label className="pform-field">
            Series name
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="mid_T"
              disabled={busy}
              autoFocus
            />
          </label>
          {name.length > 0 && !validName && (
            <Callout tone="error">
              {project.datasets.includes(name)
                ? "A series with this name already exists in the project."
                : "Series names use letters, digits, dots, dashes, and underscores."}
            </Callout>
          )}
          <label className="drop">
            <input
              type="file"
              accept=".tif,.tiff,image/tiff"
              multiple
              disabled={busy}
              onChange={(e) => setFiles(e.target.files)}
              aria-label="Image sequence (TIFF frames)"
            />
            {files?.length ? (
              <>
                <b>{files.length}</b> frame{files.length === 1 ? "" : "s"}{" "}
                selected · <b>browse</b> to change
              </>
            ) : (
              <>
                <b>Browse</b> for the high-speed TIFF frames
              </>
            )}
          </label>

          <div className="frm">
            {REQUIRED_CONDITIONS.map((spec) => (
              <MeasurementField
                key={spec.key}
                spec={spec}
                value={conditions[spec.key]}
                disabled={busy}
                onChange={(next) =>
                  setConditions((current) => ({ ...current, [spec.key]: next }))
                }
              />
            ))}
          </div>
          <p className="note">
            <b>Why these two now</b> They are the only conditions preprocessing
            itself consumes: the frame interval sets the time axis and the
            channel width calibrates µm/px. The rest of the operating conditions
            are editable in the conditions panel once the series is in.
          </p>

          {phase !== "form" && (
            <div className="modal-progress">
              <div
                className="meter indeterminate"
                role="progressbar"
                aria-label={phase}
              >
                <i />
              </div>
              <p className="state-note" role="status">
                {phase === "uploading"
                  ? `Uploading ${files?.length ?? 0} frames…`
                  : "Preprocessing: calibrating, segmenting, building tensors…"}
              </p>
            </div>
          )}
          {error && <Callout tone="error">{error}</Callout>}

          <div className="pform-actions">
            <Button
              variant="primary"
              onClick={() => void start()}
              disabled={
                busy || !validName || !files?.length || !conditionsValid
              }
            >
              {busy ? "Working…" : "Upload & preprocess"}
            </Button>
            <Button onClick={onClose} disabled={phase === "uploading"}>
              {phase === "preprocessing" ? "Continue in background" : "Cancel"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

interface MeasurementFieldProps {
  spec: (typeof REQUIRED_CONDITIONS)[number];
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
}

/** A required measurement with its unit, in the conditions form's own shell.
 * Unlike the shared NumberField it starts empty: there is no honest default for
 * a series nobody has described yet, so the value has to be entered. */
function MeasurementField({
  spec,
  value,
  disabled,
  onChange,
}: MeasurementFieldProps) {
  const id = useId();
  const out =
    value.trim() !== "" && parseCondition(value, spec.min, spec.max) === null;
  return (
    <div className="fld">
      <label htmlFor={id}>
        {spec.label}
        <span className="hint">{spec.hint}</span>
      </label>
      <span className="ug" data-disabled={disabled || undefined}>
        <input
          id={id}
          type="number"
          inputMode="decimal"
          value={value}
          step={spec.step}
          min={spec.min}
          max={spec.max}
          placeholder={spec.placeholder}
          disabled={disabled}
          aria-invalid={out || undefined}
          aria-describedby={`${id}-why`}
          onChange={(e) => onChange(e.target.value)}
        />
        <span className="sfx">{spec.unit}</span>
      </span>
      <span id={`${id}-why`} className="fld-why">
        {out
          ? `Must be between ${spec.min} and ${spec.max} ${spec.unit}.`
          : spec.why}
      </span>
    </div>
  );
}
