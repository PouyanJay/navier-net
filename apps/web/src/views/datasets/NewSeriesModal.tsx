import { useEffect, useMemo, useRef, useState } from "react";

import { Button, Callout } from "../../components";
import { useToast } from "../../components/Toast";
import { api, type ProjectSummary } from "../../lib/api";
import { errorMessage } from "../../lib/errors";
import {
  ALL_CONDITIONS,
  type ConditionInputs,
  DEFAULT_FLUID_ID,
  FluidSelect,
  MeasurementField,
  OPERATING_CONDITIONS,
  parseCondition,
  REQUIRED_CONDITIONS,
  useFluids,
} from "./conditions";

const POLL_INTERVAL_MS = 1000;
const SERIES_ID_RE = /^[A-Za-z0-9._-]+$/;

/** Measurements start empty (must be entered); operating conditions start at
 * their defaults and are edited to override. */
const INITIAL_CONDITIONS: ConditionInputs = {
  dt_frame_ms: "",
  channel_width_um: "",
  channel_height_um: "",
  q_wall_W_cm2: "2.0",
  flow_rate_mL_hr: "5.0",
  U_ref: "0.20",
};

type Phase = "form" | "uploading" | "preprocessing";

interface NewSeriesModalProps {
  project: ProjectSummary;
  onClose: () => void;
  /** Fired as soon as the series is attached (even if preprocessing is still
   * running or failed) so the library reflects reality. */
  onAttached: (project: ProjectSummary, seriesId: string) => void;
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
    useState<ConditionInputs>(INITIAL_CONDITIONS);
  const fluids = useFluids();
  const [fluidId, setFluidId] = useState(DEFAULT_FLUID_ID);
  const [phase, setPhase] = useState<Phase>("form");
  const [error, setError] = useState<string | null>(null);
  const attached = useRef(false);
  const toast = useToast();

  const fluidList = useMemo(
    () => (fluids.status === "ready" ? fluids.list : []),
    [fluids],
  );
  // Default to FC-72 when present, else the first characterised fluid.
  useEffect(() => {
    if (fluidList.length > 0 && !fluidList.some((f) => f.id === fluidId)) {
      setFluidId(fluidList[0].id);
    }
  }, [fluidList, fluidId]);
  const fluidReady = fluidList.length > 0;

  const validName = SERIES_ID_RE.test(name) && !project.datasets.includes(name);
  const parsed = ALL_CONDITIONS.map((spec) => ({
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
    if (!validName || !files?.length || !conditionsValid || !fluidReady) {
      return;
    }
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
        fluid: fluidId,
        ...Object.fromEntries(
          parsed.map((entry) => [entry.spec.key, entry.value ?? undefined]),
        ),
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
        className="modal series-modal"
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

          <div className="modal-section">
            <h3 className="modal-section-hd">
              Measurements <span>frame interval & width bake into the tensors</span>
            </h3>
            <div className="frm">
              {REQUIRED_CONDITIONS.map((spec) => (
                <MeasurementField
                  key={spec.key}
                  spec={spec}
                  value={conditions[spec.key]}
                  disabled={busy}
                  onChange={(next) =>
                    setConditions((c) => ({ ...c, [spec.key]: next }))
                  }
                />
              ))}
            </div>
          </div>

          <div className="modal-section">
            <h3 className="modal-section-hd">
              Operating conditions <span>defaults · edit to override</span>
            </h3>
            <div className="frm">
              <FluidSelect
                fluids={fluids}
                fluidId={fluidId}
                disabled={busy}
                onChange={setFluidId}
              />
              {OPERATING_CONDITIONS.map((spec) => (
                <MeasurementField
                  key={spec.key}
                  spec={spec}
                  value={conditions[spec.key]}
                  disabled={busy}
                  onChange={(next) =>
                    setConditions((c) => ({ ...c, [spec.key]: next }))
                  }
                />
              ))}
            </div>
          </div>

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
                busy ||
                !validName ||
                !files?.length ||
                !conditionsValid ||
                !fluidReady
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
