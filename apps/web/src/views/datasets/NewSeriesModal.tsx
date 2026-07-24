import { useEffect, useRef, useState } from "react";

import { Button } from "../../components";
import { useToast } from "../../components/Toast";
import { api, type ProjectSummary } from "../../lib/api";
import { errorMessage } from "../../lib/errors";

const POLL_INTERVAL_MS = 1000;
const SERIES_ID_RE = /^[A-Za-z0-9._-]+$/;

type Phase = "form" | "uploading" | "preprocessing";

interface NewSeriesModalProps {
  project: ProjectSummary;
  onClose: () => void;
  /** Fired as soon as the series is attached (even if preprocessing is still
   * running or failed) so the library reflects reality. */
  onAttached: (project: ProjectSummary, seriesId: string) => void;
}

/** Upload a new series and run its preprocessing in one guided flow:
 * choose frames → upload → pipeline runs (progress) → data + QC appear. */
export function NewSeriesModal({ project, onClose, onAttached }: NewSeriesModalProps) {
  const [name, setName] = useState("");
  const [files, setFiles] = useState<FileList | null>(null);
  const [phase, setPhase] = useState<Phase>("form");
  const [error, setError] = useState<string | null>(null);
  const attached = useRef(false);
  const toast = useToast();

  const validName = SERIES_ID_RE.test(name) && !project.datasets.includes(name);

  // The pipeline reports states, not percentages — poll until it settles.
  useEffect(() => {
    if (phase !== "preprocessing") return;
    const id = window.setInterval(async () => {
      try {
        const status = await api.getPreprocessStatus(name);
        if (status.state === "done") {
          window.clearInterval(id);
          toast("Series ready", `${name} — tensors and QC available`, "ok");
          onClose();
        }
        if (status.state === "error") {
          window.clearInterval(id);
          setError(
            `Preprocessing failed: ${status.message ?? "see the API log"}. ` +
              "The frames are uploaded — you can rerun preprocessing from the sequence panel.",
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
    if (!validName || !files?.length) return;
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
      // Frames are on disk now — a failure past this point is only the link.
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
      <div className="modal" role="dialog" aria-modal="true" aria-label="Upload new series">
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
            <p className="state-note error">
              {project.datasets.includes(name)
                ? "A series with this name already exists in the project."
                : "Series names use letters, digits, dots, dashes, and underscores."}
            </p>
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
                <b>{files.length}</b> frame{files.length === 1 ? "" : "s"} selected —{" "}
                <b>browse</b> to change
              </>
            ) : (
              <>
                <b>Browse</b> for the high-speed TIFF frames
              </>
            )}
          </label>

          {phase !== "form" && (
            <div className="modal-progress">
              <div className="meter indeterminate" role="progressbar" aria-label={phase}>
                <i />
              </div>
              <p className="state-note" role="status">
                {phase === "uploading"
                  ? `Uploading ${files?.length ?? 0} frames…`
                  : "Preprocessing — calibrating, segmenting, building tensors…"}
              </p>
            </div>
          )}
          {error && (
            <p className="state-note error" role="alert">
              {error}
            </p>
          )}

          <div className="pform-actions">
            <Button
              variant="primary"
              onClick={() => void start()}
              disabled={busy || !validName || !files?.length}
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
