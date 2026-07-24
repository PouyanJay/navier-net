import { useState } from "react";

import { Button, Panel } from "../../components";
import { useToast } from "../../components/Toast";
import { api, type ProjectSummary } from "../../lib/api";
import { errorMessage } from "../../lib/errors";
import "./datasets.css";

interface EmptyProjectUploadProps {
  project: ProjectSummary;
  /** Called with the updated project once its dataset has been attached. */
  onAttached: (project: ProjectSummary) => void;
}

/** The datasets stage of a project with no data yet: upload the first
 * sequence. The dataset takes the project's id, then gets linked to it. */
export function EmptyProjectUpload({ project, onAttached }: EmptyProjectUploadProps) {
  const [files, setFiles] = useState<FileList | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const toast = useToast();

  async function upload() {
    if (!files?.length) return;
    setBusy(true);
    setError(null);
    try {
      await api.uploadFrames(project.id, files);
    } catch (err) {
      setError(`Upload failed: ${errorMessage(err)}`);
      setBusy(false);
      return;
    }
    try {
      // Frames are on disk now — a failure past this point is only the link.
      const updated = await api.updateProject(project.id, { dataset: project.id });
      toast("Sequence uploaded", `${files.length} frames — ready to preprocess`, "ok");
      onAttached(updated);
    } catch (err) {
      setError(`Uploaded, but linking the dataset failed: ${errorMessage(err)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel
      title="Upload the first sequence"
      subtitle="This project has no data yet — TIFF frames begin its calibration"
    >
      <div className="upload-row">
        <input
          type="file"
          accept=".tif,.tiff,image/tiff"
          multiple
          aria-label="Image sequence (TIFF frames)"
          onChange={(e) => setFiles(e.target.files)}
        />
        <Button variant="primary" onClick={upload} disabled={busy || !files?.length}>
          {busy ? "Uploading…" : "Upload sequence"}
        </Button>
      </div>
      {error && (
        <p className="state-note error" role="alert">
          {error}
        </p>
      )}
    </Panel>
  );
}
