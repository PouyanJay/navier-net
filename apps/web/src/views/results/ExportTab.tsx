import { Chip, Panel } from "../../components";
import { useToast } from "../../components/Toast";
import { artifactUrl, type RunDetail, type RunSummary } from "../../lib/api";
import { formatRunDate, runConditions, runDisplayName } from "./format";
import type { RunCapability } from "./runCapability";
import { StateNote } from "./StateNote";

const exportUrl = {
  iou: (id: string) => `/api/runs/${encodeURIComponent(id)}/export/iou.csv`,
  trajectory: (id: string, dataset?: string | null) =>
    `/api/runs/${encodeURIComponent(id)}/export/trajectory.csv` +
    (dataset ? `?dataset=${encodeURIComponent(dataset)}` : ""),
  frontVelocity: (id: string, dataset?: string | null) =>
    `/api/runs/${encodeURIComponent(id)}/export/front-velocity.csv` +
    (dataset ? `?dataset=${encodeURIComponent(dataset)}` : ""),
  loss: (id: string) => `/api/runs/${encodeURIComponent(id)}/export/loss.csv`,
};

interface RowProps {
  name: string;
  detail: string;
  href?: string;
  action?: React.ReactNode;
}

function Row({ name, detail, href, action }: RowProps) {
  return (
    <div className="dl-row">
      <span className="dl-name">
        {name}
        <small>{detail}</small>
      </span>
      {href ? (
        <a className="btn-link" href={href} download>
          Download
        </a>
      ) : (
        action
      )}
    </div>
  );
}

interface ExportTabProps {
  run: RunSummary;
  capability: RunCapability;
  detail: RunDetail | null;
  viewDataset: string | null;
  datasetLabels: Map<string, string>;
}

/** Everything the run produced, and the flat CSVs a paper's analysis wants —
 * all regenerable from raw data plus the run's config snapshot. */
export function ExportTab({
  run,
  capability,
  detail,
  viewDataset,
  datasetLabels,
}: ExportTabProps) {
  const toast = useToast();
  const artifacts = detail?.artifacts;
  const { all } = runConditions(run);
  const joint = all.length > 1;

  const citation = () => {
    const date = formatRunDate(run.date) ?? run.date ?? "";
    // Labels for the reader, ids in parentheses for reproducibility.
    const conditions = all
      .map((id) => {
        const label = datasetLabels.get(id) ?? id;
        return label === id ? id : `${label} (${id})`;
      })
      .join(", ");
    const text = `naviernet run ${runDisplayName(run, datasetLabels)} (outputs/${run.id}) · conditions ${conditions} · ${date} · ${window.location.href}`;
    navigator.clipboard
      ?.writeText(text)
      .then(() => toast("Citation copied", text, "ok"))
      .catch(() => toast("Copy failed", "clipboard unavailable", "err"));
  };

  return (
    <>
      <Panel title="Run artifacts" subtitle={`outputs/${run.id}/`}>
        {!artifacts ? (
          <p className="res-quiet">Loading artifacts…</p>
        ) : (
          <>
            {artifacts.checkpoint && (
              <Row
                name="Checkpoint"
                detail="checkpoints/ckpt.pt · resume-capable"
                href={artifactUrl.checkpoint(run.id)}
              />
            )}
            <Row
              name="Training tensors"
              detail={
                joint
                  ? `${all.length} datasets · data/processed/<series>/tensors.npz`
                  : "training_data tensors the model supervised on"
              }
              href={joint ? undefined : artifactUrl.tensors(run.id)}
              action={
                joint ? <Chip>per series · see Datasets</Chip> : undefined
              }
            />
            {artifacts.figures.length > 0 ? (
              <Row
                name="Figures"
                detail={`figures/ · ${artifacts.figures.length} PNG`}
                href={artifactUrl.figure(run.id, artifacts.figures[0])}
              />
            ) : (
              <Row
                name="Figures & video"
                detail={
                  joint
                    ? "not generated for joint runs yet · planned"
                    : "no figures were rendered for this run · re-run the render stage"
                }
                action={<Chip>none</Chip>}
              />
            )}
            {artifacts.video && (
              <Row
                name="Reconstruction video"
                detail="video/growth.mp4 · H.264"
                href={artifactUrl.video(run.id)}
              />
            )}
          </>
        )}
      </Panel>

      <Panel
        title="Export for analysis"
        subtitle="flat CSV · regenerable from raw"
      >
        <Row
          name="Per-frame IoU"
          detail="dataset, camera_frame, iou, role · every scored frame"
          href={exportUrl.iou(run.id)}
        />
        <Row
          name="Growth kinematics"
          detail={`series, t_ms, nose_um, area_um2${joint && viewDataset ? ` · ${datasetLabels.get(viewDataset) ?? viewDataset}` : ""}`}
          href={exportUrl.trajectory(run.id, joint ? viewDataset : null)}
        />
        {/* Offered only when the report exists: the download 404s otherwise,
            and a link that fails is worse than one that is absent. */}
        <Row
          name="Front velocity"
          detail={
            capability.hasFrontVelocity
              ? `series, t_ms, s, v_um_per_ms, v_m_per_s, heldout${joint && viewDataset ? ` · ${datasetLabels.get(viewDataset) ?? viewDataset}` : ""}`
              : "this run measured no front velocity"
          }
          href={
            capability.hasFrontVelocity
              ? exportUrl.frontVelocity(run.id, joint ? viewDataset : null)
              : undefined
          }
          action={capability.hasFrontVelocity ? undefined : <Chip>none</Chip>}
        />
        <Row
          name="Loss history"
          detail="step, lr, every recorded loss term"
          href={exportUrl.loss(run.id)}
        />
        <Row
          name="Cite this run"
          detail="run id · conditions · date · deep link, for a methods section"
          action={
            <button type="button" className="btn-link" onClick={citation}>
              Copy
            </button>
          }
        />
        <Row
          name="Validation report"
          detail="bundled HTML/PDF of this page · planned (follow-up feature)"
          action={<Chip>planned</Chip>}
        />
        {capability.replayable ? (
          <p className="note">
            <b>Everything regenerable.</b> Any export can be rebuilt from the
            raw TIFFs plus this run&apos;s config snapshot; nothing here is
            hand-edited.
          </p>
        ) : (
          <StateNote
            tone="caution"
            title="This run cannot be reproduced exactly."
          >
            It recorded no config snapshot, so the exact configuration that
            produced these numbers is not on disk. The measurements below are
            still exactly what it wrote.
          </StateNote>
        )}
      </Panel>
    </>
  );
}
