import { useMemo, useState } from "react";

import { Callout, Chip, Panel } from "../../components";
import { ArtifactImage } from "../../components/ArtifactImage";
import {
  api,
  artifactUrl,
  type DatasetDetail,
  type InterfaceData,
  type PhysicsValidation,
  type QcData,
  type RunMetrics,
} from "../../lib/api";
import { ringPath, umPath, type FrameGeometry } from "../../lib/frameGeometry";
import { fmtIou } from "./format";
import { useApiResource } from "./useApiResource";

type FrameRole = "supervised" | "validation" | "holdout" | "transfer";

const ROLE_LABEL: Record<FrameRole, string> = {
  supervised: "supervised",
  validation: "validation · axis A",
  holdout: "holdout · never supervised",
  transfer: "transfer · never trained",
};

interface FrameEntry {
  /** 1-based camera frame. */
  frame: number;
  timeMs: number;
  iou: number | null;
  role: FrameRole;
}

/** The camera frames a run was scored on, with each frame's role. */
function frameEntries(
  metrics: RunMetrics | null,
  validation: PhysicsValidation | null,
  dataset: string | null,
  dtFrameMs: number,
): FrameEntry[] {
  // Joint runs keep per-dataset agreement; single runs keep it at top level.
  const block =
    dataset && validation?.per_dataset?.[dataset]
      ? validation.per_dataset[dataset]
      : null;
  const perFrame = block?.iou_per_frame ?? metrics?.iou_per_frame ?? {};
  const validationFrames = new Set(
    block?.validation_frames ?? metrics?.validation_frames ?? [],
  );
  const holdout = block ? null : (metrics?.holdout_frame ?? null);
  const transfer =
    dataset != null &&
    validation?.heldout_datasets != null &&
    validation.heldout_datasets.includes(dataset);
  return Object.entries(perFrame)
    .map(([key, iou]) => {
      const frame = Number(key);
      const role: FrameRole = transfer
        ? "transfer"
        : frame === holdout
          ? "holdout"
          : validationFrames.has(frame)
            ? "validation"
            : "supervised";
      return { frame, timeMs: (frame - 1) * dtFrameMs, iou, role };
    })
    .sort((a, b) => a.frame - b.frame);
}

interface FrameMatchPanelProps {
  runId: string;
  /** The camera frames' dataset (the viewing condition for joint runs). */
  dataset: string | null;
  metrics: RunMetrics | null;
  validation: PhysicsValidation | null;
  /** Joint runs can't evaluate the PINN layer yet (single checkpoint loader). */
  pinnAvailable: boolean;
}

/**
 * Frame-by-frame evidence: the raw camera image, the bubble the preprocess
 * pipeline detected (QC silhouette), and the bubble the PINN models — each
 * toggleable, with the frame's IoU and validation role. The interactive
 * version of the pipeline's pinn_on_image figures.
 */
export function FrameMatchPanel({
  runId,
  dataset,
  metrics,
  validation,
  pinnAvailable,
}: FrameMatchPanelProps) {
  const detailQ = useApiResource<DatasetDetail>(dataset, (id) =>
    api.getDataset(id),
  );
  const qcQ = useApiResource<QcData>(dataset, (id) => api.getQcData(id), {
    nullOn404: true,
  });
  const ifaceQ = useApiResource<InterfaceData>(
    pinnAvailable ? runId : null,
    (id) => api.getInterface(id),
    { nullOn404: true },
  );
  const [selected, setSelected] = useState(0);
  const [layers, setLayers] = useState({ camera: true, det: true, pinn: true });

  const detail = detailQ.data;
  const frames = useMemo(
    () =>
      frameEntries(
        metrics,
        validation,
        dataset,
        detail?.conditions.dt_frame_ms ?? 0,
      ),
    [metrics, validation, dataset, detail],
  );
  const current = frames[Math.min(selected, Math.max(frames.length - 1, 0))];

  const geo: FrameGeometry | null =
    qcQ.data && detail?.frame_px && detail.um_per_px != null
      ? {
          width: detail.frame_px[0],
          height: detail.frame_px[1],
          pxPerStar: qcQ.data.interface.l_ref_um / detail.um_per_px,
          yRoiTop: qcQ.data.interface.y_roi_top,
        }
      : null;

  const detected = current
    ? (qcQ.data?.interface.frames.find(
        (f) => f.camera_frame === current.frame,
      ) ?? null)
    : null;

  // The PINN contour at this camera instant: the reconstructed frame nearest
  // in time (the continuous player samples far finer than the camera).
  const pinnContours = useMemo(() => {
    if (!ifaceQ.data || !current) return null;
    let best = ifaceQ.data.frames[0];
    for (const frame of ifaceQ.data.frames) {
      if (
        Math.abs(frame.t_ms - current.timeMs) <
        Math.abs(best.t_ms - current.timeMs)
      )
        best = frame;
    }
    return best ?? null;
  }, [ifaceQ.data, current]);

  if (!dataset)
    return (
      <Panel title="Frame matching" subtitle="camera · detected · PINN">
        <p className="state-note">Select a condition to compare frames.</p>
      </Panel>
    );
  if (detailQ.error)
    return (
      <Panel title="Frame matching" subtitle={dataset}>
        <Callout tone="error" title="Could not load the series">
          {detailQ.error}
        </Callout>
      </Panel>
    );

  const toggle = (key: keyof typeof layers) =>
    setLayers((state) => ({ ...state, [key]: !state[key] }));

  return (
    <Panel
      title="Frame matching"
      subtitle={`camera · detected · PINN — ${dataset}`}
    >
      {frames.length === 0 ? (
        <p className="state-note">
          No per-frame agreement recorded; re-run the evaluate stage.
        </p>
      ) : (
        <>
          <div className="fm-view">
            <div className="fm-stage">
              {layers.camera && detail && current ? (
                <ArtifactImage
                  src={artifactUrl.datasetFrame(dataset, current.frame)}
                  alt={`Camera frame ${current.frame} of ${dataset}`}
                />
              ) : (
                <div className="fm-dark" aria-hidden="true" />
              )}
              {geo && current && (
                <svg
                  className="fm-overlay"
                  viewBox={`0 0 ${geo.width} ${geo.height}`}
                  preserveAspectRatio="none"
                  aria-hidden="true"
                >
                  {layers.det && detected && (
                    <path
                      className="fm-detected"
                      d={detected.rings
                        .map((ring) => ringPath(ring, geo))
                        .join(" ")}
                    />
                  )}
                  {layers.pinn &&
                    detail?.um_per_px != null &&
                    pinnContours &&
                    pinnContours.contours.map((contour, i) => (
                      <path
                        key={i}
                        className="fm-pinn"
                        d={umPath(contour, geo, detail.um_per_px!)}
                      />
                    ))}
                </svg>
              )}
            </div>
            <div className="fm-bar">
              <div
                className="layerbtns"
                role="group"
                aria-label="Overlay layers"
              >
                <button
                  type="button"
                  className="lbtn"
                  aria-pressed={layers.camera}
                  onClick={() => toggle("camera")}
                >
                  camera
                </button>
                <button
                  type="button"
                  className="lbtn"
                  aria-pressed={layers.det}
                  onClick={() => toggle("det")}
                >
                  detected
                </button>
                <button
                  type="button"
                  className="lbtn"
                  aria-pressed={layers.pinn}
                  onClick={() => toggle("pinn")}
                  disabled={!pinnAvailable}
                  title={
                    pinnAvailable
                      ? undefined
                      : "PINN overlay for joint runs arrives with per-condition evaluation"
                  }
                >
                  PINN
                </button>
              </div>
              {current && (
                <span className="fm-meta">
                  f{String(current.frame).padStart(2, "0")} ·{" "}
                  {current.timeMs.toFixed(1)} ms · IoU{" "}
                  <b>{fmtIou(current.iou)}</b>
                </span>
              )}
              {current && (
                <Chip
                  tone={current.role === "supervised" ? "default" : "amber"}
                >
                  {ROLE_LABEL[current.role]}
                </Chip>
              )}
            </div>
          </div>

          <div className="fm-strip" role="group" aria-label="Camera frames">
            {frames.map((entry, index) => (
              <button
                key={entry.frame}
                type="button"
                className={"fm-thumb" + (index === selected ? " on" : "")}
                aria-pressed={index === selected}
                aria-label={`Frame ${entry.frame}, IoU ${fmtIou(entry.iou)}, ${ROLE_LABEL[entry.role]}`}
                onClick={() => setSelected(index)}
              >
                <ArtifactImage
                  src={artifactUrl.datasetFrame(dataset, entry.frame)}
                  alt=""
                />
                <span className="fm-thumb-meta">
                  f{String(entry.frame).padStart(2, "0")} · {fmtIou(entry.iou)}
                  {entry.role === "holdout" && <em> · HOLDOUT</em>}
                </span>
              </button>
            ))}
          </div>

          <p className="figcap">
            <b>Figure 2.</b> The three layers of evidence per camera instant:
            what the camera saw, what the preprocess pipeline detected, and what
            the PINN reconstructs. The gap between the detected and modeled
            contours is exactly what each frame's IoU measures.
          </p>
        </>
      )}
    </Panel>
  );
}
