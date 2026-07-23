import { Chip, type Column, IouChart, Panel, Table, ViewCanvas } from "../../components";
import type { FramePoint } from "../../components";
import type { RunDetail } from "../../lib/api";

interface FrameRow extends FramePoint {
  tMs: number | null;
}

const COLUMNS: Column<FrameRow>[] = [
  { header: "Frame", cell: (r) => r.frame, num: true },
  { header: "t (ms)", cell: (r) => (r.tMs != null ? r.tMs.toFixed(1) : "—"), num: true },
  { header: "IoU", cell: (r) => r.iou.toFixed(3), num: true },
  {
    header: "",
    cell: (r) => (r.holdout ? <Chip tone="amber">holdout — never supervised</Chip> : null),
  },
];

function frameDtMs(detail: RunDetail): number | null {
  const config = detail.config as { experiment?: { dt_frame_ms?: number } } | null;
  const dt = config?.experiment?.dt_frame_ms;
  return typeof dt === "number" ? dt : null;
}

function toRows(detail: RunDetail): FrameRow[] {
  const perFrame = detail.metrics?.iou_per_frame ?? {};
  const holdout = detail.metrics?.holdout_frame ?? null;
  const dt = frameDtMs(detail);
  return Object.entries(perFrame)
    .map(([frame, iou]) => {
      const frameNumber = Number(frame);
      return {
        frame: frameNumber,
        iou,
        holdout: frameNumber === holdout,
        tMs: dt != null ? (frameNumber - 1) * dt : null,
      };
    })
    .sort((a, b) => a.frame - b.frame);
}

/** Agreement per frame: IoU vs the segmented experiment, holdout called out. */
export function AgreementPerFrame({ detail }: { detail: RunDetail }) {
  const rows = toRows(detail);

  if (rows.length === 0) {
    return (
      <Panel title="Agreement per frame" subtitle="IoU vs the segmented experiment">
        <p className="state-note">This run has no per-frame agreement recorded.</p>
      </Panel>
    );
  }

  return (
    <Panel title="Agreement per frame" subtitle="IoU vs the segmented experiment">
      <ViewCanvas>
        <IouChart data={rows} />
      </ViewCanvas>
      <div className="agreement-table">
        <Table
          columns={COLUMNS}
          rows={rows}
          rowKey={(r) => String(r.frame)}
          rowTone={(r) => (r.holdout ? "amber" : undefined)}
          caption="Intersection-over-union per frame"
        />
      </div>
    </Panel>
  );
}
