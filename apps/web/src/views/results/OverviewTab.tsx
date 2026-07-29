import { Panel, Stat } from "../../components";
import type { PhysicsValidation, RunDetail, RunSummary } from "../../lib/api";
import { runConditions } from "./format";
import type { ResultTabId } from "./ResultsPage";

/** Nose-speed agreement within this band reads as a pass (mockup contract). */
const NOSE_SPEED_TOLERANCE_PCT = 10;

const fmtIou = (value: number | null | undefined) =>
  value != null ? value.toFixed(3) : "—";

interface OverviewTabProps {
  run: RunSummary;
  detail: RunDetail | null;
  validation: PhysicsValidation | null;
  datasetLabels: Map<string, string>;
  onOpenTab: (tab: ResultTabId) => void;
}

/** The verdict sentence the scorecard argues, per run shape. */
function narrative(
  run: RunSummary,
  validation: PhysicsValidation | null,
  labelOf: (id: string) => string,
): string {
  const { all, heldout } = runConditions(run);
  const axisA = validation?.val_iou_mean ?? validation?.iou_val ?? null;
  const transfer = validation?.transfer_iou_mean ?? null;
  const trained = all.filter((id) => !heldout.has(id));
  if (transfer != null) {
    const held = [...heldout].map(labelOf).join(", ");
    return (
      `Trained on ${trained.length} condition${trained.length === 1 ? "" : "s"}, ` +
      `this run reconstructs unseen instants of those conditions to IoU ${fmtIou(axisA)}, ` +
      `and an entirely unseen condition — ${held}, supplied only as dimensionless groups — ` +
      `to IoU ${fmtIou(transfer)}. The transfer number is the evidence that the model ` +
      `learned physics, not footage.`
    );
  }
  if (all.length > 1) {
    return (
      `Trained jointly on all ${all.length} conditions, this run reconstructs unseen ` +
      `instants to IoU ${fmtIou(axisA)}. No condition was held out, so transfer ` +
      `(axis B) is untested — hold one out in the Solver to claim it.`
    );
  }
  return (
    `A single-condition run: agreement on the never-supervised frame ` +
    `${validation?.holdout_frame ?? "—"} is ${fmtIou(validation?.iou_holdout)}` +
    (validation?.nose_speed_error_pct != null
      ? `, and the inferred nose speed lands within ${validation.nose_speed_error_pct.toFixed(1)} % of the measured value`
      : "") +
    ` — generalization in time, on one condition.`
  );
}

/**
 * The landing tab: the two-axis generalization scorecard, the verdict it
 * argues, and jump-offs to the evidence.
 */
export function OverviewTab({
  run,
  detail,
  validation,
  datasetLabels,
  onOpenTab,
}: OverviewTabProps) {
  const labelOf = (id: string) => datasetLabels.get(id) ?? id;
  const { all, heldout } = runConditions(run);
  const axisA = validation?.val_iou_mean ?? validation?.iou_val ?? null;
  const transfer = validation?.transfer_iou_mean ?? null;
  const noseErr = validation?.nose_speed_error_pct ?? null;
  const figureCount = detail?.artifacts.figures.length ?? 0;

  const glance: { tab: ResultTabId; label: string }[] = [
    {
      tab: "agreement",
      label: `Agreement · ${all.length} condition${all.length === 1 ? "" : "s"}`,
    },
    { tab: "fields", label: "Field maps" },
    { tab: "training", label: "Training diagnostics" },
    {
      tab: "export",
      label: `Artifacts${figureCount > 0 ? ` · ${figureCount} figures` : ""}`,
    },
  ];

  return (
    <>
      <Panel
        title="Generalization scorecard"
        subtitle="the two validation axes · evaluate stage"
        className="overview-scorecard"
      >
        <div data-testid="overview-scorecard" className="scorecard">
          <Stat
            label="In-distribution IoU · axis A"
            value={fmtIou(axisA)}
            tone={axisA != null ? "green" : "default"}
            hint={
              axisA != null
                ? "held-out frames of trained conditions"
                : "no validation split on this run"
            }
          />
          <Stat
            label="Transfer IoU · axis B"
            value={fmtIou(transfer)}
            tone={transfer != null ? "green" : "default"}
            hint={
              transfer != null
                ? `all frames of ${[...heldout].map(labelOf).join(", ")} · never trained`
                : "no held-out condition configured"
            }
          />
          <Stat
            label="Holdout-frame IoU"
            value={fmtIou(validation?.iou_holdout)}
            hint={
              validation?.holdout_frame != null
                ? `frame ${validation.holdout_frame} · never supervised`
                : "no holdout frame"
            }
          />
          <Stat
            label="Nose-speed error"
            value={noseErr != null ? noseErr.toFixed(1) : "—"}
            unit={noseErr != null ? "%" : undefined}
            tone={
              noseErr == null
                ? "default"
                : noseErr < NOSE_SPEED_TOLERANCE_PCT
                  ? "green"
                  : "amber"
            }
            hint={
              validation?.nose_speed_measured_mm_s != null
                ? `inferred ${validation.nose_speed_inferred_mm_s?.toFixed(1)} vs measured ${validation.nose_speed_measured_mm_s.toFixed(0)} mm·s⁻¹`
                : "no measured reference for these conditions"
            }
          />
        </div>
        <p className="verdict-narr">{narrative(run, validation, labelOf)}</p>
      </Panel>

      <Panel title="At a glance" subtitle="jump to the evidence">
        <div className="glance-chips">
          {glance.map((item) => (
            <button
              key={item.tab}
              type="button"
              className="glance-chip"
              onClick={() => onOpenTab(item.tab)}
            >
              {item.label} →
            </button>
          ))}
        </div>
      </Panel>
    </>
  );
}
