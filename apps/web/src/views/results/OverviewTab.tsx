import type { ReactNode } from "react";

import { Panel, Stat, ViewCanvas } from "../../components";
import { ChartFrame } from "../../components/ChartFrame";
import {
  IouDotChart,
  type IouFramePoint,
} from "../../components/charts/IouDotChart";
import type { PhysicsValidation, RunDetail, RunSummary } from "../../lib/api";
import { fmtIou, runConditions } from "./format";
import { physicsVerdict, unsupervisedIou } from "./physicsChecks";
import type { ResultTabId } from "./ResultsPage";

interface OverviewTabProps {
  run: RunSummary;
  detail: RunDetail | null;
  validation: PhysicsValidation | null;
  /** The validation payload is still in flight; no verdict is stated until it
   * lands, in the summary or in the tab it summarises. */
  validationLoading: boolean;
  datasetLabels: Map<string, string>;
  onOpenTab: (tab: ResultTabId) => void;
}

interface FrameScore {
  frame: number;
  iou: number;
  held: boolean;
}

/** Every evaluated frame with the role it played, from what the evaluator wrote. */
function frameScores(
  detail: RunDetail | null,
  validation: PhysicsValidation | null,
): FrameScore[] {
  const perFrame = (detail?.metrics?.["iou_per_frame"] ?? null) as Record<
    string,
    number
  > | null;
  if (!perFrame) return [];
  const held = new Set(validation?.validation_frames ?? []);
  return Object.entries(perFrame)
    .map(([key, iou]) => ({
      frame: Number(key),
      iou,
      held: held.has(Number(key)),
    }))
    .sort((a, b) => a.frame - b.frame);
}

const mean = (values: number[]) =>
  values.length ? values.reduce((sum, v) => sum + v, 0) / values.length : null;

/**
 * The verdict this run's own numbers support.
 *
 * Every branch reads fields the run actually has. The single-condition sentence
 * used to read `iou_holdout` and `holdout_frame` — both null since the Solver
 * retired the single-frame holdout — and rendered, literally, "agreement on the
 * never-supervised frame — is —".
 */
function narrative(
  run: RunSummary,
  validation: PhysicsValidation | null,
  scores: FrameScore[],
  labelOf: (id: string) => string,
): ReactNode {
  const { all, heldout } = runConditions(run);
  const axisA = unsupervisedIou(validation);
  const transfer = validation?.transfer_iou_mean ?? null;
  const trained = all.filter((id) => !heldout.has(id));

  if (transfer != null) {
    const held = [...heldout].map(labelOf).join(", ");
    return (
      <>
        Trained on <N>{trained.length}</N> condition
        {trained.length === 1 ? "" : "s"}, this run reconstructs unseen instants
        of those conditions to IoU <N>{fmtIou(axisA)}</N>, and an entirely
        unseen condition ({held}, supplied only as dimensionless groups) to IoU{" "}
        <N>{fmtIou(transfer)}</N>. The transfer number is the evidence that the
        model learned physics, not footage.
      </>
    );
  }
  if (all.length > 1) {
    return (
      <>
        Trained jointly on all <N>{all.length}</N> conditions, this run
        reconstructs unseen instants to IoU <N>{fmtIou(axisA)}</N>. No condition
        was held out, so transfer (axis B) is untested. Hold one out in the
        Solver to claim it.
      </>
    );
  }

  const supervised = scores.filter((score) => !score.held);
  const held = scores.filter((score) => score.held);
  if (held.length === 0 || supervised.length === 0) {
    return (
      <>
        A single-condition run over <N>{scores.length}</N> evaluated frame
        {scores.length === 1 ? "" : "s"}, every one of them supervised. Nothing
        here is evidence of generalization: set a validation split in the Solver
        to hold the tail of the series back.
      </>
    );
  }
  const supMean = mean(supervised.map((s) => s.iou))!;
  const worst = held.reduce((low, s) => (s.iou < low.iou ? s : low), held[0]);
  return (
    <>
      A single-condition run. On the <N>{supervised.length}</N> frames it was
      shown it agrees to <N>{fmtIou(supMean)}</N>; on the <N>{held.length}</N>{" "}
      it never saw it holds <N>{fmtIou(axisA)}</N>, falling to{" "}
      <N>{fmtIou(worst.iou)}</N> at frame <N>{worst.frame}</N>, the furthest
      extrapolation in the series. That gap of{" "}
      <N>{(supMean - worst.iou).toFixed(3)}</N> is this run&apos;s claim about
      generalization in time.
    </>
  );
}

/** A number inside the verdict, tagged so the sentence can be skimmed for its
 * quantities without reading it. */
function N({ children }: { children: ReactNode }) {
  return <b className="num">{children}</b>;
}

/**
 * The landing tab: the evidence itself.
 *
 * It used to be a scorecard whose four slots a single-condition run could fill
 * one of, a verdict sentence that rendered its own missing values, and a card of
 * links to tabs already on screen — over five hundred pixels of empty canvas.
 * The per-frame agreement is the answer to this stage's question, so it leads;
 * the axes this run has no measurement for say what would supply one.
 */
export function OverviewTab({
  run,
  detail,
  validation,
  validationLoading,
  datasetLabels,
  onOpenTab,
}: OverviewTabProps) {
  const labelOf = (id: string) => datasetLabels.get(id) ?? id;
  const { all, heldout } = runConditions(run);
  const axisA = unsupervisedIou(validation);
  const transfer = validation?.transfer_iou_mean ?? null;
  const noseErr = validation?.nose_speed_error_pct ?? null;
  const scores = frameScores(detail, validation);
  const seriesName =
    all.length === 1 ? labelOf(all[0]) : `${all.length} conditions`;
  const verdict = physicsVerdict(run, validation, seriesName);

  const points: IouFramePoint[] = scores.map((score) => ({
    frame: score.frame,
    iou: score.iou,
    role: score.held ? "validation" : "supervised",
  }));
  const worstFirst = [...scores].sort((a, b) => a.iou - b.iou).slice(0, 3);

  return (
    <div className="overview">
      <Panel
        title="Agreement per frame"
        subtitle="IoU · measured masks vs model"
      >
        {points.length > 0 ? (
          <>
            <ChartFrame
              name={`${run.id}-agreement`}
              title="Per-frame agreement"
              rows={scores as unknown as Record<string, unknown>[]}
              render={() => (
                <ViewCanvas>
                  <IouDotChart
                    frames={points}
                    mean={validation?.iou_mean ?? run.iou_mean ?? null}
                    width={470}
                    ariaLabel={`IoU for ${points.length} frames; ${
                      points.filter((p) => p.role === "validation").length
                    } were never supervised.`}
                  />
                </ViewCanvas>
              )}
            />
            <p className="verdict-narr">
              {narrative(run, validation, scores, labelOf)}
            </p>
            <div className="worst-frames">
              {worstFirst.map((score) => (
                <button
                  key={score.frame}
                  type="button"
                  className={score.held ? "worst-frame held" : "worst-frame"}
                  onClick={() => onOpenTab("recon")}
                >
                  <span>
                    Frame {score.frame}
                    <small>
                      {score.held
                        ? "validation · never supervised"
                        : "supervised"}
                    </small>
                  </span>
                  <b>{fmtIou(score.iou)}</b>
                </button>
              ))}
            </div>
          </>
        ) : (
          <p className="state-note">
            No per-frame agreement recorded; re-run the evaluate stage to
            measure it.
          </p>
        )}
      </Panel>

      <div className="overview-side">
        <Panel title="Scorecard" subtitle="what was measured">
          <div data-testid="overview-scorecard" className="scorecard">
            <Stat
              label="Validation IoU · axis A"
              value={fmtIou(axisA)}
              tone={axisA != null ? "green" : "default"}
              hint={
                validation?.validation_frames?.length
                  ? `frames ${validation.validation_frames.join(", ")}, never supervised`
                  : "held-out frames of trained conditions"
              }
            />
            <Stat
              label="Mean IoU"
              value={fmtIou(validation?.iou_mean ?? run.iou_mean)}
              hint={`all ${run.n_frames ?? scores.length} evaluated frames`}
            />
          </div>
          {/* An axis with no measurement is a configuration this run did not
              have, so it says what would supply it instead of printing a dash. */}
          {transfer != null ? (
            <div className="scorecard">
              <Stat
                label="Transfer IoU · axis B"
                value={fmtIou(transfer)}
                tone="green"
                hint={`all frames of ${[...heldout].map(labelOf).join(", ")} · never trained`}
              />
            </div>
          ) : (
            <p className="unmeasured">
              <b>Transfer IoU · axis B</b>
              Not measured: this run trained on{" "}
              {all.length === 1 ? "one condition" : "every condition it loaded"}
              . Hold one out in the Solver to test it.
            </p>
          )}
          {noseErr == null && (
            <p className="unmeasured">
              <b>Nose-speed error</b>
              Inferred{" "}
              <span className="mono">
                {validation?.nose_speed_inferred_mm_s?.toFixed(1) ?? "—"} mm·s⁻¹
              </span>
              , with no measured value recorded for {seriesName}.
            </p>
          )}
        </Panel>

        <Panel
          title="Physics"
          subtitle={
            validationLoading
              ? "reading"
              : verdict.measured.length === 0
                ? "not measured"
                : `${verdict.flags.length} of ${verdict.measured.length} outside tolerance`
          }
        >
          {validationLoading ? (
            <p className="state-note" role="status">
              Reading this run&apos;s measurements…
            </p>
          ) : verdict.measured.length === 0 ? (
            <p className="state-note">
              No physics diagnostics recorded for this run.
            </p>
          ) : verdict.flags.length === 0 ? (
            <p className="state-note">
              Every measured check is inside tolerance.
            </p>
          ) : (
            <div className="checks-brief">
              {verdict.flags.map((check) => (
                <button
                  key={check.id}
                  type="button"
                  className="check-brief"
                  onClick={() => onOpenTab("physics")}
                >
                  <span>
                    {check.name}
                    <small>{check.detail}</small>
                  </span>
                  <b>{check.value}</b>
                </button>
              ))}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
