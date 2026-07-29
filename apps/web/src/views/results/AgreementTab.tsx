import { useEffect, useState } from "react";

import { Chip, Panel, Stat, ViewCanvas } from "../../components";
import {
  IouDotChart,
  type IouFramePoint,
  type IouRole,
} from "../../components/charts/IouDotChart";
import {
  api,
  type DimensionlessGroups,
  type PhysicsValidation,
  type RunMetrics,
  type RunSummary,
} from "../../lib/api";
import { fmtIou, runConditions } from "./format";

/** Groups worth an envelope row, in display order (subset of the conditioning
 * vector; only rows present in every dataset's groups render). */
const ENVELOPE_GROUPS = ["Re", "We", "Ca", "Ja", "Pr", "hele_shaw"] as const;

interface PerFrameBlock {
  name: string;
  mean: number | null;
  frames: IouFramePoint[];
  heldout: boolean;
}

function toPoints(
  perFrame: Record<string, number>,
  role: (frame: number) => IouRole,
): IouFramePoint[] {
  return Object.entries(perFrame)
    .map(([key, iou]) => ({ frame: Number(key), iou, role: role(Number(key)) }))
    .sort((a, b) => a.frame - b.frame);
}

/** Every dataset's per-frame agreement, joint or single. */
function agreementBlocks(
  run: RunSummary,
  metrics: RunMetrics | null,
  validation: PhysicsValidation | null,
): PerFrameBlock[] {
  if (validation?.per_dataset) {
    const trained = Object.entries(validation.per_dataset).map(
      ([name, block]) => {
        const validationFrames = new Set(block.validation_frames);
        return {
          name,
          mean: block.iou_mean,
          heldout: false,
          frames: toPoints(block.iou_per_frame, (frame) =>
            validationFrames.has(frame) ? "validation" : "supervised",
          ),
        };
      },
    );
    const transfer = Object.entries(validation.transfer_per_frame ?? {}).map(
      ([name, perFrame]) => ({
        name,
        mean: validation.transfer_per_dataset?.[name] ?? null,
        heldout: true,
        frames: toPoints(perFrame, () => "transfer" as IouRole),
      }),
    );
    return [...trained, ...transfer];
  }
  if (!metrics?.iou_per_frame) return [];
  const validationFrames = new Set(metrics.validation_frames ?? []);
  const holdout = metrics.holdout_frame ?? null;
  return [
    {
      name: run.dataset ?? run.id,
      mean: metrics.iou_mean ?? null,
      heldout: false,
      frames: toPoints(metrics.iou_per_frame, (frame) =>
        frame === holdout
          ? "holdout"
          : validationFrames.has(frame)
            ? "validation"
            : "supervised",
      ),
    },
  ];
}

interface EnvelopeProps {
  training: string[];
  heldout: string[];
  labelOf: (id: string) => string;
}

/** Where each held-out condition sits inside the training envelope of the
 * conditioning groups — the credibility argument for the transfer number. */
function ConditioningEnvelope({ training, heldout, labelOf }: EnvelopeProps) {
  const [groups, setGroups] = useState<Map<string, DimensionlessGroups> | null>(
    null,
  );
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let mounted = true;
    const names = [...training, ...heldout];
    Promise.all(names.map((name) => api.getDatasetGroups(name)))
      .then((all) => {
        if (mounted) setGroups(new Map(names.map((n, i) => [n, all[i]])));
      })
      .catch(() => {
        if (mounted) setFailed(true);
      });
    return () => {
      mounted = false;
    };
  }, [training, heldout]);

  if (failed)
    return (
      <p className="state-note">
        Conditioning groups unavailable for one of the series; re-run its
        preprocessing to compare envelopes.
      </p>
    );
  if (!groups) return <p className="state-note">Loading group envelopes…</p>;

  const rows = ENVELOPE_GROUPS.filter((key) =>
    [...groups.values()].every((g) => typeof g[key] === "number"),
  );
  const fmt = (v: number) =>
    v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(2) : v.toExponential(1);

  return (
    <div>
      {rows.map((key) => {
        const trainValues = training.map((name) => groups.get(name)![key]);
        const lo = Math.min(...trainValues);
        const hi = Math.max(...trainValues);
        const held = heldout.map((name) => ({
          name,
          value: groups.get(name)![key],
        }));
        const min = Math.min(lo, ...held.map((h) => h.value));
        const max = Math.max(hi, ...held.map((h) => h.value));
        const pad = (max - min) * 0.25 + Math.abs(max) * 0.02 + 1e-12;
        const pct = (v: number) =>
          `${(((v - min + pad) / (max - min + 2 * pad)) * 100).toFixed(1)}%`;
        return (
          <div className="envrow" key={key}>
            <span className="env-key">
              {key}
              <small>held-out {held.map((h) => fmt(h.value)).join(", ")}</small>
            </span>
            <span className="env-track">
              <span className="env-base" />
              <span
                className="env-range"
                style={{
                  left: pct(lo),
                  width: `calc(${pct(hi)} - ${pct(lo)})`,
                }}
                title={`training envelope ${fmt(lo)}–${fmt(hi)}`}
              />
              {trainValues.map((value, i) => (
                <span
                  key={i}
                  className="env-tick"
                  style={{ left: pct(value) }}
                  title={`${labelOf(training[i])}: ${fmt(value)}`}
                />
              ))}
              {held.map((h) => (
                <span
                  key={h.name}
                  className="env-held"
                  style={{ left: pct(h.value) }}
                  title={`${labelOf(h.name)}: ${fmt(h.value)}`}
                />
              ))}
            </span>
            {held.every((h) => h.value >= lo && h.value <= hi) ? (
              <Chip tone="green">inside envelope</Chip>
            ) : (
              <Chip tone="amber">extrapolated</Chip>
            )}
          </div>
        );
      })}
    </div>
  );
}

interface AgreementTabProps {
  run: RunSummary;
  metrics: RunMetrics | null;
  validation: PhysicsValidation | null;
  datasetLabels: Map<string, string>;
}

/** Per-condition agreement with the measured bubble, and — for joint runs
 * with a held-out condition — the transfer evidence and its credibility. */
export function AgreementTab({
  run,
  metrics,
  validation,
  datasetLabels,
}: AgreementTabProps) {
  const labelOf = (id: string) => datasetLabels.get(id) ?? id;
  const blocks = agreementBlocks(run, metrics, validation);
  const { all, heldout } = runConditions(run);
  const training = all.filter((id) => !heldout.has(id));
  const transferMean = validation?.transfer_iou_mean ?? null;

  return (
    <>
      <Panel
        title="Agreement per frame"
        subtitle="IoU vs segmented experiment · α > 0.5"
      >
        {blocks.length === 0 ? (
          <p className="state-note">
            No per-frame agreement recorded; re-run the evaluate stage.
          </p>
        ) : (
          <>
            <div className="agree-grid" data-testid="agreement-grid">
              {blocks.map((block) => (
                <div
                  key={block.name}
                  className={"agree-mini" + (block.heldout ? " held" : "")}
                >
                  <div className="agree-mini-head">
                    <span>
                      {labelOf(block.name)}
                      {block.heldout && (
                        <Chip tone="amber">held-out condition</Chip>
                      )}
                    </span>
                    <span className="agree-mean">
                      mean {fmtIou(block.mean)}
                    </span>
                  </div>
                  <ViewCanvas>
                    <IouDotChart
                      frames={block.frames}
                      mean={block.mean}
                      ariaLabel={`Per-frame IoU for ${labelOf(block.name)}`}
                    />
                  </ViewCanvas>
                </div>
              ))}
            </div>
            <div className="legend">
              <span className="li">
                <i className="sw-dot supervised" /> supervised frame
              </span>
              <span className="li">
                <i className="sw-dot validation" /> validation frame · axis A
              </span>
              <span className="li">
                <i className="sw-dot hold" /> holdout frame
              </span>
              <span className="li">
                <i className="sw-dot transfer" /> transfer · never trained
              </span>
            </div>
            <p className="figcap">
              <b>Figure 6.</b> Validation frames are each condition's held-out
              tail — extrapolation in time, the honest split. A held-out
              condition's frames were never trained at all.
            </p>
          </>
        )}
      </Panel>

      {heldout.size > 0 ? (
        <Panel
          title="Transfer to unseen conditions — axis B"
          subtitle={`${[...heldout].map(labelOf).join(", ")} · predicted from dimensionless groups alone`}
        >
          <div className="transfer-grid">
            <div>
              <Stat
                label="Transfer IoU · all frames"
                value={fmtIou(transferMean)}
                tone={transferMean != null ? "green" : "default"}
                hint="every frame of the held-out condition(s)"
              />
              {validation?.val_iou_mean != null && transferMean != null && (
                <Stat
                  label="vs in-distribution"
                  value={`−${((validation.val_iou_mean - transferMean) * 100).toFixed(1)}`}
                  unit="IoU pts"
                  hint="the cost of leaving the condition out"
                />
              )}
            </div>
            <div>
              <h3 className="env-title">
                Conditioning envelope{" "}
                <span className="env-sub">held-out ◆ vs training range</span>
              </h3>
              <ConditioningEnvelope
                training={training}
                heldout={[...heldout]}
                labelOf={labelOf}
              />
            </div>
          </div>
          <p className="figcap">
            <b>Figure 7.</b> Transfer is credible where the held-out condition
            sits inside the training envelope of the conditioning groups;
            extrapolated groups temper the claim. This panel exists because the
            model is conditioned on physical groups, not a per-dataset
            embedding.
          </p>
        </Panel>
      ) : (
        <Panel
          title="Transfer to unseen conditions — axis B"
          subtitle="conditioning generalization"
        >
          <div className="res-empty">
            <b>No condition was held out of this run</b>
            {all.length > 1
              ? "Every dataset trained. To claim transfer, hold one out in the Solver (heldout_datasets)."
              : "Transfer validation needs a joint run spanning several conditions."}
          </div>
        </Panel>
      )}
    </>
  );
}
