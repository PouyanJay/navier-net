import { Panel } from "../../components";
import type { PhysicsValidation, RunDetail, RunSummary } from "../../lib/api";
import { runConditions } from "./format";
import { FrameMatchPanel } from "./FrameMatchPanel";
import { KinematicsPanel } from "./KinematicsPanel";
import { ReconstructionPanel } from "./ReconstructionPanel";
import type { RunCapability } from "./runCapability";
import { NotReplayable } from "./StateNote";

interface ReconTabProps {
  run: RunSummary;
  detail: RunDetail | null;
  validation: PhysicsValidation | null;
  viewDataset: string | null;
  datasetLabels: Map<string, string>;
  capability: RunCapability;
}

/**
 * The reconstruction evidence: the continuous interface player and the growth
 * kinematics it implies.
 *
 * The player rebuilds the network to sample the interface at arbitrary times,
 * so it needs the run's config snapshot; the kinematics read what the evaluate
 * stage wrote and need nothing. A run that cannot be replayed therefore keeps
 * this tab's second half, and says once at the top why it lost the first.
 */
export function ReconTab({
  run,
  detail,
  validation,
  viewDataset,
  datasetLabels,
  capability,
}: ReconTabProps) {
  const { all } = runConditions(run);
  const joint = all.length > 1;
  const scopeId = joint ? viewDataset : (run.dataset ?? viewDataset);
  const scope = scopeId ? (datasetLabels.get(scopeId) ?? scopeId) : "—";

  return (
    <>
      <Panel
        title="Continuous reconstruction"
        subtitle={`${scope} · evaluated from ckpt.pt`}
      >
        {!capability.replayable ? (
          <NotReplayable runId={run.id} />
        ) : joint ? (
          <p className="state-note">
            Per-condition reconstruction for joint runs is coming with the
            per-dataset artifact support; single-condition runs already play
            here.
          </p>
        ) : (
          <ReconstructionPanel runId={run.id} />
        )}
      </Panel>
      {capability.replayable && (
        <FrameMatchPanel
          runId={run.id}
          dataset={joint ? viewDataset : (run.dataset ?? viewDataset)}
          datasetName={scope}
          metrics={detail?.metrics ?? null}
          validation={validation}
          pinnAvailable={!joint}
        />
      )}
      <KinematicsPanel
        runId={run.id}
        dataset={joint ? viewDataset : undefined}
      />
    </>
  );
}
