import { Panel } from "../../components";
import type { RunSummary } from "../../lib/api";
import { runConditions } from "./format";
import { KinematicsPanel } from "./KinematicsPanel";
import { ReconstructionPanel } from "./ReconstructionPanel";

interface ReconTabProps {
  run: RunSummary;
  viewDataset: string | null;
  datasetLabels: Map<string, string>;
}

/**
 * The reconstruction evidence: the continuous interface player and the growth
 * kinematics it implies. Joint runs are scoped to the viewing condition once
 * per-condition artifacts exist (T6/T7); until then they state that honestly.
 */
export function ReconTab({ run, viewDataset, datasetLabels }: ReconTabProps) {
  const { all } = runConditions(run);
  const joint = all.length > 1;
  const scope = joint
    ? (datasetLabels.get(viewDataset ?? "") ?? viewDataset ?? "—")
    : (run.dataset ?? "—");

  return (
    <>
      <Panel
        title="Continuous reconstruction"
        subtitle={`${scope} · evaluated from ckpt.pt`}
      >
        {joint ? (
          <p className="state-note">
            Per-condition reconstruction for joint runs is coming with the
            per-dataset artifact support; single-condition runs already play
            here.
          </p>
        ) : (
          <ReconstructionPanel runId={run.id} />
        )}
      </Panel>
      <KinematicsPanel
        runId={run.id}
        dataset={joint ? viewDataset : undefined}
      />
    </>
  );
}
